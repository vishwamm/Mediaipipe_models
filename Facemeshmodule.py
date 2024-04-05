import cv2
import mediapipe as mp
import time

class facemeshdetector():
    def __init__(self,staticmode=False,maxfaces=2):
        self.staticmode=staticmode
        self.maxfaces=maxfaces
        self.mpfacemesh=mp.solutions.face_mesh
        self.Facemesh=self.mpfacemesh.FaceMesh(self.staticmode,self.maxfaces)
        self.mpDraw=mp.solutions.drawing_utils
        self.drawspec=self.mpDraw.DrawingSpec(thickness=1,circle_radius=1)


    def findmeshfaces(self,image,draw=True):
        faces=[]
        self.imgRGB=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results=self.Facemesh.process(self.imgRGB)
        if self.results.multi_face_landmarks:
            for facelms in self.results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(image,facelms,self.mpfacemesh.FACEMESH_CONTOURS,self.drawspec,self.drawspec)#FACEMESH_TESSELATION for connections 
                face=[]
                for id,lm in enumerate(facelms.landmark):
                    h,w,c=image.shape
                    x,y=int(lm.x*w),int(lm.y*h)
                    #cv2.putText(image,str(id),(x,y),cv2.FONT_HERSHEY_PLAIN,0.5,(255,0,255),1)
                    face.append([x,y])
                faces.append(face)
        return image,faces

def main():
    cap=cv2.VideoCapture(0)
    cTime=0
    pTime=0
    detector=facemeshdetector()
    while True:
        success,image=cap.read()
        image,faces=detector.findmeshfaces(image)
        if len(faces)!=0:
            print(faces[0])
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
        #image,fps,size,font,scale,color,thickness
        cv2.putText(image,"FPS:"+str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),4)
        cv2.imshow("Image", image)
        cv2.waitKey(1)

if __name__=="__main__":
    main()