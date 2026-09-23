"""Source-color projection study; preserves raw geometry and records limitations."""
from pathlib import Path
import json
import sys
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
import trimesh

asset=sys.argv[1] if len(sys.argv)>1 else 'owl-librarian'
FOLDER=Path(sys.argv[2]) if len(sys.argv)>2 else Path(__file__).resolve().parents[1]/'public/lunar-city/multiview-2026-09-07'/asset
mesh=trimesh.load(FOLDER/'shape-metres.glb',force='mesh')
components=trimesh.graph.connected_components(mesh.face_adjacency,nodes=np.arange(len(mesh.faces)),min_len=1)
audit={'sourceFaces':len(mesh.faces),'vertices':len(mesh.vertices),'watertight':mesh.is_watertight,
       'windingConsistent':mesh.is_winding_consistent,'components':len(components),
       'componentFaces':sorted(map(len,components),reverse=True),
       'cleanup':'No components deleted; inspect small islands before removal.'}
(FOLDER/'topology-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
v=mesh.vertices;n=mesh.vertex_normals
low,high=mesh.bounds
xyz=(v-low)/(high-low)
colors=[]
for name in ('front','back','left','top'):
    im=np.array(Image.open(FOLDER/f'{name}.png').convert('RGBA'))
    mask=im[:,:,3]>127
    rows,cols=np.where(mask)
    x0,x1,y0,y1=cols.min(),cols.max(),rows.min(),rows.max()
    if name=='top':u=xyz[:,0]
    elif name in ('front','back'):
        bx0,bx1=low[0],high[0]
        u=(v[:,0]-bx0)/(bx1-bx0)
        if name=='back':u=1-u
    else:u=1-xyz[:,2]
    px=np.clip(np.rint(x0+u*(x1-x0)).astype(int),0,im.shape[1]-1)
    py=np.clip(np.rint(y0+xyz[:,2]*(y1-y0) if name=='top' else y1-xyz[:,1]*(y1-y0)).astype(int),0,im.shape[0]-1)
    nearest=distance_transform_edt(~mask,return_distances=False,return_indices=True)
    invalid=~mask[py,px]
    qy=py.copy();qx=px.copy()
    qy[invalid]=nearest[0,py[invalid],px[invalid]]
    qx[invalid]=nearest[1,py[invalid],px[invalid]]
    colors.append(im[qy,qx,:3]/255.)
# Camera-facing weights: global bounds are not a body centre (tails shift them).
# A rear-facing surface must never receive the front projection.
normals_mesh=mesh.copy()
trimesh.smoothing.filter_laplacian(normals_mesh,lamb=.35,iterations=4)
n=normals_mesh.vertex_normals
wfront=np.maximum(n[:,2],0)**2
wback=np.maximum(-n[:,2],0)**2
# Side artwork contributes near the side, never across the centre of the rear.
wside=np.maximum(n[:,0],0)**4 * (1-np.abs(n[:,2]))**4
wtop=np.maximum(n[:,1],0)**4
weights=np.stack((wfront,wback,wside,wtop),axis=1)
fallback=weights.sum(axis=1)<1e-8
weights[fallback,0]=(n[fallback,2]>=0)
weights[fallback,1]=(n[fallback,2]<0)
weights/=weights.sum(axis=1)[:,None]
assert not np.any(weights[n[:,2]<0,0])
assert not np.any(weights[n[:,2]>0,1])
rgb=np.sum(np.stack(colors,axis=1)*weights[:,:,None],axis=1)
rgb=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
mesh.visual.vertex_colors=np.c_[np.clip(rgb*255,0,255).astype(np.uint8),np.full(len(v),255,dtype=np.uint8)]
mesh.export(FOLDER/'source-color-study-v4.glb')
(FOLDER/'color-study-v4.json').write_text(json.dumps({'stage':'multiview_color_projection_requires_visual_alignment_review',
 'method':'alpha-bounded projection with smoothed camera-facing normals; front and rear weights mutually exclusive',
 'limitations':['reference lighting is embedded','right side is unobserved','no occlusion or landmark registration','not final PBR textures'],
 'source':'shape-metres.glb','output':'source-color-study-v4.glb'},indent=2)+'\n')
print(json.dumps({k:v for k,v in audit.items() if k!='componentFaces'}));print('Largest components:',audit['componentFaces'][:20])
