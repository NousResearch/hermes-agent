"""Read GLB accessor bounds with node transforms; no Blender mutation."""
import itertools,json,math,struct
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]/'public/lunar-city'
m=json.loads((ROOT/'v2-review/world-manifest.v2.json').read_text())
def mul(a,b):return [[sum(a[i][k]*b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
I=[[1 if i==j else 0 for j in range(4)]for i in range(4)]
def transform(node):
 if 'matrix' in node:return [[node['matrix'][j*4+i]for j in range(4)]for i in range(4)]
 x,y,z,w=node.get('rotation',[0,0,0,1]);s=node.get('scale',[1,1,1]);t=node.get('translation',[0,0,0])
 r=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
 return [[r[i][j]*s[j]for j in range(3)]+[t[i]]for i in range(3)]+[[0,0,0,1]]
rows=[]
for asset in m['reviewLeaderAssets']:
 b=(ROOT/'v2-review'/asset['uri']).read_bytes();n=struct.unpack_from('<I',b,12)[0];g=json.loads(b[20:20+n]);lo=[math.inf]*3;hi=[-math.inf]*3;radial=[0.0];binary=b[28+n:]
 def visit(index,parent):
  node=g['nodes'][index];matrix=mul(parent,transform(node))
  if 'mesh' in node:
   for primitive in g['meshes'][node['mesh']]['primitives']:
    a=g['accessors'][primitive['attributes']['POSITION']]
    view=g['bufferViews'][a['bufferView']];start=view.get('byteOffset',0)+a.get('byteOffset',0);stride=view.get('byteStride',12)
    for vi in range(a['count']):
     point=struct.unpack_from('<fff',binary,start+vi*stride)
     p=[sum(matrix[i][j]*point[j] for j in range(3))+matrix[i][3]for i in range(3)]
     radial[0]=max(radial[0],math.hypot(p[0],p[2]))
     for i in range(3):lo[i]=min(lo[i],p[i]);hi[i]=max(hi[i],p[i])
  for child in node.get('children',[]):visit(child,matrix)
 for node in g['scenes'][g.get('scene',0)]['nodes']:visit(node,I)
 size=[hi[i]-lo[i]for i in range(3)];radius=radial[0]+.05
 rows.append(dict(id=asset['id'],sourceUri=asset['uri'],boundsMin=lo,boundsMax=hi,size=size,radiusMetres=radius,clearWidth=math.ceil((2*radius+.1)*10)/10))
( ROOT/'interior-plans-v1/actor-clearances.json').write_text(json.dumps(dict(method='Every POSITION vertex transformed through scene node hierarchy at bind/rest state, radial distance about GLB/actor origin XZ0 plus0.05m; includes props and off-centre parts',actors=rows),indent=2)+'\n')
print(json.dumps(rows,indent=2))
