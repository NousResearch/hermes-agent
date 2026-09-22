"""Relieve only measured tire-intersecting brace edges in the preserved study."""
import bpy,json,math
from pathlib import Path
from mathutils import Vector
scene=bpy.context.scene
body=next(o for o in scene.objects if o.type=='MESH' and o.name.startswith('Core wheel partition study'))
folder=Path(__file__).resolve().parents[1]/'public/lunar-city/worker-multiview-2026-09-07/core-runtime-ux-repairs'
audit=json.loads((folder/'wheel-clearance-review.json').read_text())
wheels={o.name:o for o in scene.objects if o.type=='EMPTY' and o.name.startswith('wheel.')}
changes={};faces=set()
for row in audit['rows']:
 center=wheels[row['wheel']].location
 for face_index in row['stationaryFaces']:
  faces.add(face_index)
  for index in body.data.polygons[face_index].vertices:
   p=changes.get(index,body.data.vertices[index].co.copy());dy=p.y-center.y;dz=p.z-center.z;radius=math.hypot(dy,dz)
   if abs(p.x-center.x)<.08 and .06<radius<.128:
    p.y=center.y+dy*.128/radius;p.z=center.z+dz*.128/radius;changes[index]=p
maximum=max(((p-body.data.vertices[i].co).length for i,p in changes.items()),default=0)
if maximum>.025 or len(faces)>100:raise ValueError('Relief exceeds local study bounds')
old=body.data;old.use_fake_user=True;body.data=old.copy();body.data.name=old.name+' | measured wheel relief'
for index,p in changes.items():body.data.vertices[index].co=p
body.data.update()
(folder/'wheel-clearance-before-fit.json').write_text(json.dumps(audit,indent=2)+'\n')
(folder/'wheel-clearance-fit.json').write_text(json.dumps({'stage':'Local study relief; full geometry acceptance pending','sourceMeshPreserved':old.name,'affectedFaces':len(faces),'adjustedVertices':len(changes),'maxDisplacementMetres':maximum,'targetRadialReliefMetres':.128},indent=2)+'\n')
print('Brace relief vertices/max displacement:',len(changes),maximum)
