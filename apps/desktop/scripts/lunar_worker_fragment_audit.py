"""Inventory disconnected geometry without deleting any component."""
import bpy,json
from pathlib import Path
scene=bpy.context.scene
body=next(o for o in scene.objects if o.type=='MESH' and any(m.type=='ARMATURE' for m in o.modifiers))
mesh=body.data
neighbors=[[] for _ in mesh.vertices]
for edge in mesh.edges:
 a,b=edge.vertices;neighbors[a].append(b);neighbors[b].append(a)
seen=set();components=[]
for vertex in mesh.vertices:
 if vertex.index in seen:continue
 todo=[vertex.index];seen.add(vertex.index);ids=[]
 while todo:
  i=todo.pop();ids.append(i)
  for j in neighbors[i]:
   if j not in seen:seen.add(j);todo.append(j)
 components.append(ids)
owner={i:c for c,ids in enumerate(components) for i in ids};faces=[0]*len(components)
for face in mesh.polygons:faces[owner[face.vertices[0]]]+=1
rows=[]
for c,ids in enumerate(components):
 if faces[c]>24:continue
 points=[mesh.vertices[i].co for i in ids]
 low=[min(p[k] for p in points) for k in range(3)];high=[max(p[k] for p in points) for k in range(3)]
 rows.append({'component':c,'vertices':len(ids),'faces':faces[c],'min':low,'max':high,'extent':[b-a for a,b in zip(low,high)],'indices':ids})
Path(OUTPUT).write_text(json.dumps({'body':body.name,'componentCount':len(components),'smallComponents':rows},indent=2)+'\n', encoding='utf-8')
print('Components',len(components),'small components',len(rows))
if globals().get('REMOVE_SINGLE_TRIANGLES',False):
 import bmesh
 fragments=[r for r in rows if r['faces']==1 and r['vertices']==3]
 old=body.data;old.use_fake_user=True
 body.data=old.copy();body.data.name=old.name+' | isolated triangle cleanup'
 bm=bmesh.new();bm.from_mesh(body.data);bm.verts.ensure_lookup_table()
 ids={i for r in fragments for i in r['indices']}
 bmesh.ops.delete(bm,geom=[bm.verts[i] for i in ids],context='VERTS')
 bm.to_mesh(body.data);bm.free();body.data.update()
 Path(OUTPUT).with_name('fragment-cleanup.json').write_text(json.dumps({'removedSingleTriangleComponents':len(fragments),'removedVertices':len(ids),'preservedSourceMesh':old.name,'method':'Only disconnected one-triangle islands; closed parts retained'},indent=2)+'\n')
 print('Removed disconnected triangle islands:',len(fragments))
