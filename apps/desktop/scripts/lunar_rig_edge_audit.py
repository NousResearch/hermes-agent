import bpy,json
from pathlib import Path
scene=bpy.context.scene
body=next(o for o in scene.objects if o.type=='MESH' and any(m.type=='ARMATURE' for m in o.modifiers))
bpy.context.view_layer.update();evaluated=body.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=evaluated.to_mesh()
rows=[]
for edge in body.data.edges:
 a,b=edge.vertices;rest=(body.data.vertices[a].co-body.data.vertices[b].co).length
 if rest<1e-5:continue
 changed=(mesh.vertices[a].co-mesh.vertices[b].co).length
 if changed/rest>3:
  rows.append({'ratio':changed/rest,'rest':rest,'posed':changed,'vertices':[{'id':i,'position':list(body.data.vertices[i].co),'weights':{body.vertex_groups[g.group].name:g.weight for g in body.data.vertices[i].groups if g.weight>.001}} for i in [a,b]]})
evaluated.to_mesh_clear()
Path(OUTPUT).write_text(json.dumps(sorted(rows,key=lambda r:r['posed']-r['rest'],reverse=True)[:50],indent=2)+'\n', encoding='utf-8')
print('Edges stretched more than 3x:',len(rows))
