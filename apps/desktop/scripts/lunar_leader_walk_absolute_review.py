import bpy,json,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'public/lunar-city/multiview-2026-09-07'
results=[]
for asset in ['owl-librarian','elephant-memory','fox-scientist','capybara-revenue','lion-steward','beaver-architect']:
 scene=bpy.data.scenes['Walk contact review | '+asset];bpy.context.window.scene=scene;rig=next(o for o in scene.objects if o.type=='ARMATURE');meshes=[o for o in scene.objects if o.type=='MESH'];walk=next(t for t in rig.animation_data.nla_tracks if ':walk' in t.name)
 for t in rig.animation_data.nla_tracks:t.mute=t!=walk
 def points():
  deps=bpy.context.evaluated_depsgraph_get();parts=[]
  for o in meshes:
   ev=o.evaluated_get(deps);m=ev.to_mesh();parts.append(np.array([(ev.matrix_world@v.co)[:] for v in m.vertices]));ev.to_mesh_clear()
  return np.concatenate(parts)
 scene.frame_set(0);rest=points();edges=[];offset=0
 for o in meshes:
  edges.extend([[e.vertices[0]+offset,e.vertices[1]+offset] for e in o.data.edges]);offset+=len(o.data.vertices)
 edges=np.array(edges);base=np.linalg.norm(rest[edges[:,0]]-rest[edges[:,1]],axis=1);worst=None;max_count=0
 for frame in range(61):
  scene.frame_set(frame);pts=points();length=np.linalg.norm(pts[edges[:,0]]-pts[edges[:,1]],axis=1);delta=length-base;idx=int(delta.argmax());significant=(length>3*base)&(delta>.005);max_count=max(max_count,int(significant.sum()))
  if worst is None or delta[idx]>worst['growthMetres']:worst={'frame':frame,'growthMetres':float(delta[idx]),'restLengthMetres':float(base[idx]),'restMidpoint':((rest[edges[idx,0]]+rest[edges[idx,1]])/2).tolist()}
 result={'asset':asset,'worstAbsoluteGrowth':worst,'maxEdgesOver3xAnd5mmGrowth':max_count};results.append(result)
 (ROOT/asset/'walk-contact-review/absolute-deformation.json').write_text(json.dumps(result,indent=2)+'\n')
(ROOT/'walk-absolute-deformation.json').write_text(json.dumps(results,indent=2)+'\n');print(results)
