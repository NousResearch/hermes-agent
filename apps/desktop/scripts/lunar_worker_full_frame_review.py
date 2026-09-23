"""Check the evaluated worker mesh on every frame of every saved clip."""
import bpy,json,numpy as np
from pathlib import Path
asset=ASSET
folder=Path(__file__).resolve().parents[1]/'public/lunar-city/worker-multiview-2026-09-07'/asset
scene=bpy.context.scene
rig=next(o for o in scene.objects if o.type=='ARMATURE')
body=max((o for o in scene.objects if o.type=='MESH' and o.parent==rig),key=lambda o:len(o.data.vertices))
tracks=list(rig.animation_data.nla_tracks)
original=[t.mute for t in tracks];original_frame=scene.frame_current
for t in tracks:t.mute=True
scene.frame_set(0)
rest=np.array([(body.matrix_world@v.co)[:] for v in body.data.vertices])
edges=np.array([e.vertices[:] for e in body.data.edges]);lengths=np.linalg.norm(rest[edges[:,0]]-rest[edges[:,1]],axis=1);valid=lengths>1e-5
results=[]
try:
 for track in tracks:
  track.mute=False
  samples=[]
  for frame in range(61):
   scene.frame_set(frame)
   ev=body.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.to_mesh()
   p=np.array([(ev.matrix_world@v.co)[:] for v in mesh.vertices]);ev.to_mesh_clear()
   stretch=np.linalg.norm(p[edges[:,0]]-p[edges[:,1]],axis=1)[valid]/lengths[valid]
   samples.append({'frame':frame,'minimumZ':float(p[:,2].min()),'maxEdgeStretch':float(stretch.max()),'edgesOver3x':int(np.count_nonzero(stretch>3))})
  results.append({'clip':track.name,'samples':samples})
  if track.name in ('walk','review','tool-use'):
   scene.frame_set(15 if track.name=='walk' else 30)
   scene.render.filepath=str(folder/('animation-'+track.name+'-review.png'))
   bpy.ops.render.render(write_still=True)
  track.mute=True
finally:
 for t,mute in zip(tracks,original):t.mute=mute
 scene.frame_set(original_frame)
summary={'asset':asset,'sampleCount':sum(len(r['samples']) for r in results),'minimumZ':min(s['minimumZ'] for r in results for s in r['samples']),'maxEdgeStretch':max(s['maxEdgeStretch'] for r in results for s in r['samples']),'maxEdgesOver3x':max(s['edgesOver3x'] for r in results for s in r['samples']),'scope':'Evaluated source Blender rig, every integer frame; not final topology or game runtime acceptance','clips':results}
(folder/'rig-full-frame-validation.json').write_text(json.dumps(summary,indent=2)+'\n')
print({k:v for k,v in summary.items() if k!='clips'})
