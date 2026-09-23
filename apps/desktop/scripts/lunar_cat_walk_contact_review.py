"""Preserved source-rig walk studies; visible Blender only, no promotion."""
import bpy,json,hashlib,numpy as np
from pathlib import Path
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[1]/'public/lunar-city/multiview-2026-09-07'
assets=['cat-arts']
results=[]
for asset in assets:
 folder=ROOT/asset/'walk-contact-review';folder.mkdir(exist_ok=True)
 scene=bpy.data.scenes.new('Walk contact review | '+asset);bpy.context.window.scene=scene;scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1;scene.render.fps=30
 bpy.ops.import_scene.gltf(filepath=str(ROOT/asset/'rigged-review.glb'))
 rig=next(o for o in scene.objects if o.type=='ARMATURE');meshes=[o for o in scene.objects if o.type=='MESH'];anim=rig.animation_data
 tracks=list(anim.nla_tracks);walk=next(t for t in tracks if ':walk' in t.name);strip=walk.strips[0]
 for t in tracks:t.mute=True
 anim.action=strip.action;anim.action_slot=strip.action_slot
 base=rig.location.copy()
 def points():
  deps=bpy.context.evaluated_depsgraph_get();out=[]
  for o in meshes:
   ev=o.evaluated_get(deps);m=ev.to_mesh();out.append(np.array([(ev.matrix_world@v.co)[:] for v in m.vertices]));ev.to_mesh_clear()
  return np.concatenate(out)
 scene.frame_set(0);bpy.context.view_layer.update();rest=points();height=float(np.ptp(rest[:,2]));lo=rest.min(axis=0);hi=rest.max(axis=0)
 edges=[];offset=0
 for o in meshes:
  edges.extend([[e.vertices[0]+offset,e.vertices[1]+offset] for e in o.data.edges]);offset+=len(o.data.vertices)
 edges=np.array(edges);lengths=np.linalg.norm(rest[edges[:,0]]-rest[edges[:,1]],axis=1);valid=lengths>1e-5
 start=int(strip.action_frame_start);end=int(strip.action_frame_end);checks=[]
 for frame in range(start,end+1):
  scene.frame_set(frame);rig.location=base;bpy.context.view_layer.update();pts=points();rig.location.z+=max(0,-float(pts[:,2].min()))+.0002;rig.keyframe_insert('location',index=2,frame=frame);bpy.context.view_layer.update();pts=points();ratio=np.linalg.norm(pts[edges[:,0]]-pts[edges[:,1]],axis=1)[valid]/lengths[valid]
  checks.append({'frame':frame,'minimumZ':float(pts[:,2].min()),'maximumEdgeStretch':float(ratio.max()),'edgesOver3x':int(np.count_nonzero(ratio>3))})
 anim.action=None
 for t in tracks:t.mute=False
 for o in scene.objects:o.select_set(o.type in {'MESH','ARMATURE'})
 bpy.context.view_layer.objects.active=rig
 output=folder/'rigged-review.glb';bpy.ops.export_scene.gltf(filepath=str(output),export_format='GLB',use_selection=True,use_active_scene=True,export_animations=True,export_animation_mode='NLA_TRACKS',export_skins=True,export_anim_slide_to_zero=True)
 for t in tracks:t.mute=t!=walk
 world=bpy.data.worlds.new('Walk review studio');world.use_nodes=True;world.node_tree.nodes['Background'].inputs[0].default_value=(.12,.15,.18,1);world.node_tree.nodes['Background'].inputs[1].default_value=.5;scene.world=world
 center=Vector(((lo[0]+hi[0])/2,(lo[1]+hi[1])/2,height*.5))
 bpy.ops.object.camera_add(location=center+Vector((height*1.3,-height*2.7,height*.5)));cam=bpy.context.object;cam.rotation_euler=(center-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.type='ORTHO';cam.data.ortho_scale=height*1.45;scene.camera=cam
 for pos,energy,size in [((2,-3,4),500,3),((-2,0,3),350,2)]:
  bpy.ops.object.light_add(type='AREA',location=tuple(c*height for c in pos));light=bpy.context.object;light.data.energy=energy*height*height;light.data.shape='DISK';light.data.size=size*height;light.rotation_euler=(center-light.location).to_track_quat('-Z','Y').to_euler()
 scene.render.engine='CYCLES';scene.cycles.samples=16;scene.render.resolution_x=800;scene.render.resolution_y=800;scene.render.resolution_percentage=100
 for fraction in [.25,.75]:
  frame=round(start+(end-start)*fraction);scene.frame_set(frame);scene.render.filepath=str(folder/f'walk-{frame}.png');bpy.ops.render.render(write_still=True)
 receipt={'asset':asset,'sourceSha256':hashlib.sha256((ROOT/asset/'rigged-review.glb').read_bytes()).hexdigest(),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'stage':'contact corrected copy; visual deformation review required','heightMetres':height,'frames':checks,'clips':[t.name for t in tracks]};(folder/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');results.append({'asset':asset,'maxStretch':max(c['maximumEdgeStretch'] for c in checks),'edgesOver3x':max(c['edgesOver3x'] for c in checks)})
 (ROOT/'cat-walk-contact-status.json').write_text(json.dumps(results,indent=2)+'\n')
bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'leader-rig-workshop.blend'))
print('Cat preserved walk contact candidate saved',results)
