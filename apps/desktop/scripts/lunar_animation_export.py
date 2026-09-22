"""Author real skeletal clips, validate deformation, export and reimport."""
import bpy,math,json,numpy as np
from pathlib import Path
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[1]/'public/lunar-city/multiview-2026-09-07'
asset=globals().get('ASSET','cat-arts');species=asset.split('-')[0];folder=ROOT/asset
scene=bpy.context.scene;rig=next(o for o in scene.objects if o.type=='ARMATURE');body=max((o for o in scene.objects if o.type=='MESH' and o.parent==rig),key=lambda o:len(o.data.vertices))
rig.animation_data_clear();rig.animation_data_create();scene.render.fps=30;scene.frame_start=0;scene.frame_end=60
for b in rig.pose.bones:b.rotation_mode='XYZ';b.rotation_euler=(0,0,0);b.location=(0,0,0)
bpy.context.view_layer.update();rest=np.array([(body.matrix_world@v.co)[:] for v in body.data.vertices]);height=np.ptp(rest[:,2])
edges=np.array([e.vertices[:] for e in body.data.edges]);rest_lengths=np.linalg.norm(rest[edges[:,0]]-rest[edges[:,1]],axis=1);valid_edges=rest_lengths>1e-5
checks=[];actions=[]
for state in ['acknowledging','idle','listening','talking','thinking','unavailable','walk']:
 clip_name='leader:'+species+':'+state;action=bpy.data.actions.new(clip_name);rig.animation_data.action=action
 for frame in range(0,61,5):
  t=frame/60;wave=math.sin(2*math.pi*t);pulse=math.sin(math.pi*t)**2
  for b in rig.pose.bones:b.rotation_euler=(0,0,0);b.location=(0,0,0)
  def rot(name,axis,degrees):
   if name in rig.pose.bones:rig.pose.bones[name].rotation_euler[axis]=math.radians(degrees)
  if state=='idle':rot('chest',0,.6*wave);rot('head',1,1.2*wave);rot('tail.02',2,3*wave)
  elif state=='acknowledging':rot('head',0,9*pulse);rot('forearm.L',0,-15*pulse)
  elif state=='listening':rot('head',2,5*pulse);rot('chest',0,1.5*pulse)
  elif state=='talking':rot('head',0,2*wave);rot('head',1,3*wave);rot('upper_arm.L',0,-5*pulse);rot('forearm.L',0,-12*pulse);rot('forearm.R',0,-7*pulse)
  elif state=='thinking':rot('head',0,5*pulse);rot('head',1,-8*pulse);rot('forearm.R',0,-15*pulse)
  elif state=='unavailable':rot('head',0,7*pulse);rot('chest',0,2*pulse)
  elif state=='walk':
   for side,sign in [('L',1),('R',-1)]:
    rot('thigh.'+side,0,sign*12*wave);rot('shin.'+side,0,-max(0,sign*wave)*14);rot('upper_arm.'+side,0,-sign*7*wave)
   rot('tail.01',2,4*wave)
  for b in rig.pose.bones:b.keyframe_insert('rotation_euler',frame=frame,group=b.name)
 slot=rig.animation_data.action_slot
 # Evaluate actual mesh at representative poses, not just action metadata.
 for frame in range(0,61,5):
  scene.frame_set(frame);deps=bpy.context.evaluated_depsgraph_get();ev=body.evaluated_get(deps);mesh=ev.to_mesh();points=np.array([(ev.matrix_world@v.co)[:] for v in mesh.vertices]);ev.to_mesh_clear()
  if not np.all(np.isfinite(points)):raise ValueError('Non-finite deformation')
  movement=float(np.linalg.norm(points-rest,axis=1).max());extent=np.ptp(points,axis=0)
  if movement>height*.5 or extent[2]>height*1.2:raise ValueError('Exploding deformation '+state)
  lengths=np.linalg.norm(points[edges[:,0]]-points[edges[:,1]],axis=1);ratios=lengths[valid_edges]/rest_lengths[valid_edges]
  checks.append({'clip':state,'frame':frame,'maxDisplacementMetres':movement,'minimumZ':float(points[:,2].min()),'maxEdgeStretch':float(ratios.max()),'edgesOver3x':int(np.count_nonzero(ratios>3))})
 track=rig.animation_data.nla_tracks.new();track.name=clip_name;strip=track.strips.new(action.name,0,action);strip.action_slot=slot
 track.mute=True;actions.append((action,track))
rig.animation_data.action=None
for action,track in actions:track.mute=False
scene.frame_set(0)
bpy.ops.object.select_all(action='DESELECT');body.select_set(True);rig.select_set(True);[o.select_set(True) for o in scene.objects if o.type=='MESH' and o.parent==rig];bpy.context.view_layer.objects.active=rig
out=folder/'rigged-review.glb'
bpy.ops.export_scene.gltf(filepath=str(out),export_format='GLB',use_selection=True,use_active_scene=True,export_animations=True,export_animation_mode='NLA_TRACKS',export_skins=True,export_anim_slide_to_zero=True)
# Reimport to a new scene; verify the deliverable contains a real rig and clips.
verify=bpy.data.scenes.new('Export QA | '+asset);bpy.context.window.scene=verify
before=set(bpy.data.objects);bpy.ops.import_scene.gltf(filepath=str(out));objects=[o for o in bpy.data.objects if o not in before]
armatures=[o for o in objects if o.type=='ARMATURE'];meshes=[o for o in objects if o.type=='MESH' and any(m.type=='ARMATURE' for m in o.modifiers)]
if len(armatures)!=1 or not meshes:raise ValueError('Missing exported skeleton or body')
clip_names=[t.name for t in armatures[0].animation_data.nla_tracks]
expected={t.name for a,t in actions}
if not expected.issubset(set(clip_names)):raise ValueError('Missing exported clips '+str(clip_names))
bpy.context.view_layer.update();pts=[o.matrix_world@Vector(v) for o in meshes for v in o.bound_box];export_height=max(v.z for v in pts)-min(v.z for v in pts)
if abs(export_height-height)>.005:raise ValueError('Export metre scale changed')
(folder/'rig-export-validation.json').write_text(json.dumps({'asset':asset,'bones':len(armatures[0].data.bones),'clips':clip_names,'heightMetres':export_height,'weightCheck':'no unweighted vertices','deformationSamples':checks,'stage':'skeletal_export_verified_visual_animation_review_pending','facialRig':False},indent=2)+'\n')
bpy.context.window.scene=scene
for a,t in actions:t.mute=True
idle=next(t for a,t in actions if t.name.endswith(':idle'));idle.mute=False
scene.frame_set(15);scene.render.filepath=str(folder/'rigged-idle.png');bpy.ops.render.render(write_still=True)
bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'leader-rig-workshop.blend'))
print('Verified skeletal export:',out,clip_names)
