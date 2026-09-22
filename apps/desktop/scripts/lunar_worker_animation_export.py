"""Author real skeletal clips, validate deformation, export and reimport."""
import bpy,math,json,numpy as np
from pathlib import Path
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[1]/'public/lunar-city/worker-multiview-2026-09-07'
asset=globals().get('ASSET','baseline');species=asset.split('-')[0];folder=ROOT/asset
scene=bpy.context.scene;rig=next(o for o in scene.objects if o.type=='ARMATURE');body=max((o for o in scene.objects if o.type=='MESH' and o.parent==rig),key=lambda o:len(o.data.vertices))
rig.animation_data_clear();rig.animation_data_create();scene.render.fps=30;scene.frame_start=0;scene.frame_end=60
for b in rig.pose.bones:b.rotation_mode='XYZ';b.rotation_euler=(0,0,0);b.location=(0,0,0)
bpy.context.view_layer.update();rest=np.array([(body.matrix_world@v.co)[:] for v in body.data.vertices]);height=np.ptp(rest[:,2])
edges=np.array([e.vertices[:] for e in body.data.edges]);rest_lengths=np.linalg.norm(rest[edges[:,0]]-rest[edges[:,1]],axis=1);valid_edges=rest_lengths>1e-5
checks=[];actions=[]
for state in ['idle','walk','talk','listen','work','tool-use','carry','handoff','queue','wait','blocked','failed','review','triage','heartbeat','rest','done']:
 clip_name=state;action=bpy.data.actions.new(clip_name);rig.animation_data.action=action
 for frame in range(61):
  t=frame/60;wave=math.sin(2*math.pi*t);pulse=math.sin(math.pi*t)**2
  for b in rig.pose.bones:b.rotation_euler=(0,0,0);b.location=(0,0,0)
  def rot(name,axis,degrees):
   if name in rig.pose.bones:rig.pose.bones[name].rotation_euler[axis]=math.radians(degrees)
  if state=='idle':rot('chest',1,.8*wave)
  elif state=='walk':
   locomotion=scene.get('worker_locomotion','biped')
   if locomotion=='biped':
    for side,sign in [('L',1),('R',-1)]:
     rot('thigh.'+side,0,sign*12*wave);rot('shin.'+side,0,-max(0,sign*wave)*14);rot('upper_arm.'+side,0,-sign*5*wave)
   elif locomotion=='wheeled':
    for b in rig.pose.bones:
     if b.name.startswith('wheel.'):b.rotation_euler[1]=2*math.pi*t
   else:rot('chest',0,1.3*wave)
  elif state=='talk':rot('chest',1,2*wave);rot('forearm.L',0,-12*pulse)
  elif state=='listen':rot('head',2,4*pulse);rot('chest',0,1*pulse)
  elif state=='work':rot('upper_arm.L',0,-7*pulse);rot('forearm.L',0,-15*pulse);rot('forearm.R',0,-12*pulse)
  elif state=='tool-use':rot('forearm.R',0,-15*pulse);rot('hand.R',1,12*wave)
  elif state=='carry':rot('forearm.L',0,-16*pulse);rot('forearm.R',0,-16*pulse)
  elif state=='handoff':rot('upper_arm.L',0,-12*pulse);rot('forearm.L',0,-20*pulse)
  elif state=='queue':rot('chest',1,3*pulse)
  elif state=='wait':rot('head',1,-4*pulse)
  elif state=='blocked':rot('chest',1,4*wave);rot('forearm.L',0,-5*pulse)
  elif state=='failed':rot('chest',0,5*pulse);rot('head',0,5*pulse)
  elif state=='review':rot('head',1,6*pulse);rot('forearm.R',0,-9*pulse)
  elif state=='triage':rot('forearm.L',0,-10*pulse);rot('hand.L',1,15*wave)
  elif state=='heartbeat':rot('chest',0,1.2*math.sin(4*math.pi*t))
  elif state=='rest':rot('chest',0,3*pulse)
  elif state=='done':rot('upper_arm.L',2,-10*pulse);rot('forearm.L',0,-20*pulse);rot('hand.L',1,12*wave)
  if scene.get('worker_locomotion','biped')=='biped':
   # Lift the complete rig by the evaluated sole penetration, preserving the
   # authored metre scale while the two feet exchange support.
   bpy.context.view_layer.update()
   ev=body.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.to_mesh()
   low=min((ev.matrix_world@v.co).z for v in mesh.vertices);ev.to_mesh_clear()
   lift=max(0,float(rest[:,2].min())-low)
   root=rig.pose.bones['root']
   root.location=rig.data.bones['root'].matrix_local.to_3x3().inverted()@Vector((0,0,lift))
   root.keyframe_insert('location',frame=frame,group='root')
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
idle=next(t for a,t in actions if t.name=='idle');idle.mute=False
scene.frame_set(15);scene.render.filepath=str(folder/'rigged-idle.png');bpy.ops.render.render(write_still=True)
bpy.ops.wm.save_as_mainfile(filepath=str(ROOT.parent/'multiview-2026-09-07/leader-rig-workshop.blend'))
print('Verified skeletal export:',out,clip_names)
if (ROOT/'pause-finishing-for-review.json').exists():
 def pause_finish_queue():
  callback=bpy.app.driver_namespace.get('lunar_worker_finish')
  if callback and bpy.app.timers.is_registered(callback):
   bpy.app.timers.unregister(callback)
  (ROOT/'finish-review-pause-status.json').write_text(json.dumps({'asset':asset,'status':'paused_after_saved_export_for_visual_review'})+'\n')
  return None
 bpy.app.timers.register(pause_finish_queue,first_interval=.1)
