"""Fit and deformation-check the first character rig in a separate Blender scene."""
import bpy, math, json, bmesh
from pathlib import Path
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[1]/'public/lunar-city/worker-multiview-2026-09-07'
WORKSHOP=ROOT.parent/'multiview-2026-09-07/leader-rig-workshop.blend'
asset=globals().get('ASSET','baseline');f=ROOT/asset
bipeds={'baseline','acceptance-release','engineering-guild'}
hover={'data-performance-repairs','federation-council'}
tracks={'ci-repair-triage','control-plane-incidents','memory-stewardship','pr-merge-train','upstream-hermes-maintenance'}
tripods={'arts-studio','research-review-board'}
locomotion='biped' if asset in bipeds else ('hover' if asset in hover else ('tracked' if asset in tracks else ('tripod' if asset in tripods else 'wheeled')))
head_z,shoulder_z,hip_z,knee_z=(.72,.64,.30,.17)
if asset=='engineering-guild':
 head_z,shoulder_z,hip_z,knee_z=(.72,.65,.42,.22)
source='material-review.glb'
scene=bpy.data.scenes.new('Rig | '+asset);bpy.context.window.scene=scene
scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1
before=set(bpy.data.objects);bpy.ops.import_scene.gltf(filepath=str(f/source))
imported=[o for o in bpy.data.objects if o not in before and o.type=='MESH'];body=max(imported,key=lambda o:len(o.data.vertices));accessories=[o for o in imported if o!=body]
matrix=body.matrix_world.copy();body.parent=None;body.matrix_world=matrix
bpy.context.view_layer.objects.active=body;bpy.ops.object.transform_apply(location=False,rotation=True,scale=True)
body.name=asset+' | skinned body'
# glTF duplicates vertices at UV seams. Weld positions before weight smoothing;
# Blender stores UVs per loop, so the atlas seams remain intact.
bm=bmesh.new();bm.from_mesh(body.data);bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=1e-6);bm.to_mesh(body.data);bm.free();body.data.update()
points=[body.matrix_world@v.co for v in body.data.vertices];h=max(p.z for p in points)
torso=[p.y for p in points if (shoulder_z-.1)*h<p.z<shoulder_z*h];cy=(min(torso)+max(torso))/2
arm_extent=max(abs(p.x) for p in points if .3*h<p.z<shoulder_z*h)/h
neck_z=head_z-.07;chest_z=shoulder_z-.05;spine_z=hip_z+.12
ax=arm_extent*.90;elbow_z=.44;wrist_z=.30
if asset=='acceptance-release':
 ax=.305
for mat in body.data.materials:
 if mat.use_nodes:
  for n in mat.node_tree.nodes:
   if n.type=='NORMAL_MAP':n.inputs['Strength'].default_value=.35
   if n.type=='BSDF_PRINCIPLED':n.inputs['Roughness'].default_value=.62
bpy.ops.object.armature_add(enter_editmode=True,location=(0,0,0));rig=bpy.context.object;rig.name=asset+' | animation rig'
e=rig.data.edit_bones;e.remove(e[0])
def bone(name,start,end,parent=None):
 b=e.new(name);b.head=(start[0]*h,cy+start[1]*h,start[2]*h);b.tail=(end[0]*h,cy+end[1]*h,end[2]*h)
 if parent:b.parent=e[parent]
 return b
bone('root',(0,0,0),(0,0,.1)).use_deform=False
bone('pelvis',(0,0,hip_z),(0,0,spine_z),'root')
bone('spine',(0,0,spine_z),(0,0,chest_z),'pelvis')
bone('chest',(0,0,chest_z),(0,0,neck_z),'spine')
bone('neck',(0,0,neck_z),(0,0,head_z),'chest')
bone('head',(0,0,head_z),(0,0,.96),'neck')
for side,sign in [('L',1),('R',-1)]:
 bone('clavicle.'+side,(0,0,shoulder_z),(sign*ax*.58,0,shoulder_z),'chest')
 bone('upper_arm.'+side,(sign*ax*.58,0,shoulder_z),(sign*ax*.83,-.005,elbow_z),'clavicle.'+side)
 bone('forearm.'+side,(sign*ax*.83,-.005,elbow_z),(sign*ax,-.01,wrist_z),'upper_arm.'+side)
 bone('hand.'+side,(sign*ax,-.01,wrist_z),(sign*ax,-.01,wrist_z-.055),'forearm.'+side)
 leg_x=.145 if asset=='acceptance-release' else .10
 bone('thigh.'+side,(sign*leg_x,0,hip_z),(sign*leg_x,-.015,knee_z),'pelvis')
 bone('shin.'+side,(sign*leg_x,-.015,knee_z),(sign*leg_x,0,.055),'thigh.'+side)
 bone('foot.'+side,(sign*leg_x,0,.055),(sign*leg_x,-.115,.035),'shin.'+side)
if locomotion=='wheeled':
 for side,sign in [('L',1),('R',-1)]:
  for end,depth in [('front',-.12),('rear',.12)]:
   bone('wheel.'+end+'.'+side,(sign*.18,depth,.09),(sign*.20,depth,.09),'root')
if asset in ('acceptance-release','engineering-guild'):
 import numpy as np
 for side,sign in [('L',1),('R',-1)]:
  centers=[]
  for level in [hip_z,knee_z,.055]:
   limit=.235 if asset=='acceptance-release' else .34
   samples=[tuple(p) for p in points if .04*h<p.x*sign<limit*h and abs(p.z/h-level)<.02]
   centers.append(Vector(np.median(samples,axis=0)))
  for name,start,end in zip(['thigh.','shin.'],centers,centers[1:]):
   e[name+side].head=start;e[name+side].tail=end
  e['foot.'+side].head=centers[-1]
  e['foot.'+side].tail=centers[-1]+Vector((0,-.10*h,-.02*h))
if asset=='engineering-guild':
 import numpy as np
 wrist_samples=[tuple(p) for p in points if p.x<-.35*h and p.y<-.20*h and .44*h<p.z<.48*h]
 wrist=Vector(np.median(wrist_samples,axis=0))
 e['forearm.R'].tail=wrist;e['hand.R'].head=wrist
 e['hand.R'].tail=wrist+Vector((-.04*h,-.04*h,-.10*h))
if asset=='ci-repair-triage':
 # Its manipulators project forward at different depths in the side view.
 # Fit each chain in 3D instead of keeping all joints in the torso plane.
 import numpy as np
 for side,sign in [('L',1),('R',-1)]:
  centers=[]
  for level in [.64,.44,.30,.20 if sign>0 else .26]:
   samples=[tuple(p) for p in points if p.x*sign>.30*h and abs(p.z/h-level)<.035]
   centers.append(Vector(np.median(samples,axis=0)))
  e['clavicle.'+side].tail=centers[0]
  for name,start,end in zip(['upper_arm.','forearm.','hand.'],centers,centers[1:]):
   e[name+side].head=start;e[name+side].tail=end
bpy.ops.object.mode_set(mode='OBJECT');rig.show_in_front=True
bpy.ops.object.select_all(action='DESELECT');body.select_set(True);rig.select_set(True);bpy.context.view_layer.objects.active=rig
# Generated overlapping shells do not solve reliably with bone heat. Fit
# normalized anatomical weights explicitly, then inspect deformation visually.
body.parent=rig
mod=body.modifiers.new('Deformation','ARMATURE');mod.object=rig
segments={b.name:(b.head_local.copy(),b.tail_local.copy()) for b in rig.data.bones if b.use_deform}
groups={name:body.vertex_groups.new(name=name) for name in segments}
def distance(p,a,b):
 d=b-a;t=max(0,min(1,(p-a).dot(d)/d.length_squared));return (p-a-t*d).length
for vertex in body.data.vertices:
 p=body.matrix_world@vertex.co;z=p.z/h;x=p.x/h;y=(p.y-cy)/h
 # The release robot has a visible gap between hips (x < .235H) and
 # manipulators. Euclidean proximity must not pull its hip shell with a hand.
 if asset=='ci-repair-triage':
  side='L' if x>=0 else 'R'
  names=['upper_arm.'+side,'forearm.'+side,'hand.'+side]
  if p.y<-.25*h and abs(x)>.25 and z<.38:weights={'hand.'+side:1.0}
  elif (abs(x)<.355 and z<.29) or (abs(x)<.30 and z<.34):weights={'pelvis':1.0}
  elif abs(x)<.31 and z>=.34:weights={'chest':1.0}
  else:
   ranked=sorted((distance(p,*segments[n]),n) for n in names)[:2]
   weights={n:1/max(d,.015*h)**8 for d,n in ranked};total=sum(weights.values());weights={n:w/total for n,w in weights.items()}
 elif asset=='engineering-guild' and x<-.35 and p.y<-.25*h and z<.46:
  # The forward-held tool extends above wrist height; keep its complete shell
  # with the gripper instead of blending the tool head into the upper arm.
  weights={'hand.R':1.0}
 elif asset=='engineering-guild' and ((z<.18 and abs(x)<.37) or (abs(x)<.34 and z<.44)):
  side='L' if x>=0 else 'R';names=['thigh.'+side,'shin.'+side,'foot.'+side,'pelvis']
  ranked=sorted((distance(p,*segments[n]),n) for n in names)[:2]
  weights={n:1/max(d,.02*h)**4 for d,n in ranked};total=sum(weights.values());weights={n:w/total for n,w in weights.items()}
 elif asset=='acceptance-release' and (z<.18 or (abs(x)<.235 and z<.34)):
  side='L' if x>=0 else 'R';names=['thigh.'+side,'shin.'+side,'foot.'+side,'pelvis']
  ranked=sorted((distance(p,*segments[n]),n) for n in names)[:2]
  weights={n:1/max(d,.015*h)**4 for d,n in ranked};total=sum(weights.values());weights={n:w/total for n,w in weights.items()}
 elif asset=='acceptance-release' and z>=.65:
  blend=max(0,min(1,(z-.65)/.07));blend=blend*blend*(3-2*blend)
  weights={'head':blend,'chest':1-blend}
 elif asset=='acceptance-release' and abs(x)<.25 and .34<=z<.65:
  side='L' if x>=0 else 'R'
  names=['upper_arm.'+side,'forearm.'+side,'clavicle.'+side]
  ranked=sorted((distance(p,*segments[n]),n) for n in names)[:2]
  weights={n:1/max(d,.015*h)**4 for d,n in ranked};total=sum(weights.values())
  blend=max(0,min(1,(abs(x)-.19)/.06));blend=blend*blend*(3-2*blend)
  weights={n:w/total*blend for n,w in weights.items()};weights['chest']=1-blend
 elif abs(x)<ax*.62 and z>.30:
  weights={'chest':1.0} if asset in ('baseline','ci-repair-triage','data-performance-repairs','knowledge-commons','engineering-guild') or z<head_z else {'head':1.0}
 elif abs(x)>ax*.72 and z<wrist_z+.04:
  # Low grippers overlap the track/wheel height range. Keep their complete
  # rigid shell on the hand instead of stretching fingertips to the chassis.
  weights={'hand.L' if x>=0 else 'hand.R':1.0}
 elif z<.23 and locomotion!='biped':
  if locomotion=='wheeled' and abs(x)>.10:
   side='L' if x>=0 else 'R';end='front' if y<0 else 'rear';weights={'wheel.'+end+'.'+side:1.0}
  else:weights={'pelvis':1.0}
 else:
  if abs(x)>ax*.55 and z>.18:
   side='L' if x>=0 else 'R';names=['upper_arm.'+side,'forearm.'+side,'hand.'+side,'clavicle.'+side,'chest','pelvis','thigh.'+side]
  elif z<.30:
   side='L' if x>=0 else 'R';names=['thigh.'+side,'shin.'+side,'foot.'+side,'pelvis']
  else:names=['chest']
  ranked=sorted((distance(p,*segments[n]),n) for n in names)[:3]
  weights={n:1/max(d,.015*h)**4 for d,n in ranked};total=sum(weights.values());weights={n:w/total for n,w in weights.items()}
 for name,w in weights.items():
  if w>1e-6:groups[name].add([vertex.index],w,'REPLACE')
# Smooth region boundaries through mesh adjacency, then retain four influences.
import numpy as np
weights=np.zeros((len(body.data.vertices),len(groups)),dtype=np.float32)
for v in body.data.vertices:
 for g in v.groups:weights[v.index,g.group]=g.weight
edges=np.array([e.vertices[:] for e in body.data.edges]);degree=np.bincount(edges.ravel(),minlength=len(weights));degree=np.maximum(degree,1)
for iteration in range(6):
 adjacent=np.zeros_like(weights);np.add.at(adjacent,edges[:,0],weights[edges[:,1]]);np.add.at(adjacent,edges[:,1],weights[edges[:,0]])
 weights=.6*weights+.4*adjacent/degree[:,None]
for v in body.data.vertices:
 row=weights[v.index];keep=np.argsort(row)[-4:];total=row[keep].sum()
 for group in groups.values():group.remove([v.index])
 for i in keep:
  if row[i]>1e-6:body.vertex_groups[int(i)].add([v.index],float(row[i]/total),'REPLACE')
unweighted=[v.index for v in body.data.vertices if not any(g.weight>1e-6 for g in v.groups)]
(f/'rig-weight-check.json').write_text(json.dumps({'method':'fitted anatomical segment weights with connected-mesh smoothing','unweighted':len(unweighted),'vertices':len(body.data.vertices),'bones':len(rig.data.bones)},indent=2)+'\n')
if unweighted:raise RuntimeError(f'Weights left {len(unweighted)} vertices unweighted')
for o in accessories:
 matrix=o.matrix_world.copy();o.parent=rig;o.parent_type='BONE';o.parent_bone='head';bpy.context.view_layer.update();o.matrix_world=matrix
for p in rig.pose.bones:p.rotation_mode='XYZ'
# Deliberate stress pose before creating animation clips.
rig.pose.bones['upper_arm.L'].rotation_euler[2]=math.radians(-25)
rig.pose.bones['forearm.L'].rotation_euler[0]=math.radians(-45)
rig.pose.bones['chest'].rotation_euler[1]=math.radians(5)
if locomotion=='biped':
 rig.pose.bones['thigh.L'].rotation_euler[0]=math.radians(20)
 rig.pose.bones['shin.L'].rotation_euler[0]=math.radians(-25)
bpy.ops.object.camera_add(location=(h*.3,cy-h*2.8,h*1.1));cam=bpy.context.object
cam.rotation_euler=(Vector((0,cy,h*.5))-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.type='ORTHO';cam.data.ortho_scale=h*1.28;scene.camera=cam
for pos,power in [((-2,-3,3),550),((2,-2,2),260),((1,2,3),350)]:
 bpy.ops.object.light_add(type='AREA',location=Vector(pos)*h/1.5);o=bpy.context.object;o.data.energy=power;o.data.size=h*2;o.rotation_euler=(Vector((0,cy,h*.5))-o.location).to_track_quat('-Z','Y').to_euler()
scene.world=bpy.data.worlds.new('Rig studio');scene.world.use_nodes=True;scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.1,.12,.15,1)
scene.world.node_tree.nodes['Background'].inputs[1].default_value=.4
scene.render.engine='CYCLES';scene.cycles.samples=24;scene.cycles.use_denoising=True;scene.view_settings.view_transform='AgX';scene.render.resolution_x=960;scene.render.resolution_y=960;scene.render.resolution_percentage=100
scene.render.filepath=str(f/'rig-stress-pose.png');bpy.ops.render.render(write_still=True)
bpy.ops.wm.save_as_mainfile(filepath=str(WORKSHOP))
print(asset+' weight audit and stress pose ready.')

scene['worker_locomotion']=locomotion
