"""Fit and deformation-check the first character rig in a separate Blender scene."""
import bpy, math, json, bmesh
from pathlib import Path
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[1]/'public/lunar-city/multiview-2026-09-07'
asset=globals().get('ASSET','fox-scientist');f=ROOT/asset
config={'owl-librarian':(.62,.52,.20,.12),'elephant-memory':(.66,.59,.25,.14),'fox-scientist':(.77,.66,.31,.20),'capybara-revenue':(.73,.61,.25,.15),'lion-steward':(.73,.61,.27,.16),'beaver-architect':(.72,.60,.24,.14),'monkey-poet':(.77,.65,.30,.18),'cat-arts':(.77,.65,.31,.20)}
head_z,shoulder_z,hip_z,knee_z=config[asset]
source='owl-detail-review.glb' if asset=='owl-librarian' else ('material-review-v4.glb' if asset in ('cat-arts','fox-scientist','beaver-architect','monkey-poet') else 'material-review.glb')
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
ax=arm_extent*.90;elbow_z=shoulder_z-.16;wrist_z=shoulder_z-.25
if asset=='monkey-poet':elbow_z=.45;wrist_z=.32
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
 bone('thigh.'+side,(sign*.10,0,hip_z),(sign*.10,-.015,knee_z),'pelvis')
 bone('shin.'+side,(sign*.10,-.015,knee_z),(sign*.10,0,.055),'thigh.'+side)
 bone('foot.'+side,(sign*.10,0,.055),(sign*.10,-.115,.035),'shin.'+side)
if asset not in ('owl-librarian','elephant-memory'):
 bone('tail.01',(0,.08,hip_z),(0,.24,.16),'pelvis')
 bone('tail.02',(0,.24,.16),(0,.43,.20),'tail.01')
 bone('tail.03',(0,.43,.20),(0,.60,.30),'tail.02')
if asset=='monkey-poet':
 import numpy as np
 for side,sign in [('L',1),('R',-1)]:
  centers=[]
  for level in [.65,.45,.32,.25]:
   samples=[tuple(p) for p in points if p.x*sign>.18*h and abs(p.z/h-level)<.025]
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
monkey_hands={}
if asset=='monkey-poet':
 # Follow connected palm geometry below the cuff. A straight x/depth cut
 # divides curved fingers and incorrectly mixes them with leg influences.
 adjacency=[[] for _ in points]
 for edge in body.data.edges:
  a,b=edge.vertices;adjacency[a].append(b);adjacency[b].append(a)
 for side,sign in [('L',1),('R',-1)]:
  allowed={i for i,p in enumerate(points) if .12<p.z/h<.36 and p.x*sign>.18*h}
  target=Vector((sign*.27*h,-.27*h,.25*h))
  seed=min(allowed,key=lambda i:(points[i]-target).length_squared)
  pending=[seed];seen={seed}
  while pending:
   i=pending.pop();monkey_hands[i]=side
   for other in adjacency[i]:
    if other in allowed and other not in seen:seen.add(other);pending.append(other)
def distance(p,a,b):
 d=b-a;t=max(0,min(1,(p-a).dot(d)/d.length_squared));return (p-a-t*d).length
for vertex in body.data.vertices:
 p=body.matrix_world@vertex.co;z=p.z/h;x=p.x/h;y=(p.y-cy)/h
 if vertex.index in monkey_hands:
  weights={'hand.'+monkey_hands[vertex.index]:1.0}
 elif asset=='elephant-memory' and abs(x)>.23 and z>.45:
  weights={'hand.L' if x>0 else 'hand.R':1.0}
 elif asset=='elephant-memory' and y<-.17 and abs(x)<.12 and z>.43:
  weights={'head':1.0}
 elif z>head_z:
  weights={'head':1.0}
 elif z>neck_z and abs(x)<.20:
  t=max(0,min(1,(z-neck_z)/.07));weights={'head':t,'neck':1-t}
 else:
  if y>.17 and z<.40 and 'tail.01' in segments:
   names=['tail.01','tail.02','tail.03','pelvis']
  elif abs(x)>(.235 if asset=='monkey-poet' and z<.40 else ax*.55) and hip_z*.65<z<shoulder_z+.02:
   side='L' if x>=0 else 'R';names=['upper_arm.'+side,'forearm.'+side,'hand.'+side,'clavicle.'+side,'pelvis','spine','chest']
  elif z<hip_z and y<.16:
   side='L' if x>=0 else 'R';names=['thigh.'+side,'shin.'+side,'foot.'+side,'pelvis']
  else:
   names=['pelvis','spine','chest','neck']
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
rig.pose.bones['head'].rotation_euler[1]=math.radians(15)
rig.pose.bones['thigh.L'].rotation_euler[0]=math.radians(25)
rig.pose.bones['shin.L'].rotation_euler[0]=math.radians(-35)
bpy.ops.object.camera_add(location=(h*.3,cy-h*2.8,h*1.1));cam=bpy.context.object
cam.rotation_euler=(Vector((0,cy,h*.5))-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.type='ORTHO';cam.data.ortho_scale=h*1.28;scene.camera=cam
for pos,power in [((-2,-3,3),550),((2,-2,2),260),((1,2,3),350)]:
 bpy.ops.object.light_add(type='AREA',location=Vector(pos)*h/1.5);o=bpy.context.object;o.data.energy=power;o.data.size=h*2;o.rotation_euler=(Vector((0,cy,h*.5))-o.location).to_track_quat('-Z','Y').to_euler()
scene.world=bpy.data.worlds.new('Rig studio');scene.world.use_nodes=True;scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.1,.12,.15,1)
scene.world.node_tree.nodes['Background'].inputs[1].default_value=.4
scene.render.engine='CYCLES';scene.cycles.samples=24;scene.cycles.use_denoising=True;scene.view_settings.view_transform='AgX';scene.render.resolution_x=960;scene.render.resolution_y=960;scene.render.resolution_percentage=100
scene.render.filepath=str(f/'rig-stress-pose.png');bpy.ops.render.render(write_still=True)
bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'leader-rig-workshop.blend'))
print(asset+' weight audit and stress pose ready.')
