"""Fit and deformation-check the first character rig in a separate Blender scene."""
import bpy, math, json
from pathlib import Path
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[1]/'public/lunar-city/multiview-2026-09-07'
f=ROOT/'cat-arts'
scene=bpy.data.scenes.new('Rig | cat-arts');bpy.context.window.scene=scene
scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1
before=set(bpy.data.objects);bpy.ops.import_scene.gltf(filepath=str(f/'material-review-v4.glb'))
body=next(o for o in bpy.data.objects if o not in before and o.type=='MESH')
matrix=body.matrix_world.copy();body.parent=None;body.matrix_world=matrix
bpy.context.view_layer.objects.active=body;bpy.ops.object.transform_apply(location=False,rotation=True,scale=True)
body.name='Cat | skinned body'
points=[body.matrix_world@v.co for v in body.data.vertices];h=max(p.z for p in points)
torso=[p.y for p in points if .55*h<p.z<.7*h];cy=(min(torso)+max(torso))/2
for mat in body.data.materials:
 if mat.use_nodes:
  for n in mat.node_tree.nodes:
   if n.type=='NORMAL_MAP':n.inputs['Strength'].default_value=.35
   if n.type=='BSDF_PRINCIPLED':n.inputs['Roughness'].default_value=.62
bpy.ops.object.armature_add(enter_editmode=True,location=(0,0,0));rig=bpy.context.object;rig.name='Cat | animation rig'
e=rig.data.edit_bones;e.remove(e[0])
def bone(name,start,end,parent=None):
 b=e.new(name);b.head=(start[0]*h,cy+start[1]*h,start[2]*h);b.tail=(end[0]*h,cy+end[1]*h,end[2]*h)
 if parent:b.parent=e[parent]
 return b
bone('root',(0,0,0),(0,0,.1)).use_deform=False
bone('pelvis',(0,0,.31),(0,0,.43),'root')
bone('spine',(0,0,.43),(0,0,.59),'pelvis')
bone('chest',(0,0,.59),(0,0,.70),'spine')
bone('neck',(0,0,.70),(0,0,.77),'chest')
bone('head',(0,0,.77),(0,0,.96),'neck')
for side,sign in [('L',1),('R',-1)]:
 bone('clavicle.'+side,(0,0,.67),(sign*.16,0,.65),'chest')
 bone('upper_arm.'+side,(sign*.16,0,.65),(sign*.245,-.005,.49),'clavicle.'+side)
 bone('forearm.'+side,(sign*.245,-.005,.49),(sign*.29,-.01,.39),'upper_arm.'+side)
 bone('hand.'+side,(sign*.29,-.01,.39),(sign*.29,-.01,.33),'forearm.'+side)
 bone('thigh.'+side,(sign*.10,0,.35),(sign*.10,-.015,.20),'pelvis')
 bone('shin.'+side,(sign*.10,-.015,.20),(sign*.10,0,.065),'thigh.'+side)
 bone('foot.'+side,(sign*.10,0,.065),(sign*.10,-.115,.045),'shin.'+side)
bone('tail.01',(0,.08,.31),(0,.24,.20),'pelvis')
bone('tail.02',(0,.24,.20),(0,.43,.23),'tail.01')
bone('tail.03',(0,.43,.23),(0,.60,.35),'tail.02')
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
 if z>.77:
  weights={'head':1.0}
 elif z>.70 and abs(x)<.20:
  t=max(0,min(1,(z-.70)/.07));weights={'head':t,'neck':1-t}
 else:
  if y>.17 and z<.40:
   names=['tail.01','tail.02','tail.03','pelvis']
  elif z<.30 and y<.16:
   side='L' if x>=0 else 'R';names=['thigh.'+side,'shin.'+side,'foot.'+side,'pelvis']
  elif abs(x)>.20 and .30<z<.68:
   side='L' if x>=0 else 'R';names=['upper_arm.'+side,'forearm.'+side,'hand.'+side,'clavicle.'+side]
  else:
   names=['pelvis','spine','chest','neck']
  ranked=sorted((distance(p,*segments[n]),n) for n in names)[:3]
  weights={n:1/max(d,.015*h)**4 for d,n in ranked};total=sum(weights.values());weights={n:w/total for n,w in weights.items()}
 for name,w in weights.items():
  if w>1e-6:groups[name].add([vertex.index],w,'REPLACE')
unweighted=[v.index for v in body.data.vertices if not any(g.weight>1e-6 for g in v.groups)]
(f/'rig-weight-check.json').write_text(json.dumps({'method':'fitted anatomical segment weights','unweighted':len(unweighted),'vertices':len(body.data.vertices),'bones':len(rig.data.bones)},indent=2)+'\n')
if unweighted:raise RuntimeError(f'Weights left {len(unweighted)} vertices unweighted')
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
print('Cat weight audit and stress pose ready.')
