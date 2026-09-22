"""Non-destructive surface finish for the current fitted rig scene."""
import bpy, json, numpy as np
from pathlib import Path
from mathutils import Vector
ROOT=Path(globals().get('ASSET_ROOT',Path(__file__).resolve().parents[1]/'public/lunar-city/multiview-2026-09-07'))
WORKSHOP=Path(__file__).resolve().parents[1]/'public/lunar-city/multiview-2026-09-07/leader-rig-workshop.blend'
scene=bpy.context.scene;asset=globals().get('ASSET','cat-arts');folder=ROOT/asset
body=next(o for o in scene.objects if o.type=='MESH' and any(m.type=='ARMATURE' for m in o.modifiers))
rig=body.parent
for bone in rig.pose.bones:bone.rotation_euler=(0,0,0)
bpy.context.view_layer.update()
points=[body.matrix_world@v.co for v in body.data.vertices];h=max(p.z for p in points);t=[p.y for p in points if .55*h<p.z<.70*h];cy=(min(t)+max(t))/2
# Add a dedicated tail surface; its fur must not inherit coat projections.
if asset in ('cat-arts','monkey-poet','fox-scientist'):
 material=bpy.data.materials.new(asset+' | tail fur');material.use_nodes=True
 nodes=material.node_tree.nodes;links=material.node_tree.links;shader=nodes.get('Principled BSDF')
 palette={'cat-arts':((.065,.033,.018,1),(.26,.13,.055,1)), 'monkey-poet':((.10,.045,.016,1),(.24,.12,.05,1)), 'fox-scientist':((.45,.14,.045,1),(.68,.29,.085,1))}[asset]
 tex=nodes.new('ShaderNodeTexNoise');tex.inputs['Scale'].default_value=85;tex.inputs['Detail'].default_value=2
 ramp=nodes.new('ShaderNodeValToRGB');ramp.color_ramp.elements[0].color=palette[0];ramp.color_ramp.elements[1].color=palette[1]
 links.new(tex.outputs['Fac'],ramp.inputs[0]);links.new(ramp.outputs[0],shader.inputs['Base Color']);shader.inputs['Roughness'].default_value=.75
 if asset=='cat-arts':
  wave=nodes.new('ShaderNodeTexWave');wave.wave_type='BANDS';wave.bands_direction='Y';wave.inputs['Scale'].default_value=7;wave.inputs['Distortion'].default_value=2
  links.new(wave.outputs['Color'],ramp.inputs[0])
 body.data.materials.append(material);index=len(body.data.materials)-1
 assigned=0
 for poly in body.data.polygons:
  p=body.matrix_world@poly.center
  if p.y>cy+.13*h and p.z<.43*h:
   poly.material_index=index;assigned+=1
else:assigned=0
# Bake the combined approved body texture and new tail surface into a portable atlas.
bpy.ops.object.select_all(action='DESELECT');body.select_set(True);bpy.context.view_layer.objects.active=body
atlas=bpy.data.images.new(asset+' | finish atlas',width=2048,height=2048,alpha=False)
atlas.colorspace_settings.name='sRGB'
for material in body.data.materials:
 material=material
 nodes=material.node_tree.nodes;links=material.node_tree.links;shader=nodes.get('Principled BSDF');out=nodes.get('Material Output')
 emit=nodes.new('ShaderNodeEmission')
 if shader.inputs['Base Color'].is_linked:links.new(shader.inputs['Base Color'].links[0].from_socket,emit.inputs['Color'])
 else:emit.inputs['Color'].default_value=shader.inputs['Base Color'].default_value
 links.new(emit.outputs[0],out.inputs['Surface'])
 target=nodes.new('ShaderNodeTexImage');target.image=atlas;nodes.active=target
scene.render.engine='CYCLES';scene.cycles.samples=1;scene.render.bake.use_selected_to_active=False;scene.render.bake.margin=16
bpy.ops.object.bake(type='EMIT')
# Fill zero-valued bake misses only; retain all observed colour pixels.
pixels=np.array(atlas.pixels[:],dtype=np.float32).reshape(2048,2048,4)
for iteration in range(32):
 missing=np.max(pixels[:,:,:3],axis=2)==0
 valid=~missing
 total=np.zeros_like(pixels[:,:,:3]);count=np.zeros(valid.shape,dtype=np.float32)
 for axis,step in [(0,1),(0,-1),(1,1),(1,-1)]:
  v=np.roll(valid,step,axis);total+=np.roll(pixels[:,:,:3],step,axis)*v[:,:,None];count+=v
 fill=missing&(count>0)
 if not np.any(fill):break
 pixels[fill,:3]=total[fill]/count[fill,None]
atlas.pixels.foreach_set(pixels.ravel());atlas.update()
atlas.filepath_raw=str(folder/'finish-basecolor.png');atlas.file_format='PNG';atlas.save();atlas.pack()
for material in body.data.materials:
 nodes=material.node_tree.nodes;links=material.node_tree.links;shader=nodes.get('Principled BSDF');out=nodes.get('Material Output')
 target=next(n for n in nodes if n.type=='TEX_IMAGE' and n.image==atlas)
 links.new(target.outputs['Color'],shader.inputs['Base Color']);links.new(shader.outputs[0],out.inputs['Surface'])
 for n in nodes:
  if n.type=='NORMAL_MAP':n.inputs['Strength'].default_value=.3
scene.cycles.samples=24
scene.render.filepath=str(folder/'finish-front.png');bpy.ops.render.render(write_still=True)
cam=scene.camera;old=cam.matrix_world.copy()
for name,pos in [('rear',(0,cy+h*3,h*.85)),('side',(-h*3,cy,h*.85))]:
 cam.location=pos;cam.rotation_euler=(Vector((0,cy,h*.5))-cam.location).to_track_quat('-Z','Y').to_euler();scene.render.filepath=str(folder/('finish-'+name+'.png'));bpy.ops.render.render(write_still=True)
cam.matrix_world=old
(folder/'finish-surface.json').write_text(json.dumps({'tailFaces':assigned,'atlas':'finish-basecolor.png','normalStrength':.3,'stage':'visual_review'},indent=2)+'\n')
bpy.ops.wm.save_as_mainfile(filepath=str(WORKSHOP))
