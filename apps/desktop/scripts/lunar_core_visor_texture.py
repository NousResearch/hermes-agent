"""Register clean display colours on the existing visor surface, then bake for GLB."""
import bpy,json,hashlib
from pathlib import Path
from mathutils import Vector
scene=bpy.data.scenes['Worker static finish | core v3 probe'];bpy.context.window.scene=scene
body=next(o for o in scene.objects if o.name.startswith('Finish v2 | Core wheel partition'))
folder=Path(__file__).resolve().parents[1]/'public/lunar-city/worker-multiview-2026-09-07/core-runtime-ux-repairs/static-finish-v3'
assert not body.get('visor_texture_finished')
img=bpy.data.images.new('Core registered visor base',width=2048,height=2048,alpha=False)
saved=[]
for i,old in enumerate(body.data.materials):
 mat=old.copy();body.data.materials[i]=mat;nodes=mat.node_tree.nodes;links=mat.node_tree.links
 bsdf=next(n for n in nodes if n.type=='BSDF_PRINCIPLED');out=next(n for n in nodes if n.type=='OUTPUT_MATERIAL');prior=bsdf.inputs['Base Color'].links[0].from_socket
 geo=nodes.new('ShaderNodeNewGeometry');sep=nodes.new('ShaderNodeSeparateXYZ');links.new(geo.outputs['Position'],sep.inputs[0])
 def mathnode(op,a,b=None):
  n=nodes.new('ShaderNodeMath');n.operation=op
  for index,val in enumerate([a] if b is None else [a,b]):
   if isinstance(val,(int,float)):n.inputs[index].default_value=val
   else:links.new(val,n.inputs[index])
  return n.outputs[0]
 def rounded(cx,cz,w,h,r):
  x=mathnode('MAXIMUM',mathnode('SUBTRACT',mathnode('ABSOLUTE',mathnode('SUBTRACT',sep.outputs['X'],cx)),w/2-r),0)
  z=mathnode('MAXIMUM',mathnode('SUBTRACT',mathnode('ABSOLUTE',mathnode('SUBTRACT',sep.outputs['Z'],cz)),h/2-r),0)
  return mathnode('LESS_THAN',mathnode('SQRT',mathnode('ADD',mathnode('MULTIPLY',x,x),mathnode('MULTIPLY',z,z))),r)
 front=mathnode('LESS_THAN',sep.outputs['Y'],-.025)
 plate=mathnode('MULTIPLY',rounded(0,.835,.39,.19,.045),front)
 eyes=mathnode('MULTIPLY',mathnode('MAXIMUM',rounded(-.075,.835,.043,.056,.018),rounded(.075,.835,.043,.056,.018)),front)
 mix=nodes.new('ShaderNodeMixRGB');links.new(plate,mix.inputs[0]);links.new(prior,mix.inputs[1]);mix.inputs[2].default_value=(.006,.015,.022,1)
 eye=nodes.new('ShaderNodeMixRGB');links.new(eyes,eye.inputs[0]);links.new(mix.outputs[0],eye.inputs[1]);eye.inputs[2].default_value=(.025,.72,.82,1)
 emit=nodes.new('ShaderNodeEmission');links.new(eye.outputs[0],emit.inputs[0]);links.new(emit.outputs[0],out.inputs['Surface'])
 tex=nodes.new('ShaderNodeTexImage');tex.image=img;nodes.active=tex;saved.append((mat,bsdf,out,emit,tex))
for o in scene.objects:o.select_set(False)
body.select_set(True);bpy.context.view_layer.objects.active=body;scene.render.bake.margin=12;bpy.ops.object.bake(type='EMIT')
img.filepath_raw=str(folder/'base-color-visor.png');img.file_format='PNG';img.save();img.pack()
for mat,bsdf,out,emit,tex in saved:
 mat.node_tree.links.new(bsdf.outputs[0],out.inputs['Surface']);mat.node_tree.links.new(tex.outputs[0],bsdf.inputs['Base Color']);mat.node_tree.nodes.remove(emit)
body['visor_texture_finished']=True
for o in scene.objects:o.select_set(o.type=='MESH' and not o.hide_render)
bpy.ops.export_scene.gltf(filepath=str(folder/'static-finish.glb'),export_format='GLB',use_selection=True,use_active_scene=True,export_animations=False)
cam=scene.camera
for name,pos in [('front',(0,-3,.66)),('rear',(0,3,.66)),('left',(-3,0,.66)),('right',(3,0,.66)),('top',(0,0,3)),('overview',(2,-3,1.65))]:
 cam.location=pos;cam.rotation_euler=(Vector((0,0,.60))-cam.location).to_track_quat('-Z','Y').to_euler();scene.render.filepath=str(folder/(name+'.png'));bpy.ops.render.render(write_still=True)
r=json.loads((folder/'complete.json').read_text());r['visor']='Baked source-surface graphite display and two cyan eyes; rejected separate panels remain hidden';r['sha256']=hashlib.sha256((folder/'static-finish.glb').read_bytes()).hexdigest();(folder/'complete.json').write_text(json.dumps(r,indent=2)+'\n')
bpy.ops.wm.save_as_mainfile(filepath=str(folder.parents[2]/'multiview-2026-09-07/leader-rig-workshop.blend'))
print('Core registered visor bake saved')
