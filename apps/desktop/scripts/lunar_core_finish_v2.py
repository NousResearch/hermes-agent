"""Preserve the mechanical study; produce a portable, source-derived static candidate."""
import bpy, json, math, hashlib
from pathlib import Path
from mathutils import Vector
from mathutils.bvhtree import BVHTree
folder=Path(__file__).resolve().parents[1]/'public/lunar-city/worker-multiview-2026-09-07/core-runtime-ux-repairs/static-finish-v2'
folder.mkdir(exist_ok=True)
assert not (folder/'complete.json').exists(), 'Preserve prior completed candidate'
original=bpy.context.scene;original.frame_set(0)
scene=bpy.data.scenes.new('Worker static finish | core v2');scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1
scene.world=original.world.copy();scene.render.engine='CYCLES';scene.cycles.samples=16
clones={};deps=bpy.context.evaluated_depsgraph_get()
for source in original.objects:
 if not source.visible_get() or source.hide_render or source.type not in {'MESH','CAMERA','LIGHT'}:continue
 if source.type=='MESH':
  mesh=bpy.data.meshes.new_from_object(source.evaluated_get(deps),preserve_all_data_layers=True,depsgraph=deps)
  obj=bpy.data.objects.new('Finish v2 | '+source.name,mesh)
 else:obj=source.copy();obj.data=source.data.copy()
 obj.animation_data_clear();scene.collection.objects.link(obj);obj.matrix_world=source.matrix_world.copy();clones[source]=obj
 if source==original.camera:scene.camera=obj
bpy.context.window.scene=scene
body=next(o for o in scene.objects if o.name.startswith('Finish v2 | Core wheel partition'))
for i,m in enumerate(body.data.materials):body.data.materials[i]=m.copy()
# Smooth only the upward helmet shell; no global limb/tool smoothing.
group=body.vertex_groups.new(name='Upper helmet shell only')
weights=[]
for v in body.data.vertices:
 p=body.matrix_world@v.co;n=body.matrix_world.to_3x3()@v.normal
 w=max(0,min(1,(p.z-.91)/.08))*max(0,min(1,(n.z-.25)/.4))
 if abs(p.x)<.28 and abs(p.y)<.26 and w>0:group.add([v.index],w,'REPLACE');weights.append(v.index)
mod=body.modifiers.new('Local shell noise relief','SMOOTH');mod.factor=.35;mod.iterations=4;mod.vertex_group=group.name
bpy.context.view_layer.objects.active=body;body.select_set(True);bpy.ops.object.modifier_apply(modifier=mod.name)
# Card PLAN and FRONT both show a violet gear on the upper shell. Conform it to that surface.
bpy.context.view_layer.update();tree=BVHTree.FromObject(body,bpy.context.evaluated_depsgraph_get())
verts=[];faces=[];cx=0;cy=0
for i in range(64):
 a=2*math.pi*i/64;r=.052 if i%8 in (0,1,2,3) else .066
 for radius in (.027,r):
  x=cx+math.cos(a)*radius;y=cy+math.sin(a)*radius
  hit,normal,_,_=tree.ray_cast(body.matrix_world.inverted()@Vector((x,y,1.15)),Vector((0,0,-1)),.4)
  if hit is None:raise RuntimeError('Roof emblem ray missed source shell')
  p=body.matrix_world@hit;verts.append((p.x,p.y,p.z+.0015))
for i in range(64):
 j=(i+1)%64;faces.append((i*2,i*2+1,j*2+1,j*2))
mesh=bpy.data.meshes.new('Card gear conformal surface');mesh.from_pydata(verts,[],faces);mesh.update()
emblem=bpy.data.objects.new('Core roof gear | card matched',mesh);scene.collection.objects.link(emblem)
mat=bpy.data.materials.new('Core roof violet enamel');mat.diffuse_color=(.25,.18,.38,1);mat.use_nodes=True
bsdf=mat.node_tree.nodes.get('Principled BSDF');bsdf.inputs['Base Color'].default_value=(.25,.18,.38,1);bsdf.inputs['Metallic'].default_value=.25;bsdf.inputs['Roughness'].default_value=.4;mesh.materials.append(mat)
# Bake the existing registered base colour plus the local helmet repair into its existing UV map.
image=bpy.data.images.new('Core static finish v2 base color',width=2048,height=2048,alpha=False)
scene.render.bake.margin=12;scene.render.bake.use_clear=True
saved=[]
for mat in body.data.materials:
 nodes=mat.node_tree.nodes;links=mat.node_tree.links;bsdf=next(n for n in nodes if n.type=='BSDF_PRINCIPLED');out=next(n for n in nodes if n.type=='OUTPUT_MATERIAL')
 previous=out.inputs['Surface'].links[0].from_socket;emit=nodes.new('ShaderNodeEmission')
 if bsdf.inputs['Base Color'].is_linked:links.new(bsdf.inputs['Base Color'].links[0].from_socket,emit.inputs['Color'])
 else:emit.inputs['Color'].default_value=bsdf.inputs['Base Color'].default_value
 links.new(emit.outputs[0],out.inputs['Surface']);tex=nodes.new('ShaderNodeTexImage');tex.image=image;nodes.active=tex;saved.append((mat,bsdf,out,previous,emit,tex))
for o in scene.objects:o.select_set(False)
body.select_set(True);bpy.context.view_layer.objects.active=body;bpy.ops.object.bake(type='EMIT')
image.filepath_raw=str(folder/'base-color.png');image.file_format='PNG';image.save();image.pack()
for mat,bsdf,out,previous,emit,tex in saved:
 mat.node_tree.links.new(previous,out.inputs['Surface']);mat.node_tree.links.new(tex.outputs['Color'],bsdf.inputs['Base Color']);mat.node_tree.nodes.remove(emit)
# Save static candidate without overwriting source rig or review GLB.
for o in scene.objects:o.select_set(o.type=='MESH')
bpy.context.view_layer.objects.active=body
bpy.ops.export_scene.gltf(filepath=str(folder/'static-finish.glb'),export_format='GLB',use_selection=True,use_active_scene=True,export_animations=False,export_yup=True)
scene.render.resolution_x=800;scene.render.resolution_y=800;scene.render.resolution_percentage=100
cam=scene.camera;cam.data.type='ORTHO';cam.data.ortho_scale=1.55
for name,pos in [('front',(0,-3,.66)),('rear',(0,3,.66)),('left',(-3,0,.66)),('right',(3,0,.66)),('top',(0,0,3))]:
 cam.location=pos;cam.rotation_euler=(Vector((0,0,.60))-cam.location).to_track_quat('-Z','Y').to_euler();scene.render.filepath=str(folder/(name+'.png'));bpy.ops.render.render(write_still=True)
clay=bpy.data.materials.new('Core v2 neutral inspection');clay.diffuse_color=(.38,.38,.38,1);clay.use_nodes=True
clay.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value=(.38,.38,.38,1)
scene.view_layers[0].material_override=clay;cam.location=(2,-3,1.65);cam.rotation_euler=(Vector((0,0,.60))-cam.location).to_track_quat('-Z','Y').to_euler();scene.render.filepath=str(folder/'clay.png');bpy.ops.render.render(write_still=True);scene.view_layers[0].material_override=None
scene.render.filepath=str(folder/'overview.png');bpy.ops.render.render(write_still=True)
points=[o.matrix_world@v.co for o in scene.objects if o.type=='MESH' for v in o.data.vertices]
receipt={'stage':'Static candidate, detailed appendage repairs and rig binding pending','sourceScene':original.name,'scene':scene.name,'heightMetres':max(p.z for p in points)-min(p.z for p in points),'groundMin':min(p.z for p in points),'helmetVerticesSmoothed':len(weights),'changes':['source-only local helmet smoothing','card roof gear restored','portable2048 basecolor bake','preserved repaired antenna and unified wheels'],'sha256':hashlib.sha256((folder/'static-finish.glb').read_bytes()).hexdigest(),'pending':['arm/tool/support shape repairs','final static acceptance','rig binding and animation refinement']}
(folder/'complete.json').write_text(json.dumps(receipt,indent=2)+'\n')
bpy.ops.wm.save_as_mainfile(filepath=str(folder.parents[2]/'multiview-2026-09-07/leader-rig-workshop.blend'))
print('Core v2 static candidate saved',folder)
