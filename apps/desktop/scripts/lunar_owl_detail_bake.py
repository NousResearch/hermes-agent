"""Bake source detail onto the reduced owl review mesh."""
from pathlib import Path
import bpy
F=Path(__file__).resolve().parents[1]/'public/lunar-city/multiview-2026-09-07/owl-librarian'
scene=bpy.data.scenes['Owl | UV cleanup review'];bpy.context.window.scene=scene
low=next(o for o in scene.objects if o.name.startswith('Owl | UV body'))
source=next(o for o in bpy.data.scenes['Owl | source color alignment.001'].objects if o.type=='MESH' and 'projected' in o.name)
high=source.copy();high.data=source.data.copy();scene.collection.objects.link(high);high.name='Owl | high detail bake source'
mat=source.data.materials[0].copy();high.data.materials.clear();high.data.materials.append(mat)
color=next(n for n in mat.node_tree.nodes if n.type=='VERTEX_COLOR');output=next(n for n in mat.node_tree.nodes if n.type=='OUTPUT_MATERIAL')
emit=mat.node_tree.nodes.new('ShaderNodeEmission');mat.node_tree.links.new(color.outputs['Color'],emit.inputs['Color']);mat.node_tree.links.new(emit.outputs[0],output.inputs['Surface'])
lmat=low.data.materials[0];nodes=lmat.node_tree.nodes;links=lmat.node_tree.links
tex=next(n for n in nodes if n.type=='TEX_IMAGE');nodes.active=tex
bpy.ops.object.select_all(action='DESELECT');low.select_set(True);high.select_set(True);bpy.context.view_layer.objects.active=low
for link in list(tex.outputs['Color'].links): links.remove(link)
scene.cycles.samples=1;scene.render.bake.use_selected_to_active=True;scene.render.bake.cage_extrusion=.015;scene.render.bake.max_ray_distance=.035
bpy.ops.object.bake(type='EMIT')
links.new(tex.outputs['Color'],nodes.get('Principled BSDF').inputs['Base Color'])
tex.image.filepath_raw=str(F/'owl-basecolor-detail.png');tex.image.save();tex.image.pack()
normal=bpy.data.images.new('Owl source normal atlas',width=2048,height=2048,alpha=False);normal.colorspace_settings.name='Non-Color'
nt=nodes.new('ShaderNodeTexImage');nt.image=normal;nodes.active=nt
bpy.ops.object.bake(type='NORMAL')
normal.filepath_raw=str(F/'owl-normal-detail.png');normal.file_format='PNG';normal.save();normal.pack()
normal_node=nodes.new('ShaderNodeNormalMap');normal_node.uv_map=low.data.uv_layers.active.name;links.new(nt.outputs['Color'],normal_node.inputs['Color']);links.new(normal_node.outputs['Normal'],nodes.get('Principled BSDF').inputs['Normal'])
high.hide_render=True;high.hide_set(True)
scene.render.bake.use_selected_to_active=False;scene.cycles.samples=32
bpy.ops.object.select_all(action='DESELECT')
for obj in scene.objects:
 if obj.type=='MESH' and obj!=high:obj.select_set(True)
bpy.context.view_layer.objects.active=low
bpy.ops.export_scene.gltf(filepath=str(F/'owl-detail-review.glb'),export_format='GLB',use_selection=True,use_active_scene=True,export_animations=False)
scene.render.filepath=str(F/'detail-review.png')
bpy.ops.wm.save_as_mainfile(filepath=str(F.parent/'leader-workshop.blend'))
bpy.ops.render.render(write_still=True)
