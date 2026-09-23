"""Preview source-projected color without altering generated source scenes."""
from pathlib import Path
import bpy
from mathutils import Vector
F=Path(__file__).resolve().parents[1]/'public/lunar-city/multiview-2026-09-07/owl-librarian'
scene=bpy.data.scenes.new('Owl | source color alignment')
bpy.context.window.scene=scene
scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1
before=set(bpy.data.objects)
bpy.ops.import_scene.gltf(filepath=str(F/'source-color-study-v2.glb'))
objects=[o for o in bpy.data.objects if o not in before and o.type=='MESH']
for obj in objects:
    obj.name='Owl | projected source color study'
    for p in obj.data.polygons:p.use_smooth=True
    smooth=obj.modifiers.new('Gentle smoothing','SMOOTH');smooth.factor=.18;smooth.iterations=2
    mat=bpy.data.materials.new('Owl | source color - provisional surface');mat.use_nodes=True
    shader=mat.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Roughness'].default_value=.65
    color=mat.node_tree.nodes.new('ShaderNodeVertexColor');color.layer_name=obj.data.color_attributes[0].name
    mat.node_tree.links.new(color.outputs['Color'],shader.inputs['Base Color'])
    obj.data.materials.clear();obj.data.materials.append(mat)
    obj['status']='source_projection_alignment_review'
bpy.ops.object.camera_add(location=(.3,-4,1.9));cam=bpy.context.object
cam.rotation_euler=(Vector((0,0,.77))-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.type='ORTHO';cam.data.ortho_scale=1.95;scene.camera=cam
for name,pos,power,size in [('Key',(-3,-4,4),450,4),('Fill',(3,-2,2),160,3),('Rim',(1,3,3),350,3)]:
    bpy.ops.object.light_add(type='AREA',location=pos);o=bpy.context.object;o.name=name;o.data.energy=power;o.data.shape='DISK';o.data.size=size;o.rotation_euler=(Vector((0,0,.8))-o.location).to_track_quat('-Z','Y').to_euler()
scene.world=bpy.data.worlds.new('Owl studio');scene.world.use_nodes=True;scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.09,.11,.13,1);scene.world.node_tree.nodes['Background'].inputs[1].default_value=.4
scene.render.engine='CYCLES';scene.cycles.samples=24;scene.cycles.use_denoising=True
scene.render.resolution_x=960;scene.render.resolution_y=960;scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG';scene.render.filepath=str(F/'source-color-review-v2.png')
scene.view_settings.view_transform='AgX'
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type=='VIEW_3D':
            area.spaces.active.region_3d.view_perspective='CAMERA'
            area.spaces.active.shading.type='MATERIAL'
bpy.ops.wm.save_as_mainfile(filepath=str(F.parent/'leader-workshop.blend'))
bpy.ops.render.render(write_still=True)
