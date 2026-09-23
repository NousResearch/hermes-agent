"""Open a separate metric review scene; never replace source scenes."""
from pathlib import Path
import math
import json
import bpy
from mathutils import Vector

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'public/lunar-city/multiview-2026-09-07/owl-librarian'
scene=bpy.data.scenes.new('Card-derived owl - metre review')
bpy.context.window.scene=scene
scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1
scene.unit_settings.length_unit='METERS'
mat=bpy.data.materials.new('Metric guide blue');mat.diffuse_color=(.07,.5,.7,1)

def label(words,loc,size=.09):
    bpy.ops.object.text_add(location=loc,rotation=(math.pi/2,0,0))
    obj=bpy.context.object;obj.name='Guide - '+words;obj.data.body=words;obj.data.size=size
    obj.data.materials.append(mat)

for i,name in enumerate(('front','left','back')):
    bpy.ops.object.empty_add(type='IMAGE',location=(-4.3+i*1.3,.7,1.05),rotation=(math.pi/2,0,0))
    obj=bpy.context.object;obj.name='Source view - '+name;obj.data=bpy.data.images.load(str(FOLDER/f'{name}.png'));obj.empty_display_size=1.2
    label(name.upper()+' INPUT',(-4.75+i*1.3,.69,.4))
label('MULTI-VIEW RECONSTRUCTION | 1 UNIT = 1 METRE',(-2.95,.7,2.5),.11)
label('Owl target height: 1.50 m',(.8,0,2.35))
# A labelled two-metre gauge, solely for judging physical size.
for z in [i/10 for i in range(21)]:
    bpy.ops.mesh.primitive_cube_add(size=1,location=(2.2,0,z))
    obj=bpy.context.object;obj.name='Metric gauge';obj.scale=(.12 if round(z*10)%5==0 else .06,.015,.008);obj.data.materials.append(mat)
    if round(z*10)%5==0:label(f'{z:.1f} m',(2.3,0,z),.06)
path=FOLDER/'shape-metres.glb'
if path.exists():
    before=set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    bpy.context.view_layer.update()
    meshes=[o for o in bpy.data.objects if o not in before and o.type=='MESH']
    points=[o.matrix_world@Vector(corner) for o in meshes for corner in o.bound_box]
    low=min(p.z for p in points);high=max(p.z for p in points)
    assert abs(high-low-1.5)<.001,(low,high)
    for obj in meshes:
        obj['source_type']='Hunyuan3D-2mv card reconstruction';obj['production_status']='shape_review'
        for poly in obj.data.polygons:poly.use_smooth=True
    scene['verified_height_metres']=high-low
    label('GENERATED SHAPE | MATERIALS PENDING',(-.65,-.05,-.15),.065)
else:
    label('GPU GENERATION RUNNING',(.2,0,1.5))
    label('No replacement geometry',(.2,0,1.3),.065)
bpy.ops.object.camera_add(location=(-1,-9,2.8))
camera=bpy.context.object;camera.rotation_euler=(Vector((-1,0,1.25))-camera.location).to_track_quat('-Z','Y').to_euler();camera.data.type='ORTHO';camera.data.ortho_scale=8.2;scene.camera=camera
scene.render.engine='BLENDER_WORKBENCH';scene.display.shading.color_type='MATERIAL'
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.wm.save_as_mainfile(filepath=str(FOLDER/'metric-review.blend'))
