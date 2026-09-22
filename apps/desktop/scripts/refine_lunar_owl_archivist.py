"""Refine the visible owl study without replacing its scene or source geometry."""
from pathlib import Path
import json
import math
import runpy
import bpy
from mathutils import Vector

ROOT=Path(__file__).resolve().parents[1]
helpers=runpy.run_path(str(ROOT/'scripts/model_lunar_owl_archivist.py'))
scene=bpy.context.scene
assert scene.name.startswith('Owl Archivist - card study')
oval=helpers['oval'];cord=helpers['cord']
cream=bpy.data.materials['Owl - ivory facial plumage']
brown=bpy.data.materials['Owl - umber feathers']
gold=bpy.data.materials['Owl - antique brass']
teal=bpy.data.materials['Owl - teal wool']

for obj in scene.objects:
    if obj.name.startswith('Facial disk feather'):
        obj.scale.x=.036;obj.scale.y=.018;obj.scale.z=.055
    if obj.name.startswith('Crown feather'):
        obj.scale.y=.012

for side in (-1,1):
    oval('Soft facial plumage foundation',(side*.25,-.416,2.01),(.265,.035,.29),cream,40,24)
    cord('Upper feather brow',[(side*.08,-.482,2.20),(side*.23,-.455,2.30),(side*.42,-.36,2.30)],.047,brown)
    cord('Sleeve tailored seam',[(side*.64,-.23,1.40),(side*.76,-.265,1.11),(side*.78,-.24,.84)],.005,gold)
    for j in range(3):
        oval('Cuff button',(side*(.69+j*.06),-.266,.81),(.015,.009,.015),gold,12,8)

def textured(mat,scale,strength,distance):
    nodes=mat.node_tree.nodes;links=mat.node_tree.links
    bsdf=nodes.get('Principled BSDF')
    tex=nodes.new('ShaderNodeTexNoise');tex.inputs['Scale'].default_value=scale;tex.inputs['Detail'].default_value=2
    bump=nodes.new('ShaderNodeBump');bump.inputs['Strength'].default_value=strength;bump.inputs['Distance'].default_value=distance
    links.new(tex.outputs['Fac'],bump.inputs['Height']);links.new(bump.outputs['Normal'],bsdf.inputs['Normal'])

textured(teal,180,.26,.012)
textured(bpy.data.materials['Owl - aged leather'],95,.20,.007)
textured(brown,110,.22,.008)
for loc,power,size in [((-3,-4,6),650,4),((4,-2,3),400,3),((0,3,5),850,3)]:
    bpy.ops.object.light_add(type='AREA',location=loc)
    obj=bpy.context.object;obj.name='Owl studio softbox';obj.data.energy=power;obj.data.shape='DISK';obj.data.size=size
    obj.rotation_euler=(Vector((0,0,1.2))-obj.location).to_track_quat('-Z','Y').to_euler()
scene.render.engine='CYCLES';scene.cycles.samples=32
scene.cycles.use_denoising=True
scene.render.resolution_x=900;scene.render.resolution_y=900
scene.view_settings.view_transform='AgX'
scene.camera.location=(3,-8,3.0)
scene.camera.rotation_euler=(Vector((0,0,1.3))-scene.camera.location).to_track_quat('-Z','Y').to_euler()
scene.camera.data.ortho_scale=3.0
bpy.ops.object.select_all(action='DESELECT')
out=ROOT/'public/lunar-city/production-studies/owl-archivist'
scene.render.filepath=str(out/'owl-archivist-study.png')
bpy.ops.wm.save_as_mainfile(filepath=str(out/'owl-archivist-study.blend'))
receipt={'status':'modeling_study_not_production_ready','scene':scene.name,'meshObjects':sum(o.type=='MESH' for o in scene.objects),'sourceCard':scene['source_card'],'remaining':['reference fidelity refinement','retopology and UVs','texture baking','rig and deformation validation','animations','LOD and collision','game integration'],'preservedScenes':[s.name for s in bpy.data.scenes if s!=scene]}
(out/'study-status.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('Refined owl study saved; source scenes preserved.')
