"""Repair the source-matched antenna and isolated tip fragments at the measured scale."""
import bpy,bmesh,json
from pathlib import Path
from statistics import median
scene=bpy.context.scene
body=next(o for o in scene.objects if o.name.startswith('Core wheel partition'))
assert not body.get('antenna_repaired')
pts=[body.matrix_world@v.co for v in body.data.vertices]
upper=[p for p in pts if p.z>1.08]
x=median(p.x for p in upper);y=median(p.y for p in upper);top=max(p.z for p in pts)
assert len(upper)<200 and 1.19<top<1.21
old=body.data;old.use_fake_user=True;body.data=old.copy()
bm=bmesh.new();bm.from_mesh(body.data)
remove=[v for v in bm.verts if (body.matrix_world@v.co).z>1.055]
assert len(remove)<400
removed=len(remove);bmesh.ops.delete(bm,geom=remove,context='VERTS');bm.to_mesh(body.data);bm.free()
def material(name,color,metal,rough,emit=0):
 m=bpy.data.materials.new(name);m.use_nodes=True;s=m.node_tree.nodes.get('Principled BSDF');s.inputs['Base Color'].default_value=(*color,1);s.inputs['Metallic'].default_value=metal;s.inputs['Roughness'].default_value=rough
 if emit:s.inputs['Emission Color'].default_value=(*color,1);s.inputs['Emission Strength'].default_value=emit
 return m
dark=material('Core antenna graphite',(.025,.032,.04),.6,.35)
purple=material('Core antenna violet collar',(.25,.18,.38),.3,.45)
cyan=material('Core antenna cyan lens',(.025,.65,.72),.1,.25,1.2)
parts=[]
for name,radius,low,high,mat in [('base',.014,1.045,1.068,purple),('mast',.006,1.06,top-.024,dark),('lens',.008,top-.026,top,cyan)]:
 bpy.ops.mesh.primitive_cylinder_add(vertices=24,radius=radius,depth=high-low,location=(x,y,(low+high)/2))
 o=bpy.context.object;o.name='Core repaired antenna | '+name;o.data.materials.append(mat)
 bevel=o.modifiers.new('Small machined edge','BEVEL');bevel.width=.001;bevel.segments=2
 parts.append(o.name)
body['antenna_repaired']=True
folder=Path(__file__).resolve().parents[1]/'public/lunar-city/worker-multiview-2026-09-07/core-runtime-ux-repairs'
(folder/'antenna-repair.json').write_text(json.dumps({'stage':'Static local component repair','preservedSourceMesh':old.name,'removedSourceVertices':removed,'axis':[x,y],'topMetres':top,'parts':parts},indent=2)+'\n')
bpy.ops.wm.save_as_mainfile(filepath=str(folder.parents[1]/'multiview-2026-09-07/leader-rig-workshop.blend'))
