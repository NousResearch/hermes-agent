"""Separate editable eyes over the reconstructed sockets for a beauty study."""
import bpy
from pathlib import Path
F=Path(__file__).resolve().parents[1]/'public/lunar-city/multiview-2026-09-07/owl-librarian'
scene=bpy.data.scenes['Owl | source color alignment.001'];bpy.context.window.scene=scene
coll=bpy.data.collections.new('Owl | editable eye study');scene.collection.children.link(coll)
def material(name,color,roughness,metallic=0):
 m=bpy.data.materials.new(name);m.use_nodes=True;s=m.node_tree.nodes['Principled BSDF'];s.inputs['Base Color'].default_value=(*color,1);s.inputs['Roughness'].default_value=roughness;s.inputs['Metallic'].default_value=metallic;return m
iris=material('Owl iris | warm amber',(.16,.073,.016),.24)
pupil=material('Owl pupil | polished onyx',(.002,.003,.004),.1)
for x in (-.16,.16):
 for name,loc,scale,mat in [('Iris',(x,-.32,1.105),(.066,.063,.076),iris),('Pupil',(x,-.379,1.105),(.042,.01,.053),pupil)]:
  bpy.ops.mesh.primitive_uv_sphere_add(segments=48,ring_count=24,radius=1,location=loc)
  obj=bpy.context.object;obj.name='Owl '+name+(' L' if x>0 else ' R');obj.scale=scale
  for c in list(obj.users_collection):c.objects.unlink(obj)
  coll.objects.link(obj);obj.data.materials.append(mat)
  for p in obj.data.polygons:p.use_smooth=True
scene.render.filepath=str(F/'eye-study.png')
bpy.ops.wm.save_as_mainfile(filepath=str(F.parent/'leader-workshop.blend'))
bpy.ops.render.render(write_still=True)
