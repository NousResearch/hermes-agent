import bpy,json,hashlib
from pathlib import Path
scene=bpy.data.scenes['Worker static finish | core v3 probe'];bpy.context.window.scene=scene
archive=bpy.data.collections.new('Rejected visor surface fit | preserved');scene.collection.children.link(archive)
for o in list(scene.objects):
 if o.name.startswith(('Core clean visor |','Core cyan eye |')):
  for c in list(o.users_collection):c.objects.unlink(o)
  archive.objects.link(o);o.hide_render=True;o.hide_set(True)
for o in scene.objects:o.select_set(o.type=='MESH' and not o.hide_render)
folder=Path(__file__).resolve().parents[1]/'public/lunar-city/worker-multiview-2026-09-07/core-runtime-ux-repairs/static-finish-v3'
bpy.ops.export_scene.gltf(filepath=str(folder/'static-finish.glb'),export_format='GLB',use_selection=True,use_active_scene=True,export_animations=False)
r=json.loads((folder/'complete.json').read_text());r['visor']='Separate visor overlay rejected; preserved hidden. Source texture retained, registered texture repair pending.';r['sha256']=hashlib.sha256((folder/'static-finish.glb').read_bytes()).hexdigest();(folder/'complete.json').write_text(json.dumps(r,indent=2)+'\n')
scene.render.filepath=str(folder/'overview.png');bpy.ops.render.render(write_still=True)
bpy.ops.wm.save_as_mainfile(filepath=str(folder.parents[2]/'multiview-2026-09-07/leader-rig-workshop.blend'))
