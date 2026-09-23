import bpy,bmesh,json,hashlib
from pathlib import Path
from mathutils import Vector
from mathutils.bvhtree import BVHTree
folder=Path(__file__).resolve().parents[1]/'public/lunar-city/worker-multiview-2026-09-07/core-runtime-ux-repairs/static-finish-v2'
scene=bpy.data.scenes['Worker static finish | core v2'];bpy.context.window.scene=scene
body=next(o for o in scene.objects if o.name.startswith('Finish v2 | Core wheel partition'))
source=bpy.data.scenes['Mechanical study | core rigid wheels.001'].objects['Core wheel partition study | source geometry unchanged.002']
mats=list(body.data.materials);body.data=source.data.copy();body.data.materials.clear()
for m in mats:body.data.materials.append(m)
# Heal only coincident helmet seam vertices, preserving per-loop UV coordinates.
bm=bmesh.new();bm.from_mesh(body.data)
seams=[v for v in bm.verts if v.co.z>.87 and abs(v.co.x)<.30 and abs(v.co.y)<.28]
before=len(bm.verts);bmesh.ops.remove_doubles(bm,verts=seams,dist=.00001);merged=before-len(bm.verts)
bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bm.to_mesh(body.data);bm.free()
body.vertex_groups.clear();group=body.vertex_groups.new(name='Upper helmet shell only')
for v in body.data.vertices:
 p=body.matrix_world@v.co;n=body.matrix_world.to_3x3()@v.normal
 w=max(0,min(1,(p.z-.91)/.08))*max(0,min(1,(n.z-.25)/.4))
 if abs(p.x)<.28 and abs(p.y)<.26 and w>0:group.add([v.index],w,'REPLACE')
mod=body.modifiers.new('Local joined-shell noise relief','SMOOTH');mod.factor=.25;mod.iterations=3;mod.vertex_group=group.name
for o in scene.objects:o.select_set(False)
body.select_set(True);bpy.context.view_layer.objects.active=body;bpy.ops.object.modifier_apply(modifier=mod.name)
bpy.context.view_layer.update();tree=BVHTree.FromObject(body,bpy.context.evaluated_depsgraph_get())
emblem=scene.objects['Core roof gear | card matched']
for v in emblem.data.vertices:
 hit,_,_,_=tree.ray_cast(Vector((v.co.x,v.co.y,1.15)),Vector((0,0,-1)),.4)
 if hit:v.co.z=hit.z+.0015
for o in scene.objects:o.select_set(o.type=='MESH')
bpy.ops.export_scene.gltf(filepath=str(folder/'static-finish.glb'),export_format='GLB',use_selection=True,use_active_scene=True,export_animations=False,export_yup=True)
cam=scene.camera
for name,pos in [('front',(0,-3,.66)),('rear',(0,3,.66)),('left',(-3,0,.66)),('right',(3,0,.66)),('top',(0,0,3)),('overview',(2,-3,1.65))]:
 cam.location=pos;cam.rotation_euler=(Vector((0,0,.60))-cam.location).to_track_quat('-Z','Y').to_euler();scene.render.filepath=str(folder/(name+'.png'));bpy.ops.render.render(write_still=True)
scene.view_layers[0].material_override=bpy.data.materials['Core v2 neutral inspection'];scene.render.filepath=str(folder/'clay.png');bpy.ops.render.render(write_still=True);scene.view_layers[0].material_override=None
receipt=json.loads((folder/'complete.json').read_text());receipt['seamVerticesWelded']=merged;receipt['exportScope']='current scene only, selected meshes';receipt['sha256']=hashlib.sha256((folder/'static-finish.glb').read_bytes()).hexdigest();receipt['bytes']=(folder/'static-finish.glb').stat().st_size
(folder/'complete.json').write_text(json.dumps(receipt,indent=2)+'\n')
bpy.ops.wm.save_as_mainfile(filepath=str(folder.parents[2]/'multiview-2026-09-07/leader-rig-workshop.blend'))
print('Core v2 seam repair and scoped export complete',merged)
