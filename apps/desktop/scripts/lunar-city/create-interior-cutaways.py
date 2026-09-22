"""Run in visible Blender: runpy.run_path(PATH, run_name='__main__').
Creates a separate scene; does not alter or save the user's exterior scene.
"""
import json, math
from pathlib import Path
import bpy

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'public/lunar-city/interior-plans-v1'
data = json.loads((OUT / 'plans.json').read_text())
receipt_path=OUT/'blender-export-receipt.json'
if receipt_path.exists() and not (OUT/'blender-export-receipt-v2.json').exists():
 (OUT/'blender-export-receipt-v2.json').write_bytes(receipt_path.read_bytes())
scene = bpy.data.scenes.new('Lunar City Interior Studies v3')
scene.unit_settings.system = 'METRIC'
scene.unit_settings.scale_length = 1
previous_scene = bpy.context.window.scene
bpy.context.window.scene = scene
palette = {'floor':(.72,.79,.76,1), 'wall':(.75,.82,.82,1), 'header':(.75,.82,.82,1), 'desk':(.48,.25,.12,1), 'bench':(.48,.25,.12,1), 'planter':(.15,.38,.22,1), 'ceiling':(.85,.85,.81,1)}
materials={}
for kind,color in palette.items():
 material=bpy.data.materials.new('Interior '+kind)
 material.diffuse_color=color
 material.use_nodes=True
 material.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value=color
 material.node_tree.nodes.get('Principled BSDF').inputs['Roughness'].default_value=.7
 materials[kind]=material

def cuboid(name, center, size, kind, collection):
 # Shared source contract Y-up +Z-front -> Blender Z-up -Y-front.
 bpy.ops.mesh.primitive_cube_add(size=1, location=(center[0],-center[2],center[1]))
 obj=bpy.context.object;obj.name=name
 obj.dimensions=(size[0],size[2],size[1])
 bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
 for old in list(obj.users_collection): old.objects.unlink(obj)
 collection.objects.link(obj)
 obj.data.materials.append(materials[kind])
 obj['interior_part']=kind
 return obj

receipts=[]
for plan in data['plans']:
 group=bpy.data.collections.new(plan['id']+' Interior')
 scene.collection.children.link(group)
 full=bpy.data.collections.new(plan['id']+' Full Height')
 cut=bpy.data.collections.new(plan['id']+' Cutaway')
 group.children.link(full);group.children.link(cut)
 for solid in plan['solids']:
  cuboid(plan['id']+'/'+solid['id'],solid['center'],solid['size'],solid['kind'],full)
  if solid['kind'] in ('header','ceiling'):continue
  center=list(solid['center']);size=list(solid['size'])
  if solid['kind']=='wall': center[1]=center[1]-size[1]/2+.45;size[1]=.9
  cuboid(plan['id']+'/'+solid['id']+'/cutaway',center,size,solid['kind'],cut)
 for ramp in plan.get('ramps',[]):
  start,end=ramp['start'],ramp['end'];dx,dz=end[0]-start[0],end[2]-start[2];length=math.hypot(dx,dz)
  side=(-dz/length*ramp['width']/2,dx/length*ramp['width']/2)
  vertices=[]
  for drop in [0,-ramp['thickness']]:
   for point,sign in [(start,-1),(start,1),(end,1),(end,-1)]:
    vertices.append((point[0]+sign*side[0],-point[2]-sign*side[1],point[1]+drop))
  for collection in [full,cut]:
   mesh=bpy.data.meshes.new(ramp['id']);mesh.from_pydata(vertices,[],[(0,1,2,3),(7,6,5,4),(0,4,5,1),(1,5,6,2),(2,6,7,3),(3,7,4,0)]);mesh.update()
   obj=bpy.data.objects.new(ramp['id'],mesh);collection.objects.link(obj);obj.data.materials.append(materials['floor'])
 for mode,collection in [('interior',full),('cutaway',cut)]:
  bpy.ops.object.select_all(action='DESELECT')
  for obj in collection.objects: obj.select_set(True)
  if collection.objects: bpy.context.view_layer.objects.active=collection.objects[0]
  output=OUT/(plan['id']+'-'+mode+'.glb')
  bpy.ops.export_scene.gltf(filepath=str(output),export_format='GLB',use_selection=True,use_active_scene=True,export_yup=True,export_animations=False)
  receipts.append({'building':plan['id'],'mode':mode,'file':str(output),'objects':len(collection.objects),'units':'metres','front':'+Z','stage':'authored prototype; exterior fit pending'})
 # Stage cutaways in a readable 4-column gallery, without changing export origins.
 index=data['plans'].index(plan)
 for obj in full.objects: obj.hide_set(True);obj.hide_render=True
 for obj in cut.objects: obj.location.x+=(index%4)*28;obj.location.y-=(index//4)*28
 title_curve=bpy.data.curves.new(plan['title'],'FONT');title_curve.body=plan['title'];title_curve.size=.8;title_curve.align_x='CENTER'
 title=bpy.data.objects.new(plan['id']+'/title',title_curve);group.objects.link(title)
 title.location=((index%4)*28,-(index//4)*28+plan['footprint'][1]/2+1.5,1.02)
 # Text belongs only to the gallery, never inside exported collision geometry.
 for room in plan['rooms']:
  x,z,w,d=room['rect']
  curve=bpy.data.curves.new(room['name'],'FONT');curve.body=room['name'];curve.size=.35;curve.align_x='CENTER'
  label=bpy.data.objects.new(plan['id']+'/'+room['name'],curve);group.objects.link(label)
  label.location=(x+w/2+(index%4)*28,-z-d/2-(index//4)*28,1.02)
bpy.ops.object.light_add(type='AREA', location=(42,-42,65))
bpy.context.object.data.energy=5000
bpy.context.object.data.shape='DISK';bpy.context.object.data.size=80
bpy.ops.object.camera_add(location=(42,-42,120))
scene.camera=bpy.context.object;scene.camera.data.type='ORTHO';scene.camera.data.ortho_scale=112
scene.render.resolution_x=1800;scene.render.resolution_y=1800;scene.render.resolution_percentage=100
for screen in bpy.data.screens:
 for area in screen.areas:
  if area.type=='VIEW_3D':
   area.spaces.active.region_3d.view_distance=110
   area.spaces.active.region_3d.view_location=(42,-42,0)
scene['source_plan_manifest']=str(OUT/'plans.json')
scene['quality_stage']='Basic authored cutaway prototypes; shell fit and door alignment pending'
(OUT/'blender-export-receipt.json').write_text(json.dumps({'exports':receipts,'sourceWorldSha256':data['worldManifestSha256']},indent=2)+'\n')
print('Interior cutaway scene ready: 14 plans / 28 review GLBs. Existing exterior scene preserved:',previous_scene.name)
