# Blender Recipes

Common workflows assembled from the bpy API building blocks.

## Recipe 1: Low-Poly Landscape

```python
import bpy, bmesh, math, random

# Clear scene
for obj in list(bpy.data.objects): bpy.data.objects.remove(obj, do_unlink=True)

# Create subdivided plane
bpy.ops.mesh.primitive_plane_add(size=20)
plane = bpy.context.active_object
plane.name = 'Terrain'

# Subdivide
mod = plane.modifiers.new('Subdiv', 'SUBSURF')
mod.levels = 5
mod.subdivision_type = 'SIMPLE'
bpy.context.view_layer.objects.active = plane
bpy.ops.object.modifier_apply(modifier='Subdiv')

# Displace vertices for terrain
bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(plane.data)
random.seed(42)
for v in bm.verts:
    v.co.z = random.gauss(0, 0.5) * (1 - abs(v.co.x)/10) * (1 - abs(v.co.y)/10)
bmesh.update_edit_mesh(plane.data)
bpy.ops.object.mode_set(mode='OBJECT')

# Green material
mat = bpy.data.materials.new('Grass')
mat.use_nodes = True
mat.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.15, 0.4, 0.1, 1)
mat.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = 0.9
plane.data.materials.append(mat)

# Sun light
bpy.ops.object.light_add(type='SUN', rotation=(math.radians(45), 0, math.radians(30)))
bpy.context.active_object.data.energy = 3

# Camera
bpy.ops.object.camera_add(location=(12, -12, 8))
cam = bpy.context.active_object
cam.rotation_euler = (math.radians(55), 0, math.radians(45))
bpy.context.scene.camera = cam
```

## Recipe 2: Glass Sphere on Reflective Plane

```python
import bpy, math

for obj in list(bpy.data.objects): bpy.data.objects.remove(obj, do_unlink=True)

# Reflective floor
bpy.ops.mesh.primitive_plane_add(size=20)
floor = bpy.context.active_object
mat_floor = bpy.data.materials.new('Floor')
mat_floor.use_nodes = True
p = mat_floor.node_tree.nodes['Principled BSDF']
p.inputs['Base Color'].default_value = (0.02, 0.02, 0.02, 1)
p.inputs['Metallic'].default_value = 1.0
p.inputs['Roughness'].default_value = 0.05
floor.data.materials.append(mat_floor)

# Glass sphere
bpy.ops.mesh.primitive_uv_sphere_add(radius=1.5, location=(0, 0, 1.5), segments=64, ring_count=32)
sphere = bpy.context.active_object
mod = sphere.modifiers.new('Smooth', 'SUBSURF')
mod.levels = 2
mat_glass = bpy.data.materials.new('Glass')
mat_glass.use_nodes = True
p = mat_glass.node_tree.nodes['Principled BSDF']
p.inputs['Transmission Weight'].default_value = 1.0
p.inputs['IOR'].default_value = 1.45
p.inputs['Roughness'].default_value = 0.0
sphere.data.materials.append(mat_glass)

# Three-point lighting
bpy.ops.object.light_add(type='AREA', location=(4, -3, 5))
bpy.context.active_object.data.energy = 500
bpy.context.active_object.data.size = 3

bpy.ops.object.light_add(type='AREA', location=(-4, -2, 3))
bpy.context.active_object.data.energy = 200
bpy.context.active_object.data.size = 2

bpy.ops.object.light_add(type='AREA', location=(0, 4, 2))
bpy.context.active_object.data.energy = 100
bpy.context.active_object.data.size = 4

# Camera
bpy.ops.object.camera_add(location=(5, -5, 3))
cam = bpy.context.active_object
cam.rotation_euler = (math.radians(70), 0, math.radians(45))
bpy.context.scene.camera = cam

# Cycles for glass
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.samples = 256
```

## Recipe 3: Procedural Donut (Simplified)

```python
import bpy, math

for obj in list(bpy.data.objects): bpy.data.objects.remove(obj, do_unlink=True)

# Torus (donut body)
bpy.ops.mesh.primitive_torus_add(
    major_radius=1.0, minor_radius=0.4,
    major_segments=48, minor_segments=24
)
donut = bpy.context.active_object
donut.name = 'Donut'

# Subdivision for smoothness
mod = donut.modifiers.new('Subdiv', 'SUBSURF')
mod.levels = 2

# Donut material (warm brown)
mat = bpy.data.materials.new('DonutMat')
mat.use_nodes = True
p = mat.node_tree.nodes['Principled BSDF']
p.inputs['Base Color'].default_value = (0.45, 0.22, 0.08, 1.0)
p.inputs['Roughness'].default_value = 0.7
p.inputs['Subsurface Weight'].default_value = 0.3
donut.data.materials.append(mat)

# Icing (duplicate top half, scale up slightly)
bpy.ops.mesh.primitive_torus_add(
    major_radius=1.02, minor_radius=0.42,
    major_segments=48, minor_segments=24
)
icing = bpy.context.active_object
icing.name = 'Icing'

# Pink icing material
mat_icing = bpy.data.materials.new('IcingMat')
mat_icing.use_nodes = True
p = mat_icing.node_tree.nodes['Principled BSDF']
p.inputs['Base Color'].default_value = (0.9, 0.4, 0.5, 1.0)
p.inputs['Roughness'].default_value = 0.3
p.inputs['Coat Weight'].default_value = 0.5
icing.data.materials.append(mat_icing)
```

## Recipe 4: Turntable Animation

```python
import bpy, math

# Assume scene already has objects

# Create empty as rotation center
bpy.ops.object.empty_add(location=(0, 0, 0))
pivot = bpy.context.active_object
pivot.name = 'TurntablePivot'

# Parent camera to pivot
cam = bpy.data.objects.get('Camera')
if cam:
    cam.parent = pivot
    cam.location = (7, 0, 3)
    cam.rotation_euler = (math.radians(75), 0, math.radians(90))

# Animate rotation
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 120  # 5 seconds at 24fps

pivot.rotation_euler = (0, 0, 0)
pivot.keyframe_insert(data_path='rotation_euler', frame=1)
pivot.rotation_euler = (0, 0, math.radians(360))
pivot.keyframe_insert(data_path='rotation_euler', frame=121)

# Make rotation linear (not eased)
for fc in pivot.animation_data.action.fcurves:
    for kp in fc.keyframe_points:
        kp.interpolation = 'LINEAR'

# Render settings
scene.render.filepath = '/tmp/turntable_'
scene.render.image_settings.file_format = 'PNG'
scene.render.resolution_x = 1080
scene.render.resolution_y = 1080
```

## Recipe 5: Render to File and Verify

```python
import bpy, os

scene = bpy.context.scene
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.filepath = '/tmp/blender_render.png'
scene.render.image_settings.file_format = 'PNG'

# Use Cycles for quality
scene.render.engine = 'CYCLES'
scene.cycles.samples = 128
scene.cycles.use_denoising = True

# Render
bpy.ops.render.render(write_still=True)

# Verify
result = os.path.exists('/tmp/blender_render.png')
```

Then from the agent, view the render:
```python
# After execute_blender_code returns, verify and view
from hermes_tools import terminal
terminal("ls -la /tmp/blender_render.png")
# Use vision_analyze to inspect the render
```
