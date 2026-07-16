# Essential bpy API Reference

## Scene & Objects

```python
# List all objects
[obj.name for obj in bpy.data.objects]

# Get active object
obj = bpy.context.active_object

# Select object by name
bpy.data.objects['Cube'].select_set(True)

# Delete object by name (no context needed)
bpy.data.objects.remove(bpy.data.objects['Cube'], do_unlink=True)

# Delete all objects
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

# Duplicate object
import bpy
src = bpy.data.objects['Cube']
new_obj = src.copy()
new_obj.data = src.data.copy()
new_obj.name = 'Cube_Copy'
bpy.context.collection.objects.link(new_obj)

# Parent objects
child = bpy.data.objects['Child']
parent = bpy.data.objects['Parent']
child.parent = parent
```

## Transforms

```python
import math, mathutils

obj = bpy.data.objects['Cube']

# Location (world coordinates)
obj.location = (1.0, 2.0, 3.0)
obj.location.x += 0.5

# Rotation (radians)
obj.rotation_euler = (math.radians(45), 0, 0)

# Scale
obj.scale = (2.0, 2.0, 2.0)

# Apply transforms (bake into mesh)
bpy.context.view_layer.objects.active = obj
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
```

## Mesh Creation (bmesh)

```python
import bmesh

# Create mesh from scratch
mesh = bpy.data.meshes.new('CustomMesh')
obj = bpy.data.objects.new('CustomObject', mesh)
bpy.context.collection.objects.link(obj)

bm = bmesh.new()
# Add vertices
v1 = bm.verts.new((0, 0, 0))
v2 = bm.verts.new((1, 0, 0))
v3 = bm.verts.new((0.5, 1, 0))
bm.faces.new((v1, v2, v3))
bm.to_mesh(mesh)
bm.free()
```

## Materials (Principled BSDF)

```python
# Create material
mat = bpy.data.materials.new('MyMat')
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links

principled = nodes['Principled BSDF']
principled.inputs['Base Color'].default_value = (0.8, 0.1, 0.1, 1.0)
principled.inputs['Metallic'].default_value = 0.9
principled.inputs['Roughness'].default_value = 0.1

# Assign to object
obj = bpy.data.objects['Cube']
obj.data.materials.append(mat)

# Emission material
mat = bpy.data.materials.new('GlowMat')
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links
nodes.clear()
output = nodes.new('ShaderNodeOutputMaterial')
emission = nodes.new('ShaderNodeEmission')
emission.inputs['Color'].default_value = (0.0, 1.0, 0.5, 1.0)
emission.inputs['Strength'].default_value = 5.0
links.new(emission.outputs[0], output.inputs[0])
obj.data.materials.append(mat)

# Glass material
mat = bpy.data.materials.new('GlassMat')
mat.use_nodes = True
principled = mat.node_tree.nodes['Principled BSDF']
principled.inputs['Transmission Weight'].default_value = 1.0  # Blender 4.x
principled.inputs['IOR'].default_value = 1.45
principled.inputs['Roughness'].default_value = 0.0
```

## Modifiers

```python
obj = bpy.data.objects['Cube']

# Subdivision Surface
mod = obj.modifiers.new('Subdiv', 'SUBSURF')
mod.levels = 2
mod.render_levels = 3

# Solidify
mod = obj.modifiers.new('Solidify', 'SOLIDIFY')
mod.thickness = 0.05

# Boolean
mod = obj.modifiers.new('Bool', 'BOOLEAN')
mod.operation = 'DIFFERENCE'
mod.object = bpy.data.objects['BoolCutter']

# Array
mod = obj.modifiers.new('Array', 'ARRAY')
mod.count = 5
mod.relative_offset_displace = (1.2, 0, 0)

# Apply modifier
bpy.context.view_layer.objects.active = obj
bpy.ops.object.modifier_apply(modifier='Subdiv')
```

## Camera & Lighting

```python
# Create camera
bpy.ops.object.camera_add(location=(7, -7, 5))
cam = bpy.context.active_object
cam.rotation_euler = (math.radians(63), 0, math.radians(45))
bpy.context.scene.camera = cam

# Camera settings
cam.data.lens = 50  # focal length mm
cam.data.clip_start = 0.1
cam.data.clip_end = 1000

# Point light
bpy.ops.object.light_add(type='POINT', location=(3, 3, 5))
light = bpy.context.active_object
light.data.energy = 1000  # watts
light.data.color = (1.0, 0.9, 0.8)

# Sun light
bpy.ops.object.light_add(type='SUN', rotation=(math.radians(45), 0, 0))
sun = bpy.context.active_object
sun.data.energy = 3

# HDRI world lighting
world = bpy.context.scene.world
if not world:
    world = bpy.data.worlds.new('World')
    bpy.context.scene.world = world
world.use_nodes = True
nodes = world.node_tree.nodes
links = world.node_tree.links
nodes.clear()
output = nodes.new('ShaderNodeOutputWorld')
bg = nodes.new('ShaderNodeBackground')
env = nodes.new('ShaderNodeTexEnvironment')
env.image = bpy.data.images.load('/path/to/hdri.hdr')
links.new(env.outputs[0], bg.inputs[0])
links.new(bg.outputs[0], output.inputs[0])
```

## Rendering

```python
scene = bpy.context.scene

# Resolution
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.resolution_percentage = 100

# Output
scene.render.filepath = '/tmp/render.png'
scene.render.image_settings.file_format = 'PNG'  # PNG, JPEG, OPEN_EXR

# Engine
scene.render.engine = 'CYCLES'  # or 'BLENDER_EEVEE' (5.x) / 'BLENDER_EEVEE_NEXT' (4.x)

# Cycles settings
scene.cycles.samples = 128
scene.cycles.use_denoising = True

# Render
bpy.ops.render.render(write_still=True)

# Animation render
scene.frame_start = 1
scene.frame_end = 250
scene.render.filepath = '/tmp/anim_'
scene.render.image_settings.file_format = 'PNG'
bpy.ops.render.render(animation=True)
```

## Animation (Keyframes)

```python
obj = bpy.data.objects['Cube']
scene = bpy.context.scene

# Set keyframe at frame 1
scene.frame_set(1)
obj.location = (0, 0, 0)
obj.keyframe_insert(data_path='location', frame=1)

# Set keyframe at frame 60
scene.frame_set(60)
obj.location = (5, 0, 3)
obj.keyframe_insert(data_path='location', frame=60)

# Rotation keyframe
obj.rotation_euler = (0, 0, math.radians(360))
obj.keyframe_insert(data_path='rotation_euler', frame=60)

# Material keyframe
mat = obj.data.materials[0]
principled = mat.node_tree.nodes['Principled BSDF']
principled.inputs['Base Color'].default_value = (1, 0, 0, 1)
principled.inputs['Base Color'].keyframe_insert(data_path='default_value', frame=1)
principled.inputs['Base Color'].default_value = (0, 0, 1, 1)
principled.inputs['Base Color'].keyframe_insert(data_path='default_value', frame=60)
```

## Collections

```python
# Create collection
col = bpy.data.collections.new('MyCollection')
bpy.context.scene.collection.children.link(col)

# Move object to collection
col.objects.link(obj)
bpy.context.scene.collection.objects.unlink(obj)

# Hide collection
col.hide_viewport = True
col.hide_render = True
```
