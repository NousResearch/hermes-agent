# Headless Blender Automation Recipes

`procedural-3d-studio` generates ready-to-run headless Blender automation scripts.

## Basic Background Rendering

Execute background rendering without opening the GUI:

```bash
blender --background --python <asset_name>.blender.py
```

## Turntable Animation Script (360 Render)

To generate a 36-frame turntable animation of any generated GLB model:

```python
import bpy
import math

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath="models/watchtower.glb")

# Camera on curve / orbit
cam_data = bpy.data.cameras.new("Cam")
cam = bpy.data.objects.new("Cam", cam_data)
bpy.context.collection.objects.link(cam)
bpy.context.scene.camera = cam

# Orbit frames
frames = 36
for f in range(frames):
    angle = (f / frames) * (2 * math.pi)
    cam.location = (4 * math.cos(angle), 4 * math.sin(angle), 2.5)
    cam.rotation_euler = (math.radians(65), 0, angle + math.pi / 2)
    cam.keyframe_insert(data_path="location", frame=f + 1)
    cam.keyframe_insert(data_path="rotation_euler", frame=f + 1)

bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = frames
bpy.context.scene.render.filepath = "renders/turntable_#"
bpy.ops.render.render(animation=True)
```
