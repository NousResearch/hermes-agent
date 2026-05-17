# Godot Import Settings for Mobile Renderer & Quest VR

Godot 4.3+ defaults to Forward+ renderer on desktop. Quest projects must use the Mobile renderer and adjust import settings accordingly.

## Renderer Setup

In `project.godot`:

```ini
[rendering]
renderer/rendering_method="mobile"
renderer/rendering_method.mobile="mobile"
textures/vram_compression/import_etc2_astc=true
```

The Mobile renderer is required for Quest. Forward+ is too expensive and unsupported on Android XR.

## Per-File Import Settings

When a `.glb` is first seen by Godot, it creates a `.glb.import` file. These are the critical settings:

### Meshes

| Setting | Default | Quest Recommendation | Reason |
|---------|---------|----------------------|--------|
| `meshes/create_shadow_meshes` | true | false | Shadow meshes double memory; Quest is tight |
| `meshes/generate_lods` | true | true | Essential for performance |
| `meshes/force_disable_compression` | false | false | Keep OFF to save GPU memory |
| `skins/use_named_skins` | true | true | Needed for humanoid retargeting |

### Materials

| Setting | Default | Quest Recommendation | Reason |
|---------|---------|----------------------|--------|
| `materials/location` | 1 (Node) | 1 or 2 | 1 = built-in, 2 = extract to file |

Extract materials (`location = 2`) if you need to edit them in Godot. Otherwise built-in is fewer files.

### Animation

| Setting | Default | Recommendation |
|---------|---------|----------------|
| `animations/import` | true | true |
| `animations/bake_animation` | true | true (for complex rigs) |
| `animations/optimizer/enabled` | true | true |
| `animations/optimizer/max_angle` | 0.1 | 0.5 (Quest: accept more error for smaller files) |

### Storage

| Setting | Default | Recommendation |
|---------|---------|----------------|
| `nodes/apply_root_scale` | true | true (Blender 1m = Godot 1m) |
| `nodes/root_scale` | 1.0 | 1.0 |

## Extracting and Editing Materials

After import, extract materials to edit:

1. Select `.glb` in FileSystem
2. Import tab > `Materials > Location` = `Extract to File`
3. Reimport

This creates `.tres` files alongside the `.glb`. Edit these for Quest optimization:

```gdscript
# Example material adjustments for Quest
extends StandardMaterial3D

func _init():
    # Disable expensive features
    specular_mode = SPECULAR_DISABLED
    roughness = 0.8
    metallic = 0.0
    # Use alpha scissor for cutouts
    transparency = TRANSPARENCY_ALPHA_SCISSOR
    alpha_scissor_threshold = 0.5
    # Backface culling unless double-sided needed
    cull_mode = CULL_BACK
    # Disable normal map if not visible in VR
    normal_enabled = false
```

In the `.tres` file directly:

```text
[resource]
albedo_color = Color(1, 1, 1, 1)
roughness = 0.8
metallic = 0.0
specular_mode = 0
transparency = 2
alpha_scissor_threshold = 0.5
cull_mode = 2
normal_enabled = false
```

## Import Cache Invalidation

Godot caches imported assets in `.godot/imported/`. When re-exporting from Blender, Godot may not detect changes if timestamps are weird or if the file was replaced.

### Symptoms

- Old mesh still visible after re-export
- Material changes not reflected
- Missing new objects

### Fix

Delete the import metadata and cache:

```bash
# From project root
rm -f assets/models/mymodel.glb.import
rm -rf .godot/imported/*mymodel*
```

Then switch back to Godot; it will auto-reimport. Or run headless:

```bash
godot --headless --import
```

### Force Reimport All

Editor > Tools > Reload Current Project (or restart Godot) sometimes works. For CI:

```bash
# Nuke entire import cache (nuclear option)
rm -rf .godot/imported/
godot --headless --import
```

## Texture Import Settings

Textures imported alongside glTF (embedded or separate) get default settings. Override by selecting the imported texture in FileSystem:

- **Compress**: `VRAM Compressed` (ETC2/ASTC on Quest)
- **Mipmaps**: Generate. Critical for VR to avoid aliasing.
- **Filter**: Linear Mipmap. Nearest only for pixel-art.
- **Repeat**: Enable for tiling textures, disable for atlases.
- **Max Size**: 2048 for hero, 1024 for props, 512 for UI.

For lightmaps (if baked externally):
- **Compress**: `Lossless` or `VRAM Uncompressed` to avoid banding.
- **HDR**: Enable if using EXR.

## Physics Import

If the glTF contains collision proxies (e.g., simple cubes named `UCX_` or `_collision`):

Godot does NOT auto-import collision from glTF. Options:

1. **Import Script**: Assign a post-import script to the `.glb`.
2. **Manual**: Add CollisionShape3D nodes in Godot.
3. **Trimesh**: In import settings, `meshes/create_collision` = `Trimesh` (expensive) or `Convex` (faster).

Example post-import script (`res://scripts/gltf_import.gd`):

```gdscript
@tool
extends EditorScenePostImport

func _post_import(scene: Node) -> Object:
    _process_node(scene)
    return scene

func _process_node(node: Node) -> void:
    if node.name.ends_with("_collision"):
        var parent = node.get_parent()
        if parent is MeshInstance3D:
            var body = StaticBody3D.new()
            parent.add_child(body)
            body.owner = parent.owner
            var shape = CollisionShape3D.new()
            body.add_child(shape)
            shape.owner = parent.owner
            shape.shape = parent.mesh.create_trimesh_shape()
        node.queue_free()
    for child in node.get_children():
        _process_node(child)
```

Attach in Import tab > `Script`.

## Scene Instantiation Best Practices

After importing, instantiate the `.glb` into a scene:

```gdscript
# In a scene script or at runtime
var scene := preload("res://assets/models/room.glb")
var instance := scene.instantiate()
add_child(instance)
```

For static environments, use **Merge Groups** or make the instance local (right-click > Make Local) to edit in place.

For repeated props (chairs, crates):
- Use `MultiMeshInstance3D` for hundreds of identical objects.
- Or simply instance the same imported scene multiple times; Godot shares mesh resources.

## Performance Validation

After import and scene setup, check:

1. **Debugger > Monitors** while running on Quest:
   - Draw calls: aim for < 100 per frame
   - Triangles: < 300k visible
   - Texture memory: < 1GB total
2. **Rendering > Frame Time** with `display/driver/enable_vsync` off for profiling.
3. Use **XR Debugger** or `adb logcat` to check GPU frame times.

Target:
- Quest 2: GPU < 13.9ms (72 FPS)
- Quest 3: GPU < 11.1ms (90 FPS)

## Common Import Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Failed to load resource` | Corrupt glTF or missing `.bin` | Re-export from Blender; validate glTF |
| `No loader found` | Wrong file extension or import plugin missing | Ensure `.glb` or `.gltf` |
| Black material | Missing UVs or texture not found | Check UV0, embed textures in `.glb` |
| Pink material | Shader compilation failed on Mobile renderer | Simplify material; no custom shaders on first test |
| Animation plays once | Loop mode not set | Set `loop_mode = 1` in AnimationPlayer or re-export with NLA |
| Wrong scale | Unit mismatch or root scale applied | Verify Blender 1m = Godot 1 unit; check `apply_root_scale` |

## Summary Table: Quest-Optimized Import Preset

Create a custom import preset in Godot (Import tab > Preset > Save) with these values:

```ini
meshes/create_shadow_meshes=false
meshes/generate_lods=true
skins/use_named_skins=true
animations/import=true
animations/optimizer/enabled=true
animations/optimizer/max_angle=0.5
nodes/apply_root_scale=true
nodes/root_scale=1.0
materials/location=2
```

Apply this preset to all environment and prop imports.
