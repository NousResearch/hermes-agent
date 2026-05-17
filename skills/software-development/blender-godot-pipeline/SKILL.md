---
name: blender-godot-pipeline
description: |
  Use when automating Blender-to-Godot asset pipelines for Quest VR projects. Covers GLB export, gltfpack mesh optimization, Godot Mobile renderer import settings, LOD generation, texture budgets, and scene organization. Use for batch exporting environments, props, and characters from Blender 4.x into a Godot 4.3+ project with Quest 2/3 performance targets.
version: "1.0.0"
author: hermes-vr-devkit
license: MIT
metadata:
  hermes:
    tags: [blender, godot, gltf, glb, quest, vr, pipeline, optimization, mcp]
    related_skills: [godot-quest-dev, godot-xr-interactions, quest-native-toolchain, mcp-server-setup]
---

# Blender-Godot Pipeline for Quest VR

End-to-end pipeline for moving optimized 3D assets from Blender 4.x into Godot 4.3+ for Quest VR deployment.

## Pipeline Overview

```
Blender 4.x          gltfpack          Godot 4.3+
----------          ---------         ------------
Modeling    ---->   -si 0.5    ---->  Import
UV unwrap           -tc 2048          Mobile Renderer
Material            -kn               VR Compression
Lighting bake       -noq              Scene Tree
       |                               |
       v                               v
   .glb (raw)                    .glb.import
   .gltf + .bin                  Material remap
                                 Static typing
```

1. Build/optimize in Blender
2. Export GLB with Godot-compatible settings
3. (Optional) Run gltfpack for LOD/mesh optimization
4. Import into Godot, adjust import settings
5. Instantiate in scene, verify Quest budgets

## Quick Reference: Quest Performance Budgets

| Asset Type | Tri Budget | Texture | Draw Calls | Notes |
|------------|-----------|---------|-----------|-------|
| Hero prop  | 15-30k    | 2K      | 1-2       | Player can approach closely |
| Background | 5-10k     | 1K      | 1         | Far distance, aggressive LOD |
| Environment| 100-200k  | 2K-4K atlas| 5-10   | Total scene, split by room |
| UI panel   | 500       | 1K      | 1         | Unshaded, transparent |
| Character  | 20-40k    | 2K      | 2-3       | Body + head separate materials |
| Particle   | 100-500   | 256     | 1         | GPU particles, billboard |

Target: 72fps on Quest 2, 90fps on Quest 3. Total scene triangles under 300k visible.

## Scene Organization in Blender

Use collections for pipeline stages:

```
Scene Collection
|-- _EXPORT           # Final export collection
|   |-- env_static    # Static environment mesh
|   |-- env_props     # Instanced props (chairs, tables)
|   |-- characters    # Rigged meshes
|-- _BAKE             # Lightmap targets
|-- _REFERENCE        # Blueprints, scale refs
|-- _WIP              # Work in progress
|-- Camera            # Preview camera (1 unit = 1 meter)
|-- Lights            # Bake lights only (not exported)
```

Rules:
- Scale: 1 Blender unit = 1 meter. A standing human is ~1.7 units tall.
- Apply all transforms (Ctrl+A -> All Transforms) before export.
- One material per logical surface. Godot creates one StandardMaterial3D per Blender material slot.
- Name objects with `Category_Name_LodN` convention (e.g., `Chair_Wood_Lod0`).

## GLB Export Constraints (Godot Compatible)

| Feature | Support | Notes |
|---------|---------|-------|
| Meshes | Full | Triangles or quads (auto-triangulated) |
| UVs | Full | Channel 1 = albedo/normal. Channel 2 = lightmap |
| Normals | Full | Use Auto Smooth or custom split normals |
| Materials (Principled BSDF) | Partial | Base color, metallic, roughness, normal, emission, alpha clip |
| Shader nodes | None | Only Principled BSDF exports. No custom node groups. |
| Modifiers | Partial | Apply before export. Armature exports if modifier visible |
| Animations | Full | Actions as glTF animations. NLA tracks recommended |
| Shape keys | Full | Export as morph targets |
| Constraints | None | Bake to keyframes or apply |
| Drivers | None | Bake to keyframes |
| Lights | Partial | glTF punctual lights extension; Godot ignores by default |
| Cameras | Partial | Exports but Godot usually uses its own |
| Empty | Partial | Exported as node with no mesh |

Critical: Godot imports each material slot as a separate StandardMaterial3D. Keep material count low.

## Export Settings (Blender)

In File > Export > glTF 2.0 (.glb/.gltf):

- Format: glTF Binary (.glb)
- Include:
  - Limit to: Visible Objects (or active collection `_EXPORT`)
  - Data > Mesh: Apply Modifiers ON
  - Data > Mesh: Use Auto Smooth ON (or custom normals)
  - Data > Materials: Export ON
- Transform:
  - +Y Up (Godot default)
- Geometry:
  - Loose Edges OFF
  - Loose Points OFF
- Animation (if animated):
  - Limit to Playback Range ON
  - Sampling Rate: 24 or 30

## Mesh Optimization with gltfpack

Install gltfpack: https://github.com/zeux/meshoptimizer/releases

```bash
# Basic optimization for Quest
gltfpack -si 0.5 -tc 2048 -kn -noq -o optimized.glb input.glb

# Flags explained:
# -si 0.5     Simplify to ~50% triangles
# -tc 2048    Resize textures to max 2048
# -kn         Keep named nodes (preserves scene hierarchy)
# -noq        Disable quantization (sometimes needed for Godot compatibility)
# -o          Output file

# Aggressive LOD generation
gltfpack -si 0.25 -tc 1024 -kn -noq -o lod1.glb input.glb
```

Note: Godot 4.3+ has built-in LOD generation on import. gltfpack is optional but gives finer control, especially for environment meshes with many triangles.

## LOD Strategy Code (Godot)

Attach to a MeshInstance3D or use in an import script:

```gdscript
extends MeshInstance3D

@export var lod_distances: Array[float] = [10.0, 25.0, 50.0]
@export var lod_meshes: Array[Mesh] = []

var current_lod: int = -1

func _process(_delta: float) -> void:
    var cam := get_viewport().get_camera_3d()
    if not cam:
        return
    var dist := global_position.distance_to(cam.global_position)
    var target_lod := lod_meshes.size()
    for i in range(lod_distances.size()):
        if dist < lod_distances[i]:
            target_lod = i
            break
    if target_lod != current_lod:
        current_lod = target_lod
        if target_lod < lod_meshes.size():
            mesh = lod_meshes[target_lod]
```

Godot's automatic LOD system (Project Settings > Rendering > Mesh LOD) handles this for most assets. Manual override only needed for hero assets with specific pop distances.

## Texture Guidelines

| Texture | Format | Max Size | Compress | Notes |
|---------|--------|----------|----------|-------|
| Albedo  | PNG/JPG| 2048     | VRAM Compressed (ETC2/ASTC) | Alpha in separate channel if needed |
| Normal  | PNG    | 2048     | VRAM Compressed | OpenGL normal map (Y+) |
| ORM     | PNG    | 1024     | VRAM Compressed | Occlusion(R), Roughness(G), Metallic(B) |
| Emission| PNG    | 512      | VRAM Compressed | Only for glowing objects |
| Lightmap| EXR    | 4096     | VRAM Uncompressed or Basis | Baked in Blender or Godot |

- Use TextureAtlas where possible to reduce draw calls.
- Godot Mobile renderer uses ETC2/ASTC automatically on Quest.
- Avoid texture sizes that are not power-of-two (e.g., 1500x1500).

## Godot Import Settings

When a `.glb` is copied into the Godot project:

1. Select the `.glb` in FileSystem dock
2. In Import tab, change preset or adjust:
   - **Storage**: Mesh storage can be `Built-In` or separate `.mesh`/`.res`
   - **Materials**: `Extract Materials` to editable `.tres` files
   - **Meshes**: `Generate LOD` ON (Godot auto-generates LODs)
   - **Physics**: `Create Collision` if needed (convex or trimesh)
   - **Animation**: `Import Animations` ON, set default loop mode

3. Click **Reimport**

### Mobile Renderer Adjustments

In the imported material `.tres` (if extracted):
- Shading mode: `Per Pixel` (default) or `Toon` for stylized
- Disable `Specular` if not needed (saves ALU)
- Set `Cull Mode` to `Back` unless double-sided required
- Transparency: `Alpha Scissor` cheaper than `Alpha Blend` on Quest
- Disable `Ambient Occlusion` if using baked lightmaps

### Import Cache Invalidation

If Blender re-export does not show in Godot:
1. Delete `.glb.import` file
2. Delete `.godot/imported/` cached version
3. Reimport in Godot (or it auto-reimports on focus)

Or run from CLI:
```bash
# Inside Godot project root
rm -f .godot/imported/*_yourfile*
rm -f assets/models/yourfile.glb.import
# Then reopen Godot or run headless import
godot --headless --import
```

## MCP Automation

If using the Blender MCP server, the pipeline can be scripted:

```python
# Example: export active collection via MCP
import bpy

# Ensure _EXPORT collection exists
export_col = bpy.data.collections.get("_EXPORT")
if not export_col:
    raise RuntimeError("Missing _EXPORT collection")

# Deselect all, select collection objects
bpy.ops.object.select_all(action='DESELECT')
for obj in export_col.all_objects:
    if obj.type in {'MESH', 'ARMATURE', 'EMPTY'}:
        obj.select_set(True)

# Export
bpy.ops.export_scene.gltf(
    filepath="/path/to/project/assets/models/export.glb",
    export_format='GLB',
    use_selection=True,
    export_yup=True,
    export_apply=True,
    export_materials='EXPORT',
)
```

See `references/blender-pitfalls.md` for scripting gotchas.

## .gitignore Additions

Add to project `.gitignore`:

```gitignore
# Blender backup files
*.blend1
*.blend2
*.blend@

# Godot import cache (regenerates)
.godot/imported/
*.tmp

# Exported raw GLBs (if generated in CI)
# Uncomment if you only version Blender sources:
# assets/models/*.glb
# assets/models/*.gltf
# assets/models/*.bin

# gltfpack intermediate
*_gltfpack.glb
```

## Workflow Checklist

- [ ] Blender units = meters, transforms applied
- [ ] Objects named with `Category_Name_LodN`
- [ ] Materials use Principled BSDF only
- [ ] UVs unwrapped, no overlaps on lightmap UV2
- [ ] Export collection `_EXPORT` contains only desired objects
- [ ] Export settings: GLB, visible objects, apply modifiers, Y-up
- [ ] (Optional) gltfpack with `-si 0.5 -tc 2048 -kn -noq`
- [ ] Godot import: generate LOD, extract materials
- [ ] Material: backface culling, appropriate transparency mode
- [ ] Instance in scene, verify draw calls with Godot Debugger > Monitors
- [ ] Test on Quest: verify 72fps with GPU frame time < 13.9ms

## References

- `references/blender-pitfalls.md` — Blender 4.x scripting and export gotchas
- `references/gltf-export.md` — Detailed export settings and compatibility matrix
- `references/godot-import.md` — Godot import settings, material adjustments, cache handling
