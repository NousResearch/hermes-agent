# Godot-Compatible glTF Export Settings

Blender's glTF exporter is the most reliable path into Godot. This reference details the exact settings and edge cases.

## Export Format Choice

| Format | Use Case | Godot Notes |
|--------|----------|-------------|
| glTF Binary (.glb) | Default for single assets | One file, easy to move, recommended |
| glTF Separate (.gltf + .bin + textures) | When editing textures externally | Godot imports the `.gltf`, keep files together |
| glTF Embedded | Rarely used | Bloated file size, not recommended for VR |

**Recommendation:** Use `.glb` for all pipeline stages. Run gltfpack on `.glb` directly.

## Detailed Export Settings

### Include Panel

- **Limit to:** `Visible Objects` or `Active Collection`
  - Use `Active Collection` with a dedicated `_EXPORT` collection for reproducibility.
- **Data > Rendering:**
  - `Use Render Engine`: OFF (uses viewport display, simpler materials)
  - `Active Camera`: OFF unless you need a default camera
- **Data > Mesh:**
  - `Apply Modifiers`: ON. Critical for Subdivision, Mirror, Solidify.
  - `UVs`: ON. Godot needs UV0 for textures. UV1 for lightmaps.
  - `Vertex Colors`: ON if used, but they increase file size.
  - `Attributes`: OFF unless using custom mesh attributes in Godot shaders.
- **Data > Materials:**
  - `Export`: ON. Exports Principled BSDF parameters.
  - `Image Format`: Automatic. PNG for lossless, JPEG for smaller files.
- **Data > Animation:**
  - `Use Current Frame`: OFF (export full timeline)
  - `Limit to Playback Range`: ON
  - `Sampling Rate`: 24 or 30. Lower = smaller files.
  - `Always Sample Animations`: ON. Constraints/drivers bake to keyframes.

### Transform Panel

- **+Y Up**: ON. Godot uses Y-up; this avoids a rotation node on import.
- **Scale**: 1.0. Ensure Blender units are meters.

### Geometry Panel

- `Use Tangents`: ON. Needed for normal mapping in Godot.
- `Loose Edges`: OFF. Not supported by Godot rendering anyway.
- `Loose Points`: OFF.
- `Triangulate`: OFF (Godot triangulates on import, but triangulating in Blender gives control).

### Animation Panel (if applicable)

- `Export Animations`: ON
- `Export Frame Range`: ON
- `Force Sampling`: ON
- `NLA Strips`: ON if using NLA for multiple actions
- `All Actions`: ON for character rigs with multiple clips

## What Does NOT Export

These Blender features are lost in glTF and therefore in Godot:

- Procedural textures (Noise, Voronoi, Musgrave) → bake to image first
- Cycles/Eevee shader nodes beyond Principled BSDF → bake or simplify
- Geometry Nodes → apply as real mesh before export
- Curves/Surfaces/Metaballs → convert to mesh
- Text objects → convert to mesh or use Label3D in Godot
- Physics (rigid body, collision) → rebuild in Godot
- Drivers/Constraints → bake to keyframes or apply
- Layered materials / complex mix shaders → bake to single PBR set
- Subsurface scattering (glTF has limited support; Godot ignores)
- Volumetrics → not supported in glTF or Godot Mobile

## Baking Strategy

If your Blender scene uses complex materials, bake them to a PBR atlas before export:

1. Create a new UV map (`UVMap_Bake`) with no overlaps.
2. Use Blender's Bake (Cycles) to bake:
   - Combined (albedo) → `albedo.png`
   - Roughness → `roughness.png`
   - Metallic → `metallic.png`
   - Normal → `normal.png` (non-color, OpenGL Y+)
3. Replace node tree with single Principled BSDF using baked images.
4. Export. Godot will import the baked textures.

## Code Example: Batch Export Script

Save as `export_collection.py` and run in Blender or via MCP:

```python
import bpy
import os
import sys

# Configuration
OUTPUT_DIR = os.path.expanduser("~/project/assets/models")
COLLECTION_NAME = "_EXPORT"
FORMAT = 'GLB'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get collection
col = bpy.data.collections.get(COLLECTION_NAME)
if not col:
    print(f"Error: collection '{COLLECTION_NAME}' not found", file=sys.stderr)
    sys.exit(1)

# Set as active collection in view layer
vl = bpy.context.view_layer
for lc in vl.layer_collection.children:
    lc.exclude = True
export_lc = vl.layer_collection.children.get(COLLECTION_NAME)
if export_lc:
    export_lc.exclude = False
    vl.active_layer_collection = export_lc
else:
    # Link if missing
    vl.layer_collection.collection.children.link(col)
    vl.active_layer_collection = vl.layer_collection.children[COLLECTION_NAME]

# Prepare objects
bpy.ops.object.select_all(action='DESELECT')
for obj in col.all_objects:
    if obj.hide_viewport:
        continue
    if obj.type in {'MESH', 'ARMATURE', 'EMPTY'}:
        obj.select_set(True)
        # Apply transforms
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

# Sanitize filename
blend_name = bpy.path.basename(bpy.data.filepath).replace(".blend", "")
if not blend_name:
    blend_name = "untitled"
output_path = os.path.join(OUTPUT_DIR, f"{blend_name}.glb")

# Export
bpy.ops.export_scene.gltf(
    filepath=output_path,
    export_format=FORMAT,
    use_selection=True,
    export_yup=True,
    export_apply=True,
    export_materials='EXPORT',
    export_image_format='AUTO',
    export_texcoords=True,
    export_normals=True,
    export_tangents=True,
    export_draco_mesh_compression_enable=False,  # Godot handles compression differently
)

print(f"Exported: {output_path}")
```

## Draco Compression

Blender can export Draco-compressed glTF. Godot supports Draco but decompresses at import time. For Quest VR:

- **Do NOT use Draco** in Blender export. It adds import-time cost and does not reduce runtime memory.
- Use `gltfpack` or Godot's own import compression (`VRAM Compressed`) instead.

## Extension Compatibility

| glTF Extension | Blender Export | Godot Import | Notes |
|----------------|----------------|--------------|-------|
| KHR_materials_unlit | Via Shadeless | Supported | Use for UI, particles |
| KHR_lights_punctual | Yes | Ignored by default | Godot uses its own lights |
| KHR_texture_transform | Yes | Supported | UV offset/scale/tile |
| KHR_materials_clearcoat | Principled BSDF | Partial | Mobile renderer may ignore |
| KHR_materials_transmission | Principled BSDF | Partial | Use alpha modes instead |
| EXT_mesh_gpu_instancing | No | Supported | Use Godot MultiMesh instead |

## Validation

After export, validate with:
- **Online:** https://github.khronos.org/glTF-Validator/
- **CLI:** Install `gltf-validator` npm package

```bash
npx gltf-validator export.glb
```

Invalid glTF may crash Godot importer or produce silent errors.
