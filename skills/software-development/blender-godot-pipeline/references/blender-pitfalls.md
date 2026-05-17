# Blender 4.x Scripting Pitfalls for Pipeline Automation

Blender's Python API changes across 4.x releases. These are the most common gotchas when scripting exports for a Godot pipeline.

## Numpy Array Copy Behavior

Blender's `bpy_prop_collection` and mesh data often return references, not copies. Mutating them can corrupt the blend file silently.

```python
import numpy as np
import bpy

# WRONG: modifies the mesh in place without update
verts = np.array([v.co for v in bpy.context.object.data.vertices])
verts *= 2.0  # This does NOT scale the mesh

# RIGHT: assign back through the API
obj = bpy.context.object
mesh = obj.data
for i, v in enumerate(mesh.vertices):
    v.co = mesh.vertices[i].co * 2.0
mesh.update()

# Or use bmesh for complex ops
import bmesh
bm = bmesh.new()
bm.from_mesh(mesh)
# ... operate on bm ...
bm.to_mesh(mesh)
bm.free()
mesh.update()
```

## Emission Nodes Changed in 4.0+

Blender 4.0 unified emission into Principled BSDF. The old `ShaderNodeEmission` standalone node is no longer the standard path.

```python
# In 4.0+, emission is a socket on Principled BSDF
principled = node_tree.nodes["Principled BSDF"]
principled.inputs["Emission Strength"].default_value = 1.0
principled.inputs["Emission Color"].default_value = (1.0, 0.0, 0.0, 1.0)

# If you still need a separate emission shader (e.g., for additive),
# you must mix it manually; glTF exporter may ignore it.
```

When exporting to glTF, the exporter reads `Emission` from Principled BSDF. Custom emission mixes often fail to export.

## Missing Nodes After Version Upgrade

Opening a 3.x file in 4.x can leave node trees with missing node types (red boxes). Scripting must defensively check:

```python
for node in mat.node_tree.nodes:
    if node.type == 'UNKNOWN':
        # Node type removed or addon missing
        print(f"Warning: unknown node in {mat.name}")
        # Remove or replace before export
        mat.node_tree.nodes.remove(node)
```

## Undo Safety in Headless / Scripted Mode

Blender's undo stack can grow indefinitely in background mode, causing memory bloat in long batch scripts.

```python
# Disable undo for batch operations
bpy.context.preferences.edit.use_global_undo = False

# Or clear after heavy ops
bpy.ops.ed.undo_push(message="Batch step")
# ... after many ops ...
bpy.ops.ed.undo_history_clear()
```

In an MCP/automation context, each command may start a fresh Blender instance, but if running a long script inside one session, monitor undo.

## Bound Box Cache Stale After Mesh Edit

`object.bound_box` is cached. After mesh modifications, update before reading bounds:

```python
obj = bpy.context.object
obj.data.update()  # Force mesh update
# In some cases, depsgraph update is required
dg = bpy.context.evaluated_depsgraph_get()
eval_obj = obj.evaluated_get(dg)
bbox = [eval_obj.matrix_world @ Vector(v) for v in eval_obj.bound_box]
```

This matters for pipeline validation (e.g., checking if mesh fits in a 2m box before export).

## Mesh Scaling and Apply Transforms

Godot expects 1 unit = 1 meter. A common mistake is non-uniform scale on parent empties or armatures.

```python
# Apply all transforms before export
bpy.ops.object.select_all(action='DESELECT')
obj.select_set(True)
bpy.context.view_layer.objects.active = obj
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
```

If an object has negative scale, normals will be inverted in Godot. Always apply transforms on the final export collection.

## Collection Export Context

`bpy.ops.export_scene.gltf` uses the current view layer. Ensure the collection is visible:

```python
# Make sure _EXPORT collection is in the active view layer
export_col = bpy.data.collections["_EXPORT"]
layer_col = bpy.context.view_layer.layer_collection.children.get(export_col.name)
if layer_col:
    bpy.context.view_layer.active_layer_collection = layer_col
else:
    # Link collection to scene if not already linked
    bpy.context.scene.collection.children.link(export_col)
```

## Modifier Visibility

Modifiers must be visible in the viewport to be applied during export with `export_apply=True`. Render visibility does not matter.

```python
for mod in obj.modifiers:
    mod.show_viewport = True
```

## Naming Sanitization

Godot node names allow spaces and special characters, but GDScript access is easier with snake_case. Blender object names can contain dots, which become node paths in Godot.

```python
import re

def sanitize_name(name: str) -> str:
    name = re.sub(r'[^\w\-]', '_', name)
    name = re.sub(r'\.+', '_', name)
    return name

for obj in export_col.all_objects:
    obj.name = sanitize_name(obj.name)
```

## Summary of Safe Scripting Pattern

```python
import bpy
import bmesh
from mathutils import Vector

def safe_export_step():
    # 1. Disable undo
    bpy.context.preferences.edit.use_global_undo = False

    # 2. Validate collection
    col = bpy.data.collections.get("_EXPORT")
    if not col:
        raise RuntimeError("No _EXPORT collection")

    # 3. Apply transforms, ensure modifiers visible
    for obj in col.all_objects:
        if obj.type == 'MESH':
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
            for mod in obj.modifiers:
                mod.show_viewport = True
            obj.data.update()

    # 4. Export
    bpy.ops.export_scene.gltf(
        filepath="/path/to/output.glb",
        export_format='GLB',
        use_active_collection=True,
        export_yup=True,
        export_apply=True,
    )

    # 5. Cleanup undo
    bpy.ops.ed.undo_history_clear()
```
