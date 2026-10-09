# Blender MCP — Pitfalls & Lessons Learned

All interaction goes through the blender MCP tools (`hermes mcp install
blender`): `get_scene_info`, `get_object_info`, `get_viewport_screenshot`,
and `execute_blender_code` for arbitrary bpy Python.

## Setup & Connection

### 1. The addon bridge must be started BEFORE the tools work

The Blender MCP addon opens its local bridge socket only when you click
"Connect to Claude" in the BlenderMCP sidebar tab (N-panel). If the MCP
tools error with "connection refused", the addon isn't connected — fix that
in Blender, don't retry the tool.

**Verify the bridge is up:** `lsof -i :9876 -P -n | grep LISTEN`

### 2. Port 9876 is the addon's default — check for conflicts

Other services may already use 9876. If the tools fail but Blender is
running with the addon started, check with lsof. The port is configurable
in the BlenderMCP addon UI panel.

### 3. Addon installation requires user interaction

Blender addon installation requires the GUI: Edit > Preferences > Add-ons >
Install. The agent cannot automate this. Provide the addon.py path and let
the user install it.

## Python Execution (`execute_blender_code`)

### 4. Only bpy and math are in the namespace

The code runs in a namespace with only `bpy` and `math`. If you need os,
json, bmesh, mathutils, etc., import them inside the code:

```python
import bmesh
bm = bmesh.new()
...
```

### 5. Code result is always empty in Blender 5.x

The addon returns `{"executed": true, "result": ""}` for ALL code — the
eval result is not captured in Blender 5.x. To get values out:
- Use `get_scene_info` or `get_object_info` for queries
- Write results to a temp file and read back:

```python
import json
open('/tmp/result.json', 'w').write(json.dumps([o.name for o in bpy.data.objects]))
```

### 6. Errors come back as error strings — always check

The addon catches exceptions and returns them as error text rather than
crashing. Check the tool result for an error before assuming the code ran.

### 7. bpy.ops require correct context

Many `bpy.ops` functions require the right UI context, which differs when
executing through the bridge. Prefer direct data manipulation:

```python
# Prefer data API over ops
bpy.data.objects.remove(bpy.data.objects['Cube'], do_unlink=True)
```

## Objects & Scene

### 8. Default scene has Cube, Light, Camera

New Blender files start with a Cube at (0,0,0), a Light, and a Camera.
Clear them before building.

### 9. Object names are unique — Blender auto-renames duplicates

Creating an object named "Cube" when one already exists results in
"Cube.001". Always check the actual name via `get_scene_info`.

### 10. Degrees vs radians

The addon's own object-creation commands take degrees and convert
internally, but bpy code in `execute_blender_code` uses radians. Be careful
about the distinction.

## Materials

### 11. Principled BSDF is the default shader

Materials created by the addon use Principled BSDF. For other shader types,
build the node tree manually in `execute_blender_code`.

### 12. Color is RGBA 0-1, not RGB 0-255

Material colors use floating-point RGBA in the 0.0-1.0 range.

## Rendering

### 13. Render blocks the bridge — expect long calls

Rendering is synchronous; the tool call won't return until the render
finishes. Expect renders to take far longer than other calls.

### 14. Engine name varies by Blender version

In Blender 5.x, EEVEE is `'BLENDER_EEVEE'` (not `'BLENDER_EEVEE_NEXT'`,
which was Blender 4.x). Discover available engines at runtime:

```python
import json
open('/tmp/engines.json', 'w').write(json.dumps(
    list(bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items.keys())))
```

Known engine names: `BLENDER_EEVEE`, `BLENDER_WORKBENCH`, `CYCLES`

### 15. GPU rendering requires explicit setup on macOS

Use METAL compute device type on Apple Silicon. Use CUDA or OPTIX on NVIDIA.

## Reliability

### 16. All state lives in Blender's scene data

Each tool call is independent — there is no session state in the bridge.
Anything you need later must exist in the scene (or a file you wrote).

### 17. Blender crash loses the bridge

If Blender crashes, relaunch it, re-enable the addon, and click "Connect to
Claude" again. Save frequently (`bpy.ops.wm.save_mainfile()`).
