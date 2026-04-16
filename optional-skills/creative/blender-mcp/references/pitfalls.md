# Blender MCP — Pitfalls & Lessons Learned

## Setup & Connection

### 1. Addon must be started BEFORE connecting

The Blender MCP addon creates the socket server only when you click "Start MCP Server" in the BlenderMCP sidebar tab. If the agent tries to connect before this, you get ConnectionRefusedError on port 9876.

**Verify with:** `lsof -i :9876 -P -n | grep LISTEN`

### 2. Port 9876 is the default — check for conflicts

Other services may use 9876. If connection fails but Blender is running with the addon started, check with lsof. Change the port in the BlenderMCP addon UI panel if needed, and update blender_exec() accordingly.

### 3. The MCP server (uvx blender-mcp) is OPTIONAL for Hermes

The uvx blender-mcp MCP server is a subprocess bridge designed for Claude Desktop. Hermes can talk directly to the addon's socket using blender_exec() — no MCP subprocess needed. The MCP config is optional and adds a layer of indirection.

### 4. Addon installation requires user interaction

Unlike TouchDesigner (where we can paste a script into Textport), Blender addon installation requires the GUI: Edit > Preferences > Add-ons > Install. The agent cannot automate this. Provide the file path and let the user install it.

## Python Execution

### 5. Only bpy and math are in the namespace

The execute_code command provides a namespace with only bpy and math. If you need os, json, bmesh, mathutils, etc., import them inside the code string:

```python
blender_exec("import bmesh; bm = bmesh.new(); ...")
```

### 6. execute_code result is always empty in Blender 5.x addon

The blender-mcp addon's execute_code returns `{"result": {"executed": true, "result": ""}}` for ALL code — both eval and exec. The eval result is not captured in Blender 5.x. To get values:
- Use `get_scene_info` or `get_object_info` for queries
- Write results to a temp file and read back: `blender_exec("import json; open('/tmp/result.json','w').write(json.dumps([o.name for o in bpy.data.objects]))")`

### 7. Errors in execute_code return {"error": "..."} — always check

Always check for the error key in the response. The addon catches exceptions and returns them as error strings rather than crashing.

### 8. bpy.ops require correct context

Many bpy.ops functions require the right context. When executing via the socket, context may differ from the interactive UI. Prefer direct data manipulation:

```python
# Prefer data API over ops
blender_exec("bpy.data.objects.remove(bpy.data.objects['Cube'], do_unlink=True)")
```

## Objects & Scene

### 9. Default scene has Cube, Light, Camera

New Blender files start with a Cube at (0,0,0), a Light, and a Camera. Clear them before building.

### 10. Object names are unique — Blender auto-renames duplicates

Creating an object with name "Cube" when one already exists results in "Cube.001". Always check the returned name.

### 11. Rotation in create_object is in DEGREES, not radians

The addon's create_object command converts degrees to radians internally. But in execute_code, bpy uses radians. Be careful about the distinction.

## Materials

### 12. Principled BSDF is the default shader

All materials created by the addon use Principled BSDF. For other shader types, use execute_code to build the node tree manually.

### 13. Color is RGBA 0-1, not RGB 0-255

Material colors use floating-point RGBA in 0.0-1.0 range.

## Rendering

### 14. Render blocks the connection — set timeout high

Rendering is synchronous. For the agent, the command will take longer to respond. Set timeout to 120s+ for render operations.

### 15. Engine name varies by Blender version

In Blender 5.x, EEVEE is `'BLENDER_EEVEE'` (not `'BLENDER_EEVEE_NEXT'` which was Blender 4.x). Always discover available engines:
```python
blender_exec("import json; open('/tmp/engines.json','w').write(json.dumps(list(bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items.keys())))")
```
Known engine names: `BLENDER_EEVEE`, `BLENDER_WORKBENCH`, `CYCLES`

### 16. GPU rendering requires explicit setup on macOS

Use METAL compute device type on Apple Silicon. Use CUDA or OPTIX on NVIDIA.

## Connection Reliability

### 17. Each command creates a new TCP connection

blender_exec() opens and closes a TCP connection per call. All state lives in Blender's scene data, not the socket.

### 18. Large responses may need multiple recv() calls

The blender_exec() helper loops on recv(65536) until valid JSON is parsed. The 30s timeout handles edge cases.

### 19. Blender crash loses the socket server

If Blender crashes, relaunch it, re-enable the addon, click "Start MCP Server" again. Save frequently.
