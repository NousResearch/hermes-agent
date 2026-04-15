---
name: hermes-blender
description: "Control a running Blender instance programmatically — create objects, set materials, execute Python, render scenes. Uses the blender-mcp addon (socket server on port 9876) for communication. Covers: 3D modeling, materials/shaders, modifiers, animation, rendering, PolyHaven asset integration, and Geometry Nodes."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [Blender, MCP, 3D, creative-coding, modeling, rendering, animation, materials]
    related_skills: [native-mcp, touchdesigner, ascii-video, manim-video]
    security:
      allow_network: true
      allow_install: true
      allow_config_write: true
---

# Blender Integration

## Architecture

Hermes Agent -> Python socket (port 9876) -> Blender MCP Addon -> bpy Python API.

The agent controls a **running Blender instance** via the [blender-mcp](https://github.com/ahujasid/blender-mcp) addon, which creates a TCP socket server inside Blender listening on port 9876.

Two communication methods (use whichever works):
1. **MCP tools** — if configured in Hermes config, tools like `create_object`, `execute_blender_code` are available directly
2. **Direct socket** — `blender_exec()` pattern in `execute_code` for arbitrary bpy commands (always works, no MCP config needed)

## First-Time Setup

### 1. Install Blender

```bash
# macOS
brew install --cask blender

# Linux (snap)
snap install blender --classic

# Or download from https://www.blender.org/download/
```

### 2. Install the Blender MCP Addon

```bash
# Download the addon
curl -L -o /tmp/blender_mcp_addon.py \
  "https://raw.githubusercontent.com/ahujasid/blender-mcp/main/addon.py"
```

Then tell the user:

> Open Blender → Edit → Preferences → Add-ons → Install → select `/tmp/blender_mcp_addon.py`
> Enable the checkbox next to "Interface: Blender MCP"
> Press N in the 3D viewport → find "BlenderMCP" tab → click "Start MCP Server"

### 3. Verify connection

```bash
python3 -c "
import socket, json
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(5)
s.connect(('127.0.0.1', 9876))
s.sendall(json.dumps({'type': 'get_scene_info'}).encode())
data = s.recv(65536)
s.close()
print(json.loads(data))
"
```

### 4. Optional: Configure Hermes MCP

Add under `mcp_servers:` in Hermes config:
```yaml
blender:
  command: uvx
  args: ["blender-mcp"]
  timeout: 120
```

Or with pip: `pip install blender-mcp` then:
```yaml
blender:
  command: python
  args: ["-m", "blender_mcp"]
  timeout: 120
```

## Talking to Blender (the blender_exec pattern)

All communication uses this pattern in `execute_code`:

```python
import json, socket

def blender_exec(code):
    """Execute Python code in Blender via the MCP addon socket."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(30)
    s.connect(('127.0.0.1', 9876))
    cmd = json.dumps({"type": "execute_code", "params": {"code": code}})
    s.sendall(cmd.encode('utf-8'))
    chunks = []
    while True:
        try:
            chunk = s.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            try:
                return json.loads(b"".join(chunks).decode('utf-8'))
            except json.JSONDecodeError:
                continue
        except socket.timeout:
            break
    s.close()
    return json.loads(b"".join(chunks).decode('utf-8'))

# Usage:
result = blender_exec("len(bpy.data.objects)")
# Returns: {"status": "success", "result": "3"}
```

For structured commands (create, modify, delete), use the typed commands:

```python
def blender_cmd(cmd_type, params=None):
    """Send a typed command to Blender."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(30)
    s.connect(('127.0.0.1', 9876))
    cmd = {"type": cmd_type}
    if params:
        cmd["params"] = params
    s.sendall(json.dumps(cmd).encode('utf-8'))
    chunks = []
    while True:
        try:
            chunk = s.recv(65536)
            if not chunk: break
            chunks.append(chunk)
            try:
                return json.loads(b"".join(chunks).decode('utf-8'))
            except json.JSONDecodeError:
                continue
        except socket.timeout:
            break
    s.close()
    return json.loads(b"".join(chunks).decode('utf-8'))

# Examples:
blender_cmd("create_object", {"type": "cube", "name": "MyCube", "location": [0, 0, 1]})
blender_cmd("set_material", {"object_name": "MyCube", "color": [1, 0, 0, 1], "metallic": 0.8})
blender_cmd("get_scene_info")
```

## Workflow

### Step 0: Verify Connection

```python
blender_cmd("get_scene_info")
# Should return scene_name, object_count, objects list
```

### Step 1: Clear Scene + Build

```python
blender_exec("""
import bpy
# Delete all objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Create new objects
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
cube = bpy.context.active_object
cube.name = 'MyCube'
""")
```

### Step 2: Materials

```python
blender_exec("""
import bpy
obj = bpy.data.objects['MyCube']
mat = bpy.data.materials.new('MyMaterial')
mat.use_nodes = True
principled = mat.node_tree.nodes['Principled BSDF']
principled.inputs['Base Color'].default_value = (0.8, 0.1, 0.1, 1.0)
principled.inputs['Metallic'].default_value = 0.9
principled.inputs['Roughness'].default_value = 0.1
obj.data.materials.append(mat)
""")
```

### Step 3: Render

```python
blender_exec("""
import bpy
scene = bpy.context.scene
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.filepath = '/tmp/render_output.png'
scene.render.image_settings.file_format = 'PNG'
bpy.ops.render.render(write_still=True)
""")
```

## Available Commands

| Command | Params | Description |
|---------|--------|-------------|
| `get_scene_info` | — | Scene hierarchy, all objects with transforms |
| `get_object_info` | `name` | Detailed object: verts, faces, materials, children |
| `create_object` | `type, name, location, rotation, scale` | Create mesh, light, camera, empty |
| `modify_object` | `name, location, rotation, scale, visible` | Transform/visibility |
| `delete_object` | `name` | Remove object |
| `execute_code` | `code` | Run arbitrary Python (bpy + math available) |
| `set_material` | `object_name, color, metallic, roughness` | Principled BSDF material |
| `search_polyhaven` | `type, categories, search` | Search free 3D assets |
| `download_polyhaven` | `asset_id, type, resolution, format` | Import PolyHaven models/HDRIs/textures |

## Key Implementation Rules

**All bpy operations must run on the main thread.** The addon handles this via `bpy.app.timers` — commands queued from the socket thread are executed on the main thread at 0.1s intervals.

**Use `execute_code` for anything complex.** The structured commands cover basics, but for modifiers, constraints, Geometry Nodes, animation, or complex node graphs, send raw Python.

**Blender's Python namespace in execute_code:** Only `bpy` and `math` are available. Import anything else inside your code string: `"import bmesh; ..."`.

**Default port is 9876.** If it conflicts, change in both the addon UI and the `blender_exec` function.

**Connection is per-command.** Each `blender_exec()` call opens and closes a socket. This is reliable but adds ~5ms overhead per call. Batch operations in a single `execute_code` call when possible.

## Object Types

| Type | bpy.ops | Notes |
|------|---------|-------|
| `CUBE` | `mesh.primitive_cube_add` | Default 2m cube |
| `SPHERE` | `mesh.primitive_uv_sphere_add` | UV sphere, 32 segments |
| `CYLINDER` | `mesh.primitive_cylinder_add` | 32 vertices |
| `PLANE` | `mesh.primitive_plane_add` | Single face |
| `CONE` | `mesh.primitive_cone_add` | 32 vertices |
| `TORUS` | `mesh.primitive_torus_add` | Major/minor radii |
| `EMPTY` | `object.empty_add` | Transform-only |
| `CAMERA` | `object.camera_add` | Perspective camera |
| `LIGHT` | `object.light_add(type='POINT')` | Point light |
| `SUN_LIGHT` | `object.light_add(type='SUN')` | Directional |
| `SPOT_LIGHT` | `object.light_add(type='SPOT')` | Cone light |
| `AREA_LIGHT` | `object.light_add(type='AREA')` | Rectangular |

## References

| File | Contents |
|------|----------|
| `references/pitfalls.md` | Hard-won lessons from real Blender sessions |
| `references/bpy-api.md` | Essential bpy operations: modeling, materials, modifiers, rendering |
| `references/recipes.md` | Common recipes: procedural modeling, HDRI lighting, animation |
