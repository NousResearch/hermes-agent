---
name: blender-mcp
description: Drive Blender via the catalog blender MCP, with bpy recipes.
version: 2.1.0
requires: Blender 3.0+ desktop instance (headless via xvfb-run)
author: alireza78a + kshitijk4poor + Hermes Agent
tags: [blender, 3d, animation, modeling, bpy, mcp]
platforms: [linux, macos, windows]
---

# Blender MCP Skill

Companion skill for the `blender` entry in the Hermes MCP catalog. The MCP
server provides the connection to Blender; this skill teaches the bpy idioms
and pitfalls for driving it well. It does not cover Blender UI workflows —
everything here goes through the MCP tools against a live Blender session.

## When to Use

Use when the user wants to create or modify anything in a running Blender
instance: meshes, materials, animations, lighting, renders. Requires the
blender MCP server installed and a Blender desktop session with the addon
connected.

## Prerequisites

1. Install the MCP server from the Nous catalog (one-time):

       hermes mcp install blender

   This configures the pinned `blender-mcp` stdio server with the curated
   tool set: `get_scene_info`, `get_object_info`, `get_viewport_screenshot`,
   `execute_blender_code`.

2. Install the addon inside Blender (one-time — the catalog entry's
   post-install notes cover this too):
   - Download https://raw.githubusercontent.com/ahujasid/blender-mcp/main/addon.py
   - Blender > Edit > Preferences > Add-ons > Install... > select addon.py,
     enable "Interface: Blender MCP".

3. Every session: start Blender FIRST, press N in the viewport, open the
   "BlenderMCP" tab, click "Connect to Claude" (starts the local bridge
   socket). Then start your Hermes session so the MCP tools are loaded.

   The addon refuses to start under `blender -b` (background mode). On a
   machine without a display, run Blender under a virtual one:
   `xvfb-run blender`. GPU rendering works fine under Xvfb.

## Quick Reference

| MCP tool                  | Use for                                    |
|---------------------------|--------------------------------------------|
| `get_scene_info`          | List objects before touching the scene     |
| `get_object_info`         | Inspect one object (transform, materials)  |
| `get_viewport_screenshot` | Visual check of what you built             |
| `execute_blender_code`    | Everything else — arbitrary bpy Python     |

Deeper material lives in the reference files (load on demand):

| Reference | Contents |
|-----------|----------|
| `references/bpy-api.md` | Essential bpy operations: modeling, materials, modifiers, rendering |
| `references/recipes.md` | Complete working scenes: low-poly terrain, glass sphere, HDRI lighting, turntable animation |
| `references/pitfalls.md` | Hard-won lessons: empty code results in 5.x, ops-vs-data API, engine names by version |

Optional asset-service tools (PolyHaven, Sketchfab, Hyper3D, Hunyuan3D) are
disabled by default. If the user has enabled a service in the addon panel,
opt into its tools with `hermes mcp configure blender`.

## Procedure

1. Call `get_scene_info` first — never assume the scene is empty.
2. Build with `execute_blender_code`, in small focused calls (one logical
   step per call: add objects, then materials, then animation). Large
   monolithic scripts hit the bridge timeout.
3. Verify visually with `get_viewport_screenshot` between major steps.
4. Render to an absolute path and tell the user where the file is.

### Common bpy Patterns

Clear scene:

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

Add mesh objects:

    bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(0, 0, 0))
    bpy.ops.mesh.primitive_cube_add(size=2, location=(3, 0, 0))
    bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=2, location=(-3, 0, 0))

Create and assign material:

    mat = bpy.data.materials.new(name="MyMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (R, G, B, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.3
    bsdf.inputs["Metallic"].default_value = 0.0
    obj.data.materials.append(mat)

Keyframe animation:

    obj.location = (0, 0, 0)
    obj.keyframe_insert(data_path="location", frame=1)
    obj.location = (0, 0, 3)
    obj.keyframe_insert(data_path="location", frame=60)

Render to file:

    bpy.context.scene.render.filepath = "/tmp/render.png"
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.ops.render.render(write_still=True)

## Pitfalls

- The blender MCP tools only exist if the server is installed and the session
  started after install. If they're missing, run `hermes mcp install blender`
  and start a new session.
- The addon bridge must be (re)connected inside Blender each Blender session
  (N-panel > BlenderMCP > Connect). "Connection refused" from the tools means
  Blender isn't running or the addon isn't connected — fix that, don't retry.
- Break complex scenes into multiple smaller `execute_blender_code` calls to
  avoid bridge timeouts.
- Render output paths must be absolute (`/tmp/render.png`), not relative —
  they resolve on the BLENDER host's filesystem, which matters if Hermes and
  Blender run on different machines.
- `shade_smooth()` requires the object to be selected and in object mode.
- `execute_blender_code` runs arbitrary Python inside Blender with no sandbox
  — same trust level as the `terminal` tool. Don't paste untrusted code into
  it.
- Do NOT hand-roll raw TCP JSON to port 9876 from `execute_code` — that was
  this skill's pre-MCP workaround. It bypasses the catalog's version pinning
  and tool curation. The MCP tools are the supported path.

## Verification

- `get_scene_info` returns the expected object list after each build step.
- `get_viewport_screenshot` shows the scene you intended.
- After a render, confirm the output file exists and report its absolute
  path to the user.
