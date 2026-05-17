---
name: godot-quest-dev
description: Master skill for Meta Quest 3/3S VR development with Godot 4.5+. Covers headless export, OpenXR setup, Meta Vendors plugin integration, manifest requirements, ADB sideload, and on-device debugging. Use when building, exporting, or troubleshooting Godot VR projects for Quest.
category: software-development
version: 1.0.0
author: Hermes VR DevKit
license: MIT
metadata:
  hermes:
    tags: [quest, vr, openxr, godot, godot-4.5, android, linux, meta-quest, sideload]
    related_skills: [blender-godot-pipeline, godot-xr-interactions, quest-native-toolchain, mcp-server-setup]
---

# Godot Quest VR Development

Complete, battle-tested workflow for building and deploying Godot 4.5 VR applications to Meta Quest 3/3S from Linux.

## When to Use

- Creating a new Godot VR project targeting Quest
- Exporting an APK for Quest sideload
- Debugging why the app launches as a 2D screen instead of immersive VR
- Setting up OpenXR, XR controllers, or hand tracking
- Optimizing export pipeline for CI/CD (headless, no GUI)

## Verified Environment

| Component | Version | Status |
|-----------|---------|--------|
| Godot | 4.5-stable | Vulkan crash FIXED (was broken in 4.4.1) |
| Meta OpenXR Vendors plugin | 4.3.1-stable | Compatible with 4.5 (v5.0.1 requires 4.6+) |
| Android API level | 29+ | Required for OpenXR |
| Renderer | Mobile | Forward+ unavailable on Quest |
| Target FPS | 72 (Q2) / 90 (Q3/3S) | Must maintain consistent |

## Quick Start: One-Command Export

```bash
cd /path/to/your-godot-project
GODOT=$HOME/bin/godot ./build-and-run.sh all
```

**Prerequisites (one-time per project):**

```bash
# 1. Install Android build template headlessly (Godot 4.5+)
$GODOT --headless --install-android-build-template
# Creates android/ directory with correct .build_version

# 2. Download and extract Meta OpenXR Vendors plugin
cd /path/to/quest-dev-stack
wget https://github.com/GodotVR/godot_openxr_vendors/releases/download/4.3.1-stable/godotopenxrvendorsaddon.zip
# CRITICAL: zip has 'asset/' prefix
unzip -q godotopenxrvendorsaddon.zip && mv asset/addons /path/to/your-godot-project/ && rm -rf asset

# 3. In Godot editor: Project -> Project Settings -> Plugins -> enable godotopenxrvendors
```

## Core Scene Structure

```
Main (Node3D)
├── XROrigin3D                    <- Player's physical space origin
│   ├── XRCamera3D                <- Head-mounted display
│   ├── XRController3D (Left)     <- Left controller
│   │   └── OpenXRRenderModel     <- Platform-native mesh (Godot 4.5+)
│   └── XRController3D (Right)    <- Right controller
│       └── OpenXRRenderModel
├── WorldEnvironment
└── GameWorld (Node3D)
    └── ... level geometry
```

**Critical rule:** Apply thumbstick locomotion velocity to `XROrigin3D`, NOT the camera.

## Starting the XR Session

```gdscript
extends Node3D

func _ready() -> void:
    var xr_interface: XRInterface = XRServer.find_interface("OpenXR")
    if xr_interface and xr_interface.is_initialized():
        get_viewport().use_xr = true
    else:
        push_error("OpenXR not available")
```

## Export Preset Requirements (Android Quest)

In **Project -> Export -> Android Quest** preset:

| Setting | Value | Why |
|---------|-------|-----|
| `xr_features/xr_mode` | `1` (OpenXR) | Immersive VR, not 2D |
| `gradle_build/use_gradle_build` | `true` | Required for OpenXR loader injection |
| `package/signed` | `false` | Sign manually with apksigner (Godot headless ignores keystore) |
| `package/app_category` | `Game` | **CRITICAL** -- defaults to `Accessibility`, which makes Quest launch as 2D screen |
| Architectures | `arm64` only | Quest is ARM64 |
| Minimum API | `29` | OpenXR requirement |

**Renderer:** Project Settings -> Rendering -> Renderer -> Mobile. Forward+ is unavailable on Quest.

**VSync:** Project Settings -> Display -> Window -> VSync Mode -> Disabled. XR runtime controls frame timing.

## Godot-MCP Live Editor Integration

Control the Godot editor in real-time via MCP tools. Create scenes, add nodes, set properties, attach scripts -- all without touching the GUI.

See `references/godot-mcp-setup.md` for full setup details.

### Quick Hermes Config

```yaml
mcp_servers:
  godot:
    command: node
    args: [/home/YOUR_USER/.local/share/godot-mcp/dist/index.js]
    connect_timeout: 30
    timeout: 120
```

### Critical Bug Fix: WebSocket Protocol Handshake

Godot 4.5's `WebSocketPeer` does **not** negotiate subprotocols. The upstream TypeScript client sends `protocol: 'json'`, which causes the handshake to fail with "socket hang up."

**Fix:** Remove the protocol option from the WebSocket constructor:

```bash
# One-liner patch (apply after any `npm run build`)
sed -i "/protocol: 'json',/d" ~/.local/share/godot-mcp/dist/utils/godot_connection.js
```

Or patch the source so it survives rebuilds:

```bash
# Patch source
sed -i "/protocol: 'json',/d" /path/to/Godot-MCP/server/src/utils/godot_connection.ts
# Then rebuild
cd /path/to/Godot-MCP/server && npm run build
cp -r dist ~/.local/share/godot-mcp/
```

### Available Tools

| Tool | Action |
|------|--------|
| `mcp_godot_create_scene` | New `.tscn` with root node type |
| `mcp_godot_open_scene` | Load existing scene |
| `mcp_godot_create_node` | Add node by type + name |
| `mcp_godot_delete_node` | Remove from tree |
| `mcp_godot_update_node_property` | Set any property |
| `mcp_godot_get_node_properties` | Read node state |
| `mcp_godot_list_nodes` | Scene tree dump |
| `mcp_godot_create_script` | New GDScript |
| `mcp_godot_edit_script` | Modify source |
| `mcp_godot_execute_editor_script` | Run GDScript in editor context |
| `mcp_godot_save_scene` | Persist to disk |
| `mcp_godot_get_current_scene` | Active scene info |

### Godot-MCP Data Retrieval Pitfalls

| Tool | Returns Data? | Workaround |
|------|---------------|------------|
| `execute_editor_script` | No (generic success only) | Use `get_node_properties` or write to temp file |
| `list_nodes` | No (generic success only) | Use `get_node_properties` on known parent |
| `update_node_property` | No | Use `execute_editor_script` with `mark_scene_as_unsaved()` or patch `.tscn` directly |
| `delete_node` | Yes (success) | Also marks scene modified -- preferred over `queue_free()` in scripts |

**Always call `mcp_godot_save_scene` after modifications** to persist changes to disk.

## Headless Export Pipeline

```bash
# Export (unsigned)
$GODOT --headless --export-release "Android Quest" myapp-unsigned.apk

# Sign
apksigner sign --ks debug.keystore --ks-pass pass:android \
  --key-pass pass:android --out myapp.apk myapp-unsigned.apk

# Verify
apksigner verify myapp.apk

# Install
adb install -r myapp.apk

# Launch (prefer UI launch over adb monkey to avoid controller dialog)
adb shell am start -n com.yourcompany.yourapp/com.godot.game.GodotApp
```

**Build template version pinning:** After upgrading Godot, delete `android/` and re-run `--install-android-build-template`. Godot checks `android/.build_version`.

## Manifest Requirements for Immersive VR

For Quest OS to launch in VR mode (not 2D screen), the merged `AndroidManifest.xml` must contain:

- `android.hardware.vr.headtracking` uses-feature
- `com.oculus.intent.category.VR` in MAIN activity intent-filter
- `com.oculus.vr.focusaware` metadata in MAIN activity
- `android:isGame="true"` on application element
- `android:appCategory="game"` on application element
- `extractNativeLibs="true"` for native library extraction

**CRITICAL:** The `appCategory="game"` is set by the **Export Preset -> Package -> App Category** dropdown. Setting it to `Game` is the ONLY way. Patching manifest files manually is overridden by this dropdown.

**CRITICAL:** Godot's build template has TWO manifest files -- `android/build/AndroidManifest.xml` AND `android/build/src/debug/AndroidManifest.xml`. The debug overlay overrides the main. The Meta OpenXR Vendors plugin auto-injects correct entries, but always verify the **App Category dropdown**.

See `references/manifest-requirements.md` for the full manifest anatomy.

## Godot 4.5+ XR Features

### Foveated Rendering (Mobile Vulkan)

Standalone VR headsets now support foveated rendering via `VK_EXT_fragment_density_map`:

1. Ensure Renderer is **Mobile** (not Forward+)
2. In the vendor plugin settings, enable **Foveated Rendering** and choose density level (`LOW`, `MEDIUM`, `HIGH`)
3. Extension is requested automatically at runtime -- no Vulkan code needed

### Application SpaceWarp (ASW)

Frame-synthesis technique for Quest/Pico. GPU renders every other frame; runtime synthesizes missing frames using motion vectors:

1. Update to OpenXR Vendors plugin 4.x (supports ASW)
2. In export preset extras, enable **Application SpaceWarp**
3. Set target frame rate to half native (e.g., 40 Hz on Quest 3 at 80 Hz mode)

**Caution:** Ghosting artifacts on fast-moving objects. Disable per-object if glitches appear.

### OpenXR Render Models

Platform supplies animated, branded controller meshes at runtime. No need to bundle Quest Touch meshes:

```gdscript
@onready var render_model: Node3D = $XRController3D/OpenXRRenderModel

func _ready() -> void:
    render_model.visible = true  # Platform-native controller model
```

Add `OpenXRRenderModel` nodes as children of each `XRController3D` -- plugin populates them automatically.

## Godot 4.6+ Preview Features

### OpenXR 1.1 Support

Godot 4.6 ships native OpenXR 1.1 runtime support. No API changes required -- engine negotiates spec version at startup.

### Spatial Entities (Anchors, Plane Tracking)

```gdscript
var anchor: XRSpatialAnchor = XRSpatialAnchor.new()
anchor.position = world_position
add_child(anchor)
# Runtime keeps locked to real-world location
# Persist anchor UUID to restore on next launch
```

Requires OpenXR Spatial Entities extension enabled in the vendor plugin.

## Debugging on Quest

### Logcat Filtering

```bash
# Real-time logs
adb logcat -s godot:V XR:V DEBUG:V *:S

# Follow stream
adb logcat --follow -s godot

# Crash buffer
adb logcat --buffer crash

# Filter for specific patterns
adb logcat -d | grep -i "fatal\|crash\|exception\|anr"
```

### Symptom Diagnosis

| Symptom | Check | Fix |
|---------|-------|-----|
| App launches as 2D screen | `adb logcat -d \| grep "isVrApplication"` | Set App Category to `Game` in export preset; verify manifest has `com.oculus.intent.category.VR` |
| `XR_ERROR_FORM_FACTOR_UNSUPPORTED` | `adb logcat -s XR` | Missing `libopenxr_loader.so` or VR manifest entries. Plugin handles this if enabled. |
| Black screen in headset | `adb logcat -s godot` for Vulkan errors | Check `use_xr = true` on viewport; verify Mobile renderer; check shader compilation |
| Controller input not firing | OpenXR action map bindings | Check Project Settings -> XR -> OpenXR -> Action Map |
| Frame drops / stuttering | `adb logcat -s VrApi \| grep FPS` | Reduce draw calls (<100/eye), simplify shaders, enable foveated rendering |
| Hand tracking jittery | Raw joint data | Apply smoothing (lerp toward new position each frame) |

### Performance Budgets (Quest 3)

| Metric | Target |
|--------|--------|
| Draw calls | < 100 per eye |
| Tris per frame | < 500k |
| Texture memory | < 1GB total |
| CPU time | < 11ms/frame (90Hz) |
| GPU time | < 11ms/frame |

## Common Pitfalls

- **App Category dropdown overrides manifest patches** -- Always set to `Game`
- `--install-android-build-template` is REQUIRED -- Manual `android/` extraction is silently ignored
- Plugin zip has `asset/` prefix -- Extract with: `unzip -q zip; mv asset/addons .; rm -rf asset`
- `adb monkey -p <pkg> 1` triggers controller dialog -- Use `adb shell am start -n pkg/activity`
- Quest OS may pause app if `headtracking required="false"` -- Ensure `android.hardware.vr.headtracking` uses-feature exists
- Mobile renderer supports glow/bloom but **NOT SSAO** -- Do not enable SSAO in WorldEnvironment
- Godot 4.4.1 Vulkan crash on Quest 3 is **FIXED in 4.5** -- Do not use 4.4.1 for Quest VR
- **Opaque sky domes hide the scene** -- A `SkyBox` mesh with no override material uses default opaque white/gray. Check with `get_surface_override_material(0)` -- if null, the mesh is opaque. Fix: assign a transparent StandardMaterial3D, or scale the dome to 200+ meters so it's far beyond the scene, or remove it and use WorldEnvironment sky only.
- **`execute_editor_script` + `queue_free()` does NOT persist** -- Calling `queue_free()` inside `execute_editor_script` modifies the runtime tree but does NOT trigger the scene modification flag. **Always use `mcp_godot_delete_node` instead** -- it calls `_mark_scene_modified()` internally. Then call `mcp_godot_save_scene`.
- **`update_node_property` does NOT mark scene modified** -- Setting properties via `mcp_godot_update_node_property` changes runtime values but does NOT flag the scene as dirty. **Workaround:** Use `mcp_godot_execute_editor_script` with explicit `editor.get_editor_interface().mark_scene_as_unsaved()` call, or directly patch the `.tscn` file for batch changes. Then `mcp_godot_save_scene`.
- **`execute_editor_script` does NOT return data** -- The command runs successfully but the response is only `{"message": "Command processed", "status": "success"}`. **Use `get_node_properties` or `get_current_scene` for data retrieval**, or run scripts that write to a temp file.
- **`list_nodes` also swallows response data** -- Same as `execute_editor_script`. **Workaround:** Use `get_node_properties` on a known parent, or use `execute_editor_script` to write node names to a file.
- **Don't background Godot on Wayland GUI sessions** -- `terminal(background=true)` with Godot causes the editor window to rapidly open/close/flicker on the user's screen. **Launch Godot in foreground** (or let the user open it manually), then connect MCP tools to the running instance.
- **Godot GLB import fails with "No loader found"** -- Godot caches import metadata (UIDs, `.scn` files) in the `.godot/` folder. If you add a GLB reference to a `.tscn` with a **fake UID** before Godot has imported the file, the loader fails. **Fix sequence:**
  1. Remove the GLB `ext_resource` reference from the `.tscn`
  2. Delete the stale `.godot/` folder entirely
  3. Run headless import to rebuild: `$GODOT --headless --editor --quit-after 20`
  4. Check `.godot/imported/` for the new `.scn` and the `.import` file for the real UID
  5. Re-add the `ext_resource` to `.tscn` with the **real UID**
  6. **Never** fabricate UIDs -- always import first, reference second
- **Stale Godot-MCP node server blocks reconnection** -- If a previous `node ~/.local/share/godot-mcp/dist/index.js` process is still running, Hermes may connect to it but get stale/cached data. **Fix:** `kill $(pgrep -f "godot-mcp/dist/index.js")` before starting a new session. Then verify with `hermes mcp test godot`.
- **execute_editor_script timeout** -- Complex scripts with nested loops may exceed the MCP command timeout. Break into smaller queries.

## References

- `references/godot-mcp-setup.md` -- Godot-MCP installation, protocol fix, and usage
- `references/headless-export.md` -- Complete headless CI/CD export workflow
- `references/manifest-requirements.md` -- Exact manifest entries required for Quest immersive VR
- `references/quest-validation.md` -- Validation checklist for a working Quest build
