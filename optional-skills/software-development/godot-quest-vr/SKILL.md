---
name: godot-quest-vr
description: Build and ship Godot OpenXR apps to Meta Quest.
version: 2.0.0
author: Andre (buckster123) + Hermes Agent
license: MIT
platforms: [linux, macos]
tags: [quest, vr, openxr, godot, android, meta-quest, adb]
metadata:
  hermes:
    tags: [quest, vr, openxr, godot, android, meta-quest, adb, sideload]
    category: software-development
    related_skills: [blender-mcp]
---

# Godot Quest VR Skill

Portable workflow for Godot 4.5+ OpenXR projects targeting Meta Quest 2/3/3S
from a Linux (or macOS) host. Covers toolchain setup, XR scene setup, Android
export, APK signing, ADB sideload, Godot-MCP editor control, and the hard-won
manifest/export pitfalls that make apps launch as flat 2D panels instead of
immersive VR.

Does **not** teach Blender modeling — use the optional `blender-mcp` skill and
`hermes mcp install blender` for DCC work, then import GLB into Godot here.

## When to Use

- New Godot VR project for Quest, or diagnosing a failed immersive launch
- Headless / CI export, signing, and ADB install
- OpenXR session, XROrigin locomotion, controller/hand setup
- Godot-MCP setup (live editor tools) and WebSocket protocol fix
- Android SDK / JDK / apksigner / gltfpack toolchain on a fresh machine

Don't use for: native C++ OpenXR samples only (see Khronos samples), WebXR
browser apps, or Unreal (`unreal-mcp`).

## Prerequisites

| Tool | Typical path / install | Notes |
|------|------------------------|--------|
| Godot 4.5+ | `${HOME}/bin/godot` or PATH | 4.5 fixed Vulkan issues seen on 4.4.1 |
| JDK 17 | distro `openjdk-17-jdk` | Required by Gradle / apksigner |
| Android SDK | `${HOME}/android-sdk` | platform-tools, build-tools 34+, platform 29+ |
| Meta OpenXR Vendors | GodotVR `godot_openxr_vendors` **4.3.1-stable** | 5.0.x needs Godot 4.6+ |
| ADB / apksigner | SDK platform-tools + build-tools | On PATH |
| gltfpack (optional) | `${HOME}/bin/gltfpack` | Mesh/texture optimize before import |
| Node 18+ (optional) | for Godot-MCP server | Only if using live editor MCP |

Env (add to shell profile):

```bash
export ANDROID_HOME="${HOME}/android-sdk"
export ANDROID_SDK_ROOT="${ANDROID_HOME}"
export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-17-openjdk-amd64}"
export PATH="${HOME}/bin:${ANDROID_HOME}/platform-tools:${ANDROID_HOME}/build-tools/34.0.0:${PATH}"
export GODOT="${GODOT:-${HOME}/bin/godot}"
```

On macOS, point `JAVA_HOME` at your Temurin/Homebrew JDK 17 instead.

## How to Run

```bash
# Validate tools
command -v "$GODOT" adb apksigner java

# One-time per project
cd /path/to/your-godot-project
"$GODOT" --headless --install-android-build-template

# Export unsigned → sign → install (see references/export-pipeline.md)
"$GODOT" --headless --export-release "Android Quest" build/app-unsigned.apk
apksigner sign --ks "${KEYSTORE:-${HOME}/.android/debug.keystore}" \
  --ks-pass pass:android --key-pass pass:android \
  --out build/app.apk build/app-unsigned.apk
adb install -r build/app.apk
```

## Quick Reference

### Export preset (Android Quest)

| Setting | Value | Why |
|---------|-------|-----|
| `xr_features/xr_mode` | OpenXR (`1`) | Immersive, not 2D |
| `gradle_build/use_gradle_build` | `true` | Injects OpenXR loader |
| `package/signed` | `false` | Headless ignores keystore for OpenXR |
| `package/app_category` | **Game** (`1`) | Default Accessibility → flat panel |
| Architectures | arm64 only | Quest is ARM64 |
| Min SDK | 29+ | OpenXR |

Renderer: **Mobile**. VSync: **Disabled** (XR runtime owns timing).

### XR scene skeleton

```
Main (Node3D)
└── XROrigin3D          # move this for locomotion, never the camera
    ├── XRCamera3D
    ├── XRController3D (left)
    └── XRController3D (right)
```

```gdscript
func _ready() -> void:
    var xr := XRServer.find_interface("OpenXR")
    if xr and xr.is_initialized():
        get_viewport().use_xr = true
    else:
        push_error("OpenXR not available")
```

### Godot-MCP (optional live editor)

```yaml
# ~/.hermes/config.yaml — Hermes expands ${HOME}, not bare $HOME
mcp_servers:
  godot:
    command: node
    args: ["${HOME}/.local/share/godot-mcp/dist/index.js"]
    connect_timeout: 30
    timeout: 120
```

After every Godot-MCP build, strip the broken subprotocol (Godot 4.5
`WebSocketPeer` does not negotiate `json`):

```bash
# From this skill directory:
bash scripts/fix-godot-mcp-protocol.sh "${HOME}/.local/share/godot-mcp"
```

Full install steps: `references/godot-mcp.md`.

### Blender assets

Install/configure Blender MCP via the official optional skill — do not
duplicate setup here:

```bash
hermes skills install official/creative/blender-mcp
hermes mcp install blender
```

Export GLB (or gltfpack-compressed) → drop into Godot `res://` → import as
raw GLB (avoid quantization extensions Godot rejects). Details:
`references/glb-pipeline.md`.

## Procedure

1. **Toolchain** — `references/android-toolchain.md` until `adb devices` and
   `"$GODOT" --version` work.
2. **Project** — OpenXR enabled; Meta Vendors plugin installed (zip has an
   `asset/` prefix — extract carefully); plugin enabled in Project Settings.
3. **Scene** — XROrigin rig; locomotion on origin; Mobile renderer.
4. **Export preset** — table above; `app_category=Game` is non-negotiable.
5. **Build** — template install → headless unsigned export → `apksigner` →
   `adb install -r`.
6. **Verify immersive** — see checklist; if 2D panel, dump manifest
   (`references/manifest.md`).
7. **Optional MCP** — Godot editor + protocol fix; Blender via `blender-mcp`.

## Pitfalls

1. **`app_category` dropdown overrides manifests** — setting XML alone is not
   enough if the export preset still says Accessibility.
2. **Headless + `package/signed=true`** — keystore path is ignored for OpenXR
   presets; always unsigned export + manual `apksigner`.
3. **Manual `android_source.zip` extract** — ignored; must use
   `--install-android-build-template`.
4. **Vendors zip `asset/` prefix** — wrong depth breaks `libopenxr_loader.so`.
5. **Godot-MCP `$HOME` in YAML** — use `${HOME}` or Hermes passes a literal
   `$HOME/...` string to Node.
6. **Godot-MCP `protocol: 'json'`** — remove or handshake fails on Godot 4.5.
7. **Move camera for locomotion** — breaks tracking; move `XROrigin3D`.
8. **Forward+ renderer** — not available on Quest Mobile.
9. **After Godot upgrade** — delete `android/` and reinstall build template
   (`.build_version` pin).

## Verification

- [ ] `adb devices` shows Quest authorized
- [ ] Export preset: OpenXR + gradle build + **Game** category + unsigned
- [ ] APK contains VR markers (`aapt dump xmltree … \| rg 'VR|headtracking|isGame'`)
- [ ] App launches immersive (not a 2D floating panel)
- [ ] Controllers tracked; stick locomotion moves origin
- [ ] (Optional) `hermes mcp test godot` after editor + protocol fix
- [ ] (Optional) Blender path uses `blender-mcp`, not a second setup doc

## References

| File | Contents |
|------|----------|
| `references/android-toolchain.md` | SDK/NDK/JDK/Godot/gltfpack install |
| `references/export-pipeline.md` | Headless export, sign, install, launch |
| `references/manifest.md` | Immersive VR manifest requirements |
| `references/glb-pipeline.md` | GLB import + gltfpack; Blender handoff |
| `references/xr-basics.md` | Origin rig, controllers, input notes |
| `references/godot-mcp.md` | Godot-MCP build, config, protocol fix |
| `scripts/fix-godot-mcp-protocol.sh` | Idempotent protocol strip helper |
