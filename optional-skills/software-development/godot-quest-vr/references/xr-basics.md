# XR basics (Godot OpenXR)

## Session start

```gdscript
extends Node3D

func _ready() -> void:
    var xr_interface: XRInterface = XRServer.find_interface("OpenXR")
    if xr_interface and xr_interface.is_initialized():
        get_viewport().use_xr = true
    else:
        push_error("OpenXR not available — check Vendors plugin and export XR mode")
```

If `XRServer.primary_interface` is null at `_ready`, defer a frame and retry
(plugin timing on device can lag first frame).

## Origin rig

```
XROrigin3D          # tracking-space root — apply locomotion here
├── XRCamera3D      # HMD
├── XRController3D  # left (tracker: left_hand)
└── XRController3D  # right (tracker: right_hand)
```

**Never** move `XRCamera3D` for thumbstick locomotion — you fight the
tracking system. Move `XROrigin3D`.

## Controllers

- Assign `tracker` names expected by OpenXR (`left_hand` / `right_hand`)
- Prefer an **action map** for inputs; `get_input` without actions can return
  null — null-guard everything
- Button indices vary by runtime; action maps beat magic numbers

Optional: `OpenXRRenderModel` child nodes (Godot 4.5+) for platform controller
meshes.

## Hands / passthrough

Enable the relevant OpenXR extensions via the Meta Vendors plugin settings and
project OpenXR settings. Test on-device early — editor desktop preview is not
a substitute for Quest hand tracking.

## Performance floor

| Target | Quest 2 | Quest 3 / 3S |
|--------|---------|---------------|
| Frame rate | 72 Hz | 90 Hz |
| Renderer | Mobile | Mobile |

Drop dynamic lights, huge real-time shadows, and dense transparent overdraw
before chasing micro-optimizations.

## Camera attributes

On Mobile, physical camera attributes can underexpose dim scenes. Prefer a
practical/simple environment exposure setup for dark VR levels.
