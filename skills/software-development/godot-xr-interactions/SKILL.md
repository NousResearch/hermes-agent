---
name: godot-xr-interactions
description: Use when implementing VR interactions in Godot 4.5+ for Meta Quest -- locomotion, grabbing, hand tracking, passthrough, XR UI, haptics, and OpenXR action maps. Project-agnostic patterns for immersive controller and hand-tracked input.
category: software-development
version: 1.0.0
author: Hermes VR DevKit
license: MIT
metadata:
  hermes:
    tags: [quest, vr, openxr, godot, godot-4.5, xr-interactions, locomotion, hand-tracking, haptics, xr-ui]
    related_skills: [godot-quest-dev, blender-godot-pipeline, quest-native-toolchain, mcp-server-setup]
---

# Godot XR Interactions

Project-agnostic interaction patterns for Godot 4.5+ VR on Meta Quest. Covers locomotion, grabbing, hand tracking, passthrough, XR UI, haptics, and the OpenXR action map.

## When to Use

- Adding teleport, snap-turn, or smooth locomotion to a VR scene
- Implementing grab/pickup mechanics with controllers or hands
- Switching between controller and hand tracking input
- Setting up passthrough (AR) mode
- Building in-world XR UI panels
- Configuring haptic feedback or OpenXR actions

## OpenXR Action Map

Define actions in **Project Settings -> XR -> OpenXR -> Action Map**. Bind them to Quest controller and hand interaction profiles.

| Action | Type | Suggested Bindings |
|--------|------|-------------------|
| `move` | Vector2 | Left thumbstick / thumbstick on hand interaction |
| `turn` | Vector2 | Right thumbstick (X axis) / hand interaction thumbstick |
| `trigger_click` | Bool | Right trigger, Index pinch (hand) |
| `grab_click` | Bool | Right grip, Middle finger pinch (hand) |
| `primary_action` | Bool | A button (right), Y button (left) |
| `secondary_action` | Bool | B button (right), X button (left) |
| `menu` | Bool | Left menu button |
| `haptic` | Haptic | Both controllers |

**Hand interaction profile:** Godot 4.5+ includes `Hand Interaction Profile`. Enable it alongside `Touch Controller Profile` so the same action map works for both controllers and hand tracking without code changes.

## Core Scene Setup for Interactions

```
XROrigin3D
├── XRCamera3D
├── XRController3D (Left)
│   └── GrabArea (Area3D)          <- For controller grab detection
│       └── CollisionShape3D
├── XRController3D (Right)
│   └── GrabArea (Area3D)
│       └── CollisionShape3D
└── LeftHand (XRHandModifier3D)    <- For hand tracking mesh
```

**GrabArea setup:** Attach an `Area3D` to each controller. Give it a small spherical `CollisionShape3D` (radius ~0.05 m) centered on the controller origin. Enable **Monitoring** and **Monitorable**. Use `body_entered` / `body_exited` to detect grabbable objects.

## Grabbing with Controllers

Grabbable objects need a `RigidBody3D` or `StaticBody3D` with a collision shape and a script responding to grab events.

```gdscript
extends RigidBody3D
class_name Grabbable

var is_grabbed: bool = false
var grabber: Node3D = null
var grab_offset: Transform3D = Transform3D.IDENTITY

func _process(_delta: float) -> void:
    if is_grabbed and grabber:
        # Move to controller position plus original offset
        global_transform = grabber.global_transform * grab_offset

func grab(controller: Node3D) -> void:
    if is_grabbed:
        return
    is_grabbed = true
    grabber = controller
    grab_offset = controller.global_transform.affine_inverse() * global_transform
    freeze = true  # Disable physics while held

func release(impulse: Vector3 = Vector3.ZERO) -> void:
    if not is_grabbed:
        return
    is_grabbed = false
    grabber = null
    freeze = false
    if impulse.length() > 0.01:
        apply_central_impulse(impulse)
```

**Grab controller script** (attached to XRController3D):

```gdscript
extends XRController3D

@export var grab_area: Area3D
var held_object: Grabbable = null

func _ready() -> void:
    button_pressed.connect(_on_button_pressed)
    if grab_area:
        grab_area.body_entered.connect(_on_body_entered)

func _on_button_pressed(action: String) -> void:
    if action == "trigger_click":
        if held_object:
            # Release with throw impulse based on controller velocity
            var vel := get_input("velocity") as Vector3
            held_object.release(vel * 1.5)
            held_object = null
        else:
            # Try grab closest grabbable in area
            var closest: Grabbable = null
            var closest_dist := INF
            for body in grab_area.get_overlapping_bodies():
                if body is Grabbable:
                    var d := global_position.distance_to(body.global_position)
                    if d < closest_dist:
                        closest_dist = d
                        closest = body
            if closest:
                held_object = closest
                held_object.grab(self)
```

## Hand Tracking

Enable hand tracking in **Project Settings -> XR -> OpenXR -> Hand Tracking** (`Enabled` or `Optional`). Use `XRHandModifier3D` on a `Skeleton3D` inside a hand mesh, or use `OpenXRHand` node for basic joint visualization.

### Hand Tracking Smoothing

Raw hand joints jitter. Apply exponential smoothing:

```gdscript
extends XRHandModifier3D

@export var smoothing: float = 0.15  # 0 = instant, 1 = frozen
var smoothed_poses: Dictionary = {}

func _process(_delta: float) -> void:
    for j in XRHandTracker.HAND_JOINT_MAX:
        var tracker := XRServer.get_tracker(get_tracker()) as XRHandTracker
        if not tracker:
            continue
        var raw: Transform3D = tracker.get_hand_joint_transform(j)
        var key := str(j)
        if not smoothed_poses.has(key):
            smoothed_poses[key] = raw
        else:
            smoothed_poses[key] = smoothed_poses[key].interpolate_with(raw, 1.0 - smoothing)
        # Apply to skeleton bone if mapping exists
        _apply_to_bone(j, smoothed_poses[key])

func _apply_to_bone(joint: int, t: Transform3D) -> void:
    # Map joint index to Skeleton3D bone index based on your rig
    pass
```

**Pinch detection for hand grabbing:** Check distance between index tip and thumb tip.

```gdscript
func is_pinching(tracker: XRHandTracker, threshold: float = 0.02) -> bool:
    var index_tip := tracker.get_hand_joint_transform(XRHandTracker.HAND_JOINT_INDEX_TIP).origin
    var thumb_tip := tracker.get_hand_joint_transform(XRHandTracker.HAND_JOINT_THUMB_TIP).origin
    return index_tip.distance_to(thumb_tip) < threshold
```

## Haptic Feedback

```gdscript
# Trigger haptic pulse on a controller
func trigger_haptic(controller: XRController3D, amplitude: float = 0.5, duration: float = 0.1) -> void:
    controller.trigger_haptic_pulse("haptic", amplitude, duration, 0.0)
```

**Common haptic patterns:**

| Event | Amplitude | Duration (s) |
|-------|-----------|--------------|
| Hover over grabbable | 0.1 | 0.05 |
| Grab success | 0.6 | 0.1 |
| Release / throw | 0.3 | 0.08 |
| Invalid action | 0.8 | 0.15 |
| Teleport confirm | 0.4 | 0.1 |

## Passthrough (AR Mode)

Enable passthrough on Quest via the Meta OpenXR Vendors plugin.

```gdscript
extends Node3D

func enable_passthrough() -> void:
    var xr_interface := XRServer.find_interface("OpenXR")
    if xr_interface:
        # Requires Meta OpenXR Vendors plugin
        xr_interface.start_passthrough()
        # Set environment to transparent clear color
        get_viewport().transparent_bg = true
        get_viewport().use_xr = true
```

**Requirements:**
- Meta OpenXR Vendors plugin enabled
- `XR_MODE_PASSTHROUGH` or similar mode configured in export settings
- Scene uses transparent background or masked geometry

## XR UI Quick Setup

Render a Godot Control UI onto a 3D quad in the world using SubViewport.

```
UIQuad (MeshInstance3D)
├── SubViewport
│   └── CanvasLayer
│       └── Control
│           └── Button, Label, etc.
└── QuadMesh (size 1.0 x 0.6)
```

```gdscript
extends MeshInstance3D

@export var subviewport: SubViewport

func _ready() -> void:
    var mat := StandardMaterial3D.new()
    mat.albedo_texture = subviewport.get_texture()
    mat.cull_mode = BaseMaterial3D.CULL_DISABLED
    material_override = mat
    subviewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
```

### XR UI Placement Rules

| Rule | Value | Why |
|------|-------|-----|
| Comfortable viewing distance | 1.5 - 3.0 m | Closer causes eye strain; farther reduces legibility |
| Panel width in world units | 1.0 - 2.0 m | Maps to readable pixel density at Quest resolution |
| Minimum text size on SubViewport | 28 px | Scales to readable height at 2 m distance |
| Critical content within horizontal +/- 30 deg | Keep buttons/labels inside this cone | Edge distortion and neck strain beyond |
| Vertical placement | -15 to +15 deg from eye level | Avoid looking too far up/down |
| UI should face player | Use `look_at(player_position)` | Prevents reading at oblique angles |

**Billboard script for UI panels:**

```gdscript
extends Node3D

@export var target: Node3D  # XROrigin3D or XRCamera3D

func _process(_delta: float) -> void:
    if target:
        look_at(target.global_position, Vector3.UP)
        # Only rotate on Y (optional -- keeps panel vertical)
        rotation.x = 0
        rotation.z = 0
```

## Comfort and Safety Rules

| Interaction | Rule |
|-------------|------|
| Snap turn angle | 30 or 45 degrees per click |
| Snap turn cooldown | 0.2 - 0.3 s minimum between turns |
| Smooth turn speed | Max 60 deg/sec; provide snap-turn option |
| Smooth locomotion speed | Max 3-5 m/sec; scale by thumbstick deflection |
| Teleport arc height | Parabolic arc, max ~2 m above ground |
| Teleport surface check | Require valid navmesh or upward-facing normal; reject steep slopes (>30 deg) |
| Vertical camera movement | Never move camera vertically without user control (causes nausea) |
| Acceleration | Instant velocity changes preferred over smooth acceleration for teleport |
| Field of view reduction | Consider vignette during smooth locomotion (optional comfort setting) |

## Controller Velocity for Throwing

Use `XRController3D.get_input("velocity")` (requires velocity action in action map) or compute from transform deltas:

```gdscript
func get_controller_velocity(controller: XRController3D, delta: float) -> Vector3:
    # If action map has velocity input, prefer it:
    var vel := controller.get_input("velocity")
    if vel is Vector3:
        return vel
    # Fallback: differentiate position
    return (controller.global_position - _last_pos) / delta
```

## Input Switching: Controllers vs Hands

Godot 4.5+ automatically switches interaction profiles when the user puts down controllers and shows hands. No code changes needed if you used the OpenXR action map correctly. To detect which mode is active:

```gdscript
func get_active_tracker_name(hand: XRPositionalTracker.TrackerHand) -> String:
    var tracker := XRServer.get_tracker(XRServer.get_tracker_for_hand(hand))
    if tracker:
        return tracker.name
    return ""

# Returns something like "/user/hand/left" -- profile changes automatically
```

## References

- **Locomotion Patterns** (`references/locomotion-patterns.md`): Teleport + snap-turn, smooth locomotion, parabolic arc, surface validation.
- **XR UI Patterns** (`references/xr-ui-patterns.md`): SubViewport-on-quad deep dive, curved panels, text sizing, dynamic placement.
