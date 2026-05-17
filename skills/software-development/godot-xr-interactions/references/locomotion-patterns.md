# Locomotion Patterns

Project-agnostic locomotion implementations for Godot 4.5+ Quest VR. Teleport + snap-turn is recommended as the default. Smooth locomotion is optional and should always be user-configurable.

## Recommended: Teleport + Snap Turn

Teleport avoids nausea by eliminating continuous optic flow. Snap turn reduces vestibular conflict.

### Scene Setup

```
XROrigin3D
├── XRCamera3D
├── XRController3D (Left)   <- movement / teleport
└── XRController3D (Right)  <- snap turn
```

### Teleport Script

```gdscript
extends XRController3D

@export var other_controller: XRController3D  # For arc visualization origin if needed
@export var max_distance: float = 15.0
@export var arc_segments: int = 20
@export var valid_color: Color = Color(0.0, 1.0, 0.0, 0.8)
@export var invalid_color: Color = Color(1.0, 0.0, 0.0, 0.8)

var is_teleporting: bool = false
var teleport_target: Vector3 = Vector3.ZERO
var teleport_valid: bool = false

@onready var arc_mesh: MeshInstance3D = $ArcMesh
@onready var target_marker: MeshInstance3D = $TargetMarker

func _process(delta: float) -> void:
    var thumbstick: Vector2 = get_input("move") as Vector2
    if thumbstick.y < -0.5:  # Push forward to aim
        if not is_teleporting:
            is_teleporting = true
            arc_mesh.visible = true
            target_marker.visible = true
        _update_arc()
    else:
        if is_teleporting:
            is_teleporting = false
            arc_mesh.visible = false
            target_marker.visible = false
            if teleport_valid:
                _execute_teleport()

func _update_arc() -> void:
    var origin := global_position
    var forward := -global_transform.basis.z.normalized()
    var up := Vector3.UP
    var velocity := forward * 8.0 + up * 4.0  # Arc impulse

    var points: PackedVector3Array = PackedVector3Array()
    var pos := origin
    var vel := velocity
    var gravity := Vector3.DOWN * 9.8
    var step := 0.05

    teleport_valid = false
    teleport_target = Vector3.ZERO

    for i in range(arc_segments):
        vel += gravity * step
        var next_pos := pos + vel * step
        points.append(pos)

        # Raycast down from arc point to find ground
        var space_state := get_world_3d().direct_space_state
        var query := PhysicsRayQueryParameters3D.create(next_pos, next_pos + Vector3.DOWN * 2.0)
        query.collision_mask = 1  # Ground layer
        var result := space_state.intersect_ray(query)
        if result:
            var hit_normal: Vector3 = result.normal
            var hit_pos: Vector3 = result.position
            var slope := rad_to_deg(acos(hit_normal.dot(Vector3.UP)))
            if slope < 30.0 and origin.distance_to(hit_pos) <= max_distance:
                teleport_valid = true
                teleport_target = hit_pos
                points.append(hit_pos)
                break
        pos = next_pos

    _draw_arc(points)
    target_marker.visible = teleport_valid
    if teleport_valid:
        target_marker.global_position = teleport_target
        _set_arc_color(valid_color)
    else:
        _set_arc_color(invalid_color)

func _draw_arc(points: PackedVector3Array) -> void:
    var immediate := ImmediateMesh.new()
    immediate.surface_begin(Mesh.PRIMITIVE_LINE_STRIP)
    for p in points:
        immediate.surface_add_vertex(p)
    immediate.surface_end()
    arc_mesh.mesh = immediate

func _set_arc_color(c: Color) -> void:
    var mat := StandardMaterial3D.new()
    mat.albedo_color = c
    mat.emission_enabled = true
    mat.emission = c
    arc_mesh.material_override = mat

func _execute_teleport() -> void:
    var origin := get_parent() as XROrigin3D
    if not origin:
        return
    # Move origin so camera ends up at target
    var camera := origin.get_node("XRCamera3D") as XRCamera3D
    var offset := camera.global_position - origin.global_position
    offset.y = 0  # Keep vertical position relative
    origin.global_position = teleport_target - offset
    trigger_haptic_pulse("haptic", 0.4, 0.1, 0.0)
```

### Snap Turn Script

```gdscript
extends XRController3D

@export var snap_angle: float = 45.0
@export var cooldown: float = 0.25
var cooldown_timer: float = 0.0

func _process(delta: float) -> void:
    if cooldown_timer > 0.0:
        cooldown_timer -= delta
        return

    var thumbstick: Vector2 = get_input("turn") as Vector2
    if abs(thumbstick.x) > 0.7:
        var direction := sign(thumbstick.x)
        _snap_turn(direction * snap_angle)
        cooldown_timer = cooldown
        trigger_haptic_pulse("haptic", 0.2, 0.05, 0.0)

func _snap_turn(degrees: float) -> void:
    var origin := get_parent() as XROrigin3D
    if not origin:
        return
    var camera := origin.get_node("XRCamera3D") as XRCamera3D
    # Rotate around camera position to avoid positional offset
    var cam_pos := camera.global_position
    origin.global_rotate(Vector3.UP, deg_to_rad(degrees))
    var offset := cam_pos - camera.global_position
    origin.global_position += offset
```

**Comfort parameters:**

| Parameter | Default | Range |
|-----------|---------|-------|
| Snap angle | 45 deg | 30-45 deg (smaller = more clicks, less disorientation) |
| Cooldown | 0.25 s | 0.2-0.3 s (prevents accidental double-turns) |
| Deadzone | 0.7 | 0.5-0.8 (ignore small stick wobble) |

## Optional: Smooth Locomotion

Only provide as an option. Many users experience nausea with smooth locomotion. Always combine with a comfort vignette or offer teleport as default.

```gdscript
extends XRController3D

@export var max_speed: float = 3.5
@export var deadzone: float = 0.15
@export var use_head_direction: bool = true  # false = controller direction

func _process(delta: float) -> void:
    var input: Vector2 = get_input("move") as Vector2
    if input.length() < deadzone:
        return

    var direction := Vector3.ZERO
    if use_head_direction:
        var camera := get_parent().get_node("XRCamera3D") as XRCamera3D
        var forward := -camera.global_transform.basis.z
        forward.y = 0
        forward = forward.normalized()
        var right := camera.global_transform.basis.x
        right.y = 0
        right = right.normalized()
        direction = forward * input.y + right * input.x
    else:
        var forward := -global_transform.basis.z
        forward.y = 0
        forward = forward.normalized()
        var right := global_transform.basis.x
        right.y = 0
        right = right.normalized()
        direction = forward * input.y + right * input.x

    var origin := get_parent() as XROrigin3D
    if origin:
        origin.global_position += direction * max_speed * delta
```

**Smooth locomotion safety:**

- Clamp speed to 3-5 m/s max
- Do NOT apply vertical movement (no flying without explicit jetpack/grip input)
- Provide snap-turn or smooth-turn option separately
- Consider FOV-reduction vignette during movement

## Optional: Smooth Turn

```gdscript
extends XRController3D

@export var turn_speed: float = 60.0  # degrees per second
@export var deadzone: float = 0.2

func _process(delta: float) -> void:
    var input: Vector2 = get_input("turn") as Vector2
    if abs(input.x) < deadzone:
        return
    var origin := get_parent() as XROrigin3D
    if not origin:
        return
    var camera := origin.get_node("XRCamera3D") as XRCamera3D
    var cam_pos := camera.global_position
    origin.global_rotate(Vector3.UP, deg_to_rad(turn_speed * input.x * delta))
    var offset := cam_pos - camera.global_position
    origin.global_position += offset
```

**Smooth turn max speed:** 60 deg/sec. Higher causes discomfort.

## Surface Validation for Teleport

Always validate the teleport landing zone:

1. **Raycast ground check:** Cast downward from the arc sample point to find floor.
2. **Normal angle:** Reject surfaces where `acos(normal.dot(UP)) > 30 deg`.
3. **Collision mask:** Only collide with designated ground/navmesh layer.
4. **Obstruction check:** After finding a floor hit, raycast from player eye level to the hit point to ensure no wall is in the way.
5. **Distance clamp:** `clamp(distance, 0, max_distance)`.

```gdscript
func _is_valid_teleport_point(point: Vector3, normal: Vector3) -> bool:
    var slope := rad_to_deg(acos(normal.dot(Vector3.UP)))
    if slope > 30.0:
        return false
    # Optional: check if point is inside navmesh or valid region
    # Optional: line-of-sight from arc midpoint
    return true
```

## Parabolic Arc Visualization

The arc in the teleport script uses a simple physics simulation. For a cleaner quadratic Bezier arc:

```gdscript
func _quadratic_arc(start: Vector3, control: Vector3, end: Vector3, segments: int) -> PackedVector3Array:
    var points: PackedVector3Array = PackedVector3Array()
    for i in range(segments + 1):
        var t := float(i) / segments
        var a := start.lerp(control, t)
        var b := control.lerp(end, t)
        points.append(a.lerp(b, t))
    return points
```

Use the controller forward direction to compute `control = start + forward * distance * 0.5 + Vector3.UP * peak_height`.
