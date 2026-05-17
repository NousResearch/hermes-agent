# XR UI Patterns

Rendering flat Godot UI in 3D world space for Meta Quest VR. Covers SubViewport-on-quad setup, placement rules, curved panels, and text sizing.

## SubViewport-on-Quad Setup

The standard Godot 4.5 approach: render a `Control` scene into a `SubViewport`, then display it on a 3D quad mesh.

### Node Tree

```
UIPanel (Node3D or MeshInstance3D)
├── SubViewport (SubViewport)
│   ├── CanvasLayer
│   │   └── MainControl (Control)
│   │       ├── Panel (PanelContainer or ColorRect)
│   │       ├── TitleLabel (Label)
│   │       ├── DescriptionLabel (Label)
│   │       └── ButtonsContainer (VBoxContainer/HBoxContainer)
│   │           ├── Button1 (Button)
│   │           └── Button2 (Button)
│   └── (optional) SubViewportCamera (Camera2D) for panning
└── QuadMesh (QuadMesh or PlaneMesh)
```

### Setup Script

```gdscript
extends MeshInstance3D

@export var subviewport: SubViewport
@export var quad_size: Vector2 = Vector2(1.0, 0.6)
@export var double_sided: bool = true

func _ready() -> void:
    if not subviewport:
        push_error("SubViewport not assigned")
        return

    # Configure mesh
    var mesh := QuadMesh.new()
    mesh.size = quad_size
    self.mesh = mesh

    # Create material with SubViewport texture
    var mat := StandardMaterial3D.new()
    mat.albedo_texture = subviewport.get_texture()
    mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
    if double_sided:
        mat.cull_mode = BaseMaterial3D.CULL_DISABLED
    material_override = mat

    # Keep SubViewport rendering
    subviewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
    subviewport.size = Vector2i(int(quad_size.x * 1024), int(quad_size.y * 1024))
```

### SubViewport Settings

| Setting | Value | Reason |
|---------|-------|--------|
| `size` | 1024 x 614 (for 1.0 x 0.6 m) | ~1024 px per meter gives readable density |
| `render_target_update_mode` | `UPDATE_ALWAYS` | UI must update every frame for hover/animations |
| `transparent_bg` | `true` if using rounded panels | Allows non-rectangular UI shapes |
| `canvas_item_default_texture_filter` | `Nearest` or `Linear` | `Linear` for smooth text, `Nearest` for pixel art |

## UI Placement Rules

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| Distance from camera | 1.5 - 3.0 m | 2.0 m is the sweet spot for reading |
| Panel width | 1.0 - 2.0 m world units | Maps to 1024-2048 px at 1024 px/m |
| Panel height | 0.5 - 1.2 m world units | Taller for scrollable content |
| Horizontal angle from forward | +/- 30 deg max | Beyond this, text distortion and neck strain increase |
| Vertical offset from eye level | -0.3 to +0.3 m | Looking too far up/down is uncomfortable |
| UI facing | Directly at player | Use `look_at` or place at fixed offset in front of origin |
| Separation from world geometry | 0.05 m buffer | Prevents z-fighting and clipping |

### Billboard / Face-Player Script

```gdscript
extends Node3D

@export var target: Node3D  # Assign XRCamera3D or XROrigin3D
@export var lock_pitch: bool = true
@export var lock_roll: bool = true

func _process(_delta: float) -> void:
    if not target:
        return
    var to_player := target.global_position - global_position
    to_player.y = 0 if lock_pitch else to_player.y
    if to_player.length() > 0.01:
        look_at(target.global_position, Vector3.UP)
    if lock_pitch:
        rotation.x = 0
    if lock_roll:
        rotation.z = 0
```

### Fixed-Offset (HUD-style) Placement

Attach UI as a child of `XROrigin3D` at a fixed offset so it moves with the player:

```gdscript
extends Node3D  # Child of XROrigin3D

@export var distance: float = 2.0
@export var height_offset: float = 0.0

func _process(_delta: float) -> void:
    var camera := get_parent().get_node("XRCamera3D") as XRCamera3D
    if not camera:
        return
    var forward := -camera.global_transform.basis.z
    forward.y = 0
    forward = forward.normalized()
    var target_pos := camera.global_position + forward * distance
    target_pos.y += height_offset
    global_position = target_pos
    look_at(camera.global_position, Vector3.UP)
    rotation.x = 0
    rotation.z = 0
```

## Minimum Text Sizes

At 2 m viewing distance with Quest 3 resolution (~20 PPD), use these minimums on the SubViewport:

| Element | Minimum Size | Comfortable Size |
|---------|--------------|------------------|
| Body text | 28 px | 32-36 px |
| Button labels | 32 px | 36-40 px |
| Headings | 40 px | 48-64 px |
| Icons | 48 x 48 px | 64 x 64 px |
| Touch targets (buttons) | 80 x 80 px | 100 x 100 px |

**Scaling formula:** If panel is at distance `d` meters instead of 2 m, multiply minimum sizes by `d / 2.0`.

```gdscript
func get_scaled_font_size(base_size: int, distance: float) -> int:
    return int(base_size * (distance / 2.0))
```

Use `Theme` resources to manage font sizes consistently across XR UI scenes.

## Curved Panels

Flat panels at wide angles distort at the edges. For wide menus (>1.5 m), curve the mesh to match a cylinder segment.

### Curved Mesh Generation

```gdscript
extends MeshInstance3D

@export var radius: float = 2.0
@export var angle_degrees: float = 60.0
@export var height: float = 0.8
@export var segments: int = 24

func _ready() -> void:
    generate_curved_mesh()

func generate_curved_mesh() -> void:
    var st := SurfaceTool.new()
    st.begin(Mesh.PRIMITIVE_TRIANGLES)

    var angle_rad := deg_to_rad(angle_degrees)
    var start_angle := -angle_rad / 2.0

    for x in range(segments):
        var t0 := float(x) / segments
        var t1 := float(x + 1) / segments
        var a0 := start_angle + t0 * angle_rad
        var a1 := start_angle + t1 * angle_rad

        var y_top := height / 2.0
        var y_bottom := -height / 2.0

        var p0 := Vector3(sin(a0) * radius, y_top, -cos(a0) * radius)
        var p1 := Vector3(sin(a1) * radius, y_top, -cos(a1) * radius)
        var p2 := Vector3(sin(a0) * radius, y_bottom, -cos(a0) * radius)
        var p3 := Vector3(sin(a1) * radius, y_bottom, -cos(a1) * radius)

        var uv0 := Vector2(t0, 0)
        var uv1 := Vector2(t1, 0)
        var uv2 := Vector2(t0, 1)
        var uv3 := Vector2(t1, 1)

        # First triangle
        st.set_uv(uv0); st.add_vertex(p0)
        st.set_uv(uv1); st.add_vertex(p1)
        st.set_uv(uv2); st.add_vertex(p2)
        # Second triangle
        st.set_uv(uv1); st.add_vertex(p1)
        st.set_uv(uv3); st.add_vertex(p3)
        st.set_uv(uv2); st.add_vertex(p2)

    st.generate_normals()
    mesh = st.commit()
```

**UV mapping note:** The generated mesh maps the SubViewport texture across the curved surface. Ensure `SubViewport.size.x` is large enough (2048+) so text remains crisp on the wider curve.

## Pointer / Raycast Interaction

Use the controller forward vector to raycast against UI quads. Convert hit point to SubViewport UV coordinates.

```gdscript
extends XRController3D

@export var ui_panel: MeshInstance3D
@export var subviewport: SubViewport
@export var pointer_length: float = 10.0

var pointer_active: bool = false

func _process(_delta: float) -> void:
    var space_state := get_world_3d().direct_space_state
    var from := global_position
    var to := from + (-global_transform.basis.z) * pointer_length
    var query := PhysicsRayQueryParameters3D.create(from, to)
    query.collision_mask = 2  # UI layer
    var result := space_state.intersect_ray(query)

    if result and result.collider == ui_panel:
        pointer_active = true
        var hit_point: Vector3 = result.position
        var uv := _get_uv_on_mesh(ui_panel, hit_point)
        _send_mouse_event(uv, true)
    else:
        if pointer_active:
            pointer_active = false
            _send_mouse_event(Vector2(-1, -1), false)

func _get_uv_on_mesh(mesh_instance: MeshInstance3D, world_point: Vector3) -> Vector2:
    var local_point := mesh_instance.to_local(world_point)
    # For a quad mesh centered at origin, size (w, h):
    var quad_size: Vector2 = (mesh_instance.mesh as QuadMesh).size
    var uv_x := (local_point.x / quad_size.x) + 0.5
    var uv_y := (-local_point.y / quad_size.y) + 0.5
    return Vector2(uv_x, uv_y)

func _send_mouse_event(uv: Vector2, pressed: bool) -> void:
    var pos := Vector2i(int(uv.x * subviewport.size.x), int(uv.y * subviewport.size.y))
    var evt := InputEventMouseMotion.new()
    evt.position = pos
    evt.global_position = pos
    subviewport.push_input(evt)
```

**Alternative:** Use Godot's built-in `CollisionObject3D.input_event` with a `StaticBody3D` behind the UI quad, then use `get_viewport().get_camera_3d().project_position()` -- but the raycast-to-UV method above is more precise for custom pointer rendering.

## Dynamic UI Positioning

For context-sensitive UI (e.g., object inspection panels), spawn the UI at a computed position:

```gdscript
func place_ui_near_object(ui: Node3D, target_object: Node3D, camera: XRCamera3D) -> void:
    var obj_pos := target_object.global_position
    var cam_pos := camera.global_position
    var dir := (cam_pos - obj_pos).normalized()
    dir.y = 0
    if dir.length() < 0.01:
        dir = Vector3.FORWARD
    dir = dir.normalized()

    # Place 1.5 m from object, toward camera, at eye level
    var ui_pos := obj_pos + dir * 1.5
    ui_pos.y = cam_pos.y
    ui.global_position = ui_pos
    ui.look_at(cam_pos, Vector3.UP)
    ui.rotation.x = 0
    ui.rotation.z = 0
```

## Performance Notes

- One `SubViewport` per UI panel. Reuse the same SubViewport texture across multiple MeshInstances only if showing identical content.
- Update mode `UPDATE_ALWAYS` costs GPU fill rate. For static UI (e.g., info plaques), switch to `UPDATE_ONCE` after content loads, then back to `ALWAYS` only when interaction occurs.
- Keep SubViewport resolution reasonable: 1024x1024 or less for most panels. Higher only for large curved displays.
- Use `TextureRect` with `expand_mode = FIT_WIDTH` inside the SubViewport so UI scales cleanly if you change aspect ratio.
