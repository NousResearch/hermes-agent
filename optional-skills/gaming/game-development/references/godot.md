# Godot (GDScript / C#) Gameplay Programming

Godot is built on **nodes** organized into **scenes**. A scene is a tree of
nodes saved as a reusable unit. The idiomatic Godot mindset: compose behavior
from nodes, communicate with **signals**, and keep scenes small and reusable.

Examples use GDScript (Godot's Python-like language). C# is also fully
supported with the same concepts.

---

## Nodes & Scenes

- **Node** — a single object with one job (Sprite2D, CollisionShape2D,
  AudioStreamPlayer, CharacterBody2D, Timer, …).
- **Scene** — a tree of nodes saved as a `.tscn`. A "Player" scene might be:
  `CharacterBody2D → Sprite2D + CollisionShape2D + Camera2D`.
- **Instancing** — drop a scene inside another scene (a Level instances many
  Enemy scenes). This is Godot's prefab equivalent.

Keep scenes **small and self-contained** so they're reusable and testable
alone (you can run any scene with F6).

---

## The Two Process Callbacks

| Callback | When | Use for |
|---|---|---|
| `_process(delta)` | Every frame | Visuals, input, timers, non-physics |
| `_physics_process(delta)` | Fixed rate | Movement, physics, `move_and_slide()` |

Same fixed/variable split as core-systems: physics → `_physics_process`,
everything else → `_process`.

---

## Player Controller (2D)

```gdscript
extends CharacterBody2D

@export var move_speed: float = 300.0
@export var jump_force: float = 600.0
@export var gravity: float = 1500.0

var _coyote_timer: float = 0.0
const COYOTE_TIME := 0.1
var _jump_buffered: bool = false

func _physics_process(delta: float) -> void:
    # Gravity
    if not is_on_floor():
        velocity.y += gravity * delta
        _coyote_timer -= delta
    else:
        _coyote_timer = COYOTE_TIME

    # Horizontal movement
    var dir := Input.get_axis("move_left", "move_right")
    velocity.x = dir * move_speed

    # Jump with buffer + coyote time (game feel)
    if Input.is_action_just_pressed("jump"):
        _jump_buffered = true
    if _jump_buffered and _coyote_timer > 0.0:
        velocity.y = -jump_force
        _jump_buffered = false
        _coyote_timer = 0.0

    move_and_slide()  # built-in: handles collision response
```

`@export` exposes a variable to the Inspector (tunable without code).
`CharacterBody2D.move_and_slide()` handles collision response for you.

---

## Signals — the native event bus

Signals are Godot's first-class decoupling mechanism. A node emits; others
connect, without the emitter knowing who listens.

```gdscript
# Declaring and emitting
signal died
signal health_changed(new_health: int)

func take_damage(amount: int) -> void:
    health -= amount
    health_changed.emit(health)
    if health <= 0:
        died.emit()
```

```gdscript
# Connecting (in another node)
func _ready() -> void:
    $Player.died.connect(_on_player_died)
    $Player.health_changed.connect(_on_health_changed)
```

For **global** events across distant nodes, use an **autoload singleton** as an
event bus:
```gdscript
# EventBus.gd (autoloaded)
signal enemy_killed(enemy)
signal score_changed(amount)
# Anywhere: EventBus.enemy_killed.emit(self)
```

---

## Enemy AI — FSM with signals

```gdscript
extends CharacterBody2D

enum State { PATROL, CHASE, ATTACK }
@export var enemy_data: Resource   # data-driven (see below)
@export var chase_range := 200.0
@export var attack_range := 40.0

var _state := State.PATROL
var _player: Node2D

func _ready() -> void:
    _player = get_tree().get_first_node_in_group("player")

func _physics_process(_delta: float) -> void:
    var dist := global_position.distance_to(_player.global_position)
    _state = State.ATTACK if dist <= attack_range \
        else State.CHASE if dist <= chase_range \
        else State.PATROL

    match _state:
        State.PATROL: _patrol()
        State.CHASE:  _move_toward(_player.global_position)
        State.ATTACK: _attack()
```

`get_tree().get_first_node_in_group("player")` uses **groups** — Godot's
tag system for finding nodes without hard references.

---

## Resources (data-driven design)

Godot's equivalent of Unity ScriptableObjects — custom data assets:

```gdscript
# enemy_data.gd
class_name EnemyData
extends Resource

@export var enemy_name: String
@export var max_health: int
@export var move_speed: float
@export var damage: int
```

Create `.tres` resource files in the editor, assign to enemies via `@export`.
Designers tune without touching code.

---

## Godot Gotchas

- **`@onready` for node references:** `@onready var sprite = $Sprite2D` — fetches
  after the node tree is ready (using `$` in `_init` fails).
- **`$NodePath` is relative** to the current node; `get_node("/root/...")` for
  absolute.
- **Free nodes with `queue_free()`**, not `free()` — defers deletion safely to
  end of frame.
- **Use groups + signals over `get_node` chains** — hard paths break when you
  reorganize the scene tree.
- **Profile with the built-in Monitor** (Debugger → Monitors) and `--verbose`.
- **GDScript is fast enough for most games**; drop to C# or GDExtension (C++)
  only for proven hot paths.
- **Export templates** must be installed (Editor → Manage Export Templates)
  before you can build.
