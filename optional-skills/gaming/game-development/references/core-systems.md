# Core Systems

Engine-agnostic foundations every game needs. Patterns here are pseudocode/
conceptual — adapt to your engine using the engine reference. Build these once,
reuse forever.

---

## 1. The Game Loop

Every game runs a loop: **input → update → render**, many times per second.
Engines run this for you, but you must understand the two timesteps:

- **Variable timestep** (`update(deltaTime)`) — runs every frame. Use for
  rendering, input polling, animation, UI. Multiply movement by `deltaTime` so
  it's framerate-independent: `position += velocity * deltaTime`.
- **Fixed timestep** (`fixedUpdate()`) — runs at a fixed rate (e.g. 50 Hz).
  Use for **physics** and anything that must be deterministic. Decouples
  simulation from framerate.

**Rule:** gameplay movement that involves physics → fixed; everything visual →
variable. Mixing them up causes jitter or framerate-dependent bugs.

```
// Conceptual
on frame(deltaTime):
    pollInput()
    accumulator += deltaTime
    while accumulator >= FIXED_STEP:
        fixedUpdate(FIXED_STEP)   // physics
        accumulator -= FIXED_STEP
    update(deltaTime)             // visuals, animation
    render(accumulator / FIXED_STEP)  // interpolate for smoothness
```

---

## 2. Input Handling

Don't scatter raw key checks across gameplay code. Centralize into **actions**:

```
// Map physical inputs to abstract actions
actions = {
    "jump":  [Key.Space, Gamepad.A],
    "move":  [Axis.LeftStick, Keys.WASD],
    "fire":  [Mouse.Left, Gamepad.RT],
}
```

Gameplay code asks `input.isPressed("jump")`, never `Key.Space`. Benefits:
rebinding is trivial, controller + keyboard work for free, and platform
differences are isolated.

**Add early:** input **buffering** (queue an action for a few frames so a jump
pressed slightly early still fires) — a core game-feel ingredient.

---

## 3. State Machine (Scene/Game Flow)

Games are state machines. The top-level one manages flow:

```
States: BOOT → MAIN_MENU → PLAYING ⇄ PAUSED → GAME_OVER → MAIN_MENU
```

Implement as an explicit FSM, not a tangle of booleans:

```
class GameStateMachine:
    current: State
    def transition(to): 
        current.exit()
        current = to
        current.enter()
    def update(dt): current.update(dt)
```

Use the same pattern for entity behavior (player: idle/run/jump/fall; enemy:
patrol/chase/attack/flee). FSMs keep behavior readable and debuggable.

---

## 4. Event / Signal Bus

Decouple systems so they don't hold hard references to each other. When the
player takes damage, the HUD, audio, and score systems all react — without the
combat code knowing they exist.

```
EventBus.emit("player_damaged", {amount: 10, source: enemy})

// Elsewhere, independently:
EventBus.on("player_damaged", updateHealthBar)
EventBus.on("player_damaged", playHurtSound)
EventBus.on("player_damaged", triggerScreenShake)
```

Godot has this natively (signals). Unity uses C# events / UnityEvents or a
ScriptableObject event channel. Unreal uses delegates. **Don't overuse it** —
direct calls are fine for tightly-coupled logic; the bus is for cross-system
notifications.

---

## 5. Save / Load & Serialization

Decide early what persists and design a clean save structure:

- **What to save:** progress, unlocks, settings, current run state. NOT
  transient runtime objects.
- **Format:** JSON for readability/debugging (most games); binary for size/
  anti-tamper later.
- **Versioning:** include a `save_version` field from day one so future updates
  can migrate old saves instead of breaking them.
- **Where:** the platform's user-data directory, never the install folder.
- **Safety:** write to a temp file, then atomically rename — so a crash mid-save
  doesn't corrupt the existing save.

```json
{
  "save_version": 1,
  "player": { "level": 5, "xp": 1240, "position": [12.0, 3.5] },
  "unlocks": ["double_jump", "dash"],
  "settings": { "music": 0.8, "sfx": 1.0 }
}
```

---

## 6. Data-Driven Design

Don't hardcode game values in code — put them in data. Enemy stats, weapon
damage, level layouts, item definitions → data files (JSON, CSV, or engine
resources like Unity ScriptableObjects / Godot Resources).

```json
// enemies.json
{ "goblin": { "hp": 30, "speed": 4, "damage": 5, "xp": 10 },
  "ogre":   { "hp": 200, "speed": 2, "damage": 25, "xp": 100 } }
```

Benefits: designers tune without recompiling, balancing is fast, and modding
becomes possible. This single habit dramatically speeds up iteration.

---

## 7. Object Pooling (performance foundation)

Instantiating and destroying objects every frame (bullets, particles, enemies)
causes GC spikes and stutter. Pool them instead:

```
pool = preallocate(N bullets, all inactive)
spawn(): take an inactive bullet, activate it, position it
despawn(b): deactivate b, return it to pool   // never destroy
```

Set this up before you have hundreds of objects on screen — retrofitting is
painful. Covered again in the optimization stage, but the *pattern* belongs in
your core toolkit from the start.

---

## Putting It Together

A minimal but solid foundation, in build order:
1. Game loop + fixed/variable split (the engine gives you this — just respect it)
2. Input action map
3. Top-level state machine (menu/play/pause)
4. Event bus for cross-system messaging
5. Save/load with versioning
6. Data files for tunable values
7. Object pool for spawned entities

With these in place, gameplay programming (Stage 5) becomes assembling systems
rather than fighting architecture.
