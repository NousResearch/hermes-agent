---
name: game-development
description: |
  End-to-end game development — from a blank page to a shipped game. Covers
  concept and design, engine selection, core architecture, gameplay
  programming, art/audio pipelines, polish, optimization, building, and
  publishing across Unity, Unreal, Godot, and web/custom engines.
version: 0.1.0
author: HeLLGURD
license: MIT
platforms: [linux, macos, windows]
category: gaming
triggers:
  - "help me make a game"
  - "I want to build a game"
  - "how do I start a game in Unity / Unreal / Godot"
  - "design my game"
  - "build a game prototype"
  - "implement [game system] in [engine]"
  - "how do I publish my game"
  - "ship my game to Steam / itch.io / mobile"
  - "optimize my game"
toolsets:
  - terminal
  - file
  - web
metadata:
  hermes:
    tags: [Game-Development, Unity, Unreal, Godot, GameDesign, Gameplay, Shipping, Graphics]
    related_skills: [code-wiki, git-workflow, explain-error, pixel-art]
---

# Game Development

A complete, engine-agnostic guide to building a game from the first idea to a
shipped product. This skill is the **orchestrator** for the whole lifecycle: it
helps you scope a concept, pick the right engine, architect the project,
implement gameplay, build the art/audio pipeline, polish, optimize, and
publish — without needing any other tool to fill the gaps.

Deep, engine-specific detail lives in `references/`. The main flow below tells
you *what* to do at each stage and *which reference* to open for the *how*.

---

## When to Use

- User wants to make a game and doesn't know where to start
- User is mid-project and needs help with a specific system or stage
- User is choosing between Unity, Unreal, Godot, or building from scratch
- User wants to take a prototype to a shippable, published game

Do NOT use for:
- Playing or running existing games — see `gaming/pokemon-player`,
  `gaming/minecraft-modpack-server`
- Pure 2D asset creation in isolation — see `creative/pixel-art`
- Generic software architecture unrelated to games — see `code-wiki`

---

## Prerequisites

- `git` for version control (game projects need it from day one).
- An engine installed once chosen (Unity Hub, Unreal/Epic Launcher, or
  Godot — Godot needs no account and is the lightest to start).
- `web` toolset for fetching engine docs / asset store info when needed.
- For builds: platform SDKs (covered per-platform in `references/shipping.md`).

---

## The Lifecycle (and which reference to read)

```
1. CONCEPT & DESIGN ........... define the game, scope it, write it down
2. ENGINE SELECTION ........... pick the right tool → references/engine-selection.md
3. PROJECT ARCHITECTURE ....... structure, version control, build pipeline
4. CORE SYSTEMS ............... game loop, input, state, save → references/core-systems.md
5. GAMEPLAY PROGRAMMING ....... engine-specific → references/{unity,unreal,godot}.md
6. ART & AUDIO PIPELINE ....... import, animate, integrate assets
7. POLISH & GAME FEEL ......... juice, VFX, UI/UX
8. OPTIMIZATION ............... profiling, frame budget, memory
9. TESTING & QA ............... playtesting, bug tracking, builds for testers
10. BUILD & PUBLISH ........... platforms, stores → references/shipping.md
11. POST-LAUNCH .............. patches, analytics, community
```

Most users won't go top-to-bottom in one session. Identify which stage they're
at, work that stage, and point at the next.

---

## Stage 1 — Concept & Design

Before any code, lock down what the game *is*. A vague concept is the #1 reason
projects die.

Ask the user (or help them answer):
1. **Core fantasy** — what does the player *feel* like? ("a nimble ninja", "a
   tycoon building an empire")
2. **Core loop** — the 30-second action repeated: e.g. *explore → fight →
   loot → upgrade → explore*. If you can't state it in one sentence, the
   design isn't ready.
3. **Genre + references** — 2–3 existing games it's like, and the one thing it
   does differently.
4. **Scope reality check** — solo/small team? First game? Then it must be
   *small*. Steer away from MMOs, open-worlds, and "the next Skyrim."
5. **Win/lose + progression** — how does a session start, succeed, fail, and
   what carries over?

Produce a one-page design doc (write to
`~/.hermes/gamedev/<project>/design.md`):

```markdown
# <Title> — Design Doc

**Core fantasy:** ...
**Core loop:** ... (one sentence)
**Genre / references:** ... (X meets Y, but with Z)
**Platform target:** PC / mobile / web
**Scope:** solo, ~N months, vertical slice first
**MVP feature list:** (5-8 items max — everything else is "later")
**Win / lose / progression:** ...
```

**Scope discipline is the whole game here.** Cut features ruthlessly. The MVP
list is what ships; everything else is a wish list.

---

## Stage 2 — Engine Selection

The biggest early decision. Don't default to "whatever's popular" — match the
engine to the game and the developer.

→ **Read `references/engine-selection.md`** for the full decision matrix.

Quick guide:

| If… | Use |
|---|---|
| 2D, solo/indie, want fast iteration, free & open | **Godot** |
| 3D, want huge ecosystem + asset store + mobile | **Unity** |
| High-end 3D visuals, AAA fidelity, large team | **Unreal** |
| Tiny web game, full control, learning fundamentals | **Web (HTML5 canvas / JS / TS)** |
| Pure logic prototype, no graphics yet | **Python + Pygame** |

Once chosen, scaffold the project (engine-specific in the reference) and
**initialize git immediately** with a proper engine `.gitignore`.

---

## Stage 3 — Project Architecture

Set up for maintainability before the codebase grows:

- **Version control** — `git init`, add the engine's `.gitignore` (Unity,
  Unreal, and Godot each have a canonical one on github.com/github/gitignore).
  Use Git LFS for binary assets (textures, audio, models) — they bloat history
  fast. See the `git-workflow` skill for branching.
- **Folder structure** — separate `scripts/`, `assets/` (art/audio/models),
  `scenes/`/`levels/`, and `prefabs/`/`blueprints/`. Keep a flat, predictable
  layout.
- **Naming conventions** — pick one (PascalCase for scripts, snake_case for
  assets) and stay consistent.
- **Build pipeline** — even early, know how to produce a runnable build (Stage
  10). A game you can't build is a game you can't share.

---

## Stage 4 — Core Systems

The engine-agnostic foundations every game needs.

→ **Read `references/core-systems.md`** for implementation patterns.

Covers: the game loop (fixed vs variable timestep), input handling and
rebinding, the scene/state machine (menu → play → pause → game-over), an
event/signal bus for decoupling, save/load and serialization, and a simple
data-driven design approach (stats in data files, not hardcoded).

Build these once, reuse them across every project.

---

## Stage 5 — Gameplay Programming

Now the actual game. This is engine-specific:

- **Unity (C#)** → **`references/unity.md`** — MonoBehaviour lifecycle,
  prefabs, the new Input System, physics, coroutines, ScriptableObjects,
  common player-controller and enemy-AI patterns.
- **Unreal (C++ / Blueprints)** → **`references/unreal.md`** — Actors &
  Components, the Gameplay Framework, Blueprints vs C++, the reflection/UPROPERTY
  system, Enhanced Input, and when to drop from Blueprint to C++.
- **Godot (GDScript / C#)** → **`references/godot.md`** — nodes & scenes,
  signals, the `_process`/`_physics_process` split, `@export` vars,
  groups, and resource-based data.

Each reference includes ready-to-adapt patterns for the systems most games
need: player movement, camera, collision/damage, spawning, and a basic enemy
AI state machine.

---

## Stage 6 — Art & Audio Pipeline

Getting assets into the game correctly matters as much as making them.

- **2D:** sprite import settings (filter mode — point for pixel art!), atlases,
  sprite sheets, frame-based animation. Pair with `creative/pixel-art` for
  creation.
- **3D:** model formats (glTF/FBX), scale and orientation conventions, import
  settings, materials/PBR textures, rigging and animation retargeting.
- **Animation:** state machines / animation trees, blending, root motion vs
  in-place, triggering from gameplay code.
- **Audio:** SFX (one-shots, pooling to avoid cutoff), music (looping, layered
  stems, ducking under SFX), spatial/3D audio, a simple audio manager with
  volume buses (master/music/SFX).

Keep an asset budget — track texture sizes and poly counts early; they're the
first thing that wrecks performance and build size.

---

## Stage 7 — Polish & Game Feel

The difference between "a prototype" and "a game." This is where players feel it.

- **Juice:** screen shake, hit-stop/freeze frames, particle bursts, squash &
  stretch, tweened scale/position (easing, not linear).
- **Feedback:** every action needs an audible + visual response. Damage flashes,
  pickup pops, button hover/press states.
- **Game feel:** input buffering, coyote time (jump grace after leaving a
  ledge), acceleration/deceleration curves instead of instant velocity.
- **UI/UX:** clear menus, readable HUD, controller + keyboard support,
  resolution/aspect handling, accessibility (remappable controls, colorblind-
  safe palettes, text size).
- **Game juice rule:** add feedback until it feels *slightly* too much, then
  pull back 10%.

---

## Stage 8 — Optimization

Only after it works — premature optimization wastes the early game.

- **Profile first** — use the engine profiler (Unity Profiler, Unreal Insights,
  Godot's monitor) to find the *actual* bottleneck. Never guess.
- **Frame budget** — 16.6 ms for 60 FPS. Know where your milliseconds go (CPU
  vs GPU vs render).
- **Common wins:** object pooling (don't instantiate/destroy in loops), draw-call
  batching, texture atlasing, LODs for 3D, culling off-screen work, caching
  component lookups, avoiding per-frame allocations (GC spikes).
- **Memory:** watch texture/audio memory and load/unload by level. Profile build
  size before it balloons.

---

## Stage 9 — Testing & QA

- **Playtest early and with real people** — you're blind to your own game's
  problems. Watch silently; don't explain — if they're confused, the game is.
- **Bug tracking** — even a simple `bugs.md` or a GitHub issues board. Repro
  steps, severity, platform.
- **Test builds** — produce a build for testers (not just play-in-editor; bugs
  hide in builds). itch.io draft pages or direct zips work for early testing.
- **Edge cases** — alt-tab, window resize, controller unplug, save mid-action,
  rapid input.

---

## Stage 10 — Build & Publish

Turn the project into a real, distributable game.

→ **Read `references/shipping.md`** for per-platform build + store details.

Covers: building for Windows/macOS/Linux, web (WASM/HTML5), and mobile
(iOS/Android); store setup for **Steam** (Steamworks, depots, store page,
wishlists), **itch.io** (butler, pages, the indie-friendly path), and the
**mobile stores**; plus signing, age ratings, capsule art/trailers, and launch
checklist.

---

## Stage 11 — Post-Launch

Shipping is the start, not the end.

- **Patches** — hotfix critical bugs fast; batch smaller fixes.
- **Analytics** — track where players drop off, what they do, crash reports.
- **Community** — Discord/forums, respond to feedback, communicate roadmap.
- **Content updates** — if it has traction, plan updates around player data,
  not assumptions.

---

## Working Style for This Skill

- **Meet the user at their stage.** Don't force a beginner through Unreal C++;
  don't re-explain the game loop to a veteran. Read their level from how they
  ask.
- **Always produce something runnable.** Each session should end with code that
  compiles/runs or a concrete artifact (design doc, system, build).
- **Scope is the enemy.** Repeatedly steer toward a *finishable* game. A shipped
  small game beats an abandoned big one — every time.
- **Use the engine's idioms.** Don't write Unity code that fights MonoBehaviour,
  or Godot code that ignores nodes/signals. The references show the idiomatic way.

---

## Reference Index

| File | Read it when… |
|---|---|
| `references/engine-selection.md` | Choosing Unity vs Unreal vs Godot vs web/custom |
| `references/core-systems.md` | Building game loop, input, state, save/load |
| `references/unity.md` | Implementing gameplay in Unity (C#) |
| `references/unreal.md` | Implementing gameplay in Unreal (C++/Blueprints) |
| `references/godot.md` | Implementing gameplay in Godot (GDScript/C#) |
| `references/shipping.md` | Building and publishing to any platform/store |
