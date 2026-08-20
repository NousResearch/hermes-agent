---
name: game-development
description: Build and ship games in Unity, Unreal, or Godot.
version: 0.1.0
author: Burak Koç (@HeLLGURD), Hermes Agent
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

# Game Development Skill

Guides a game from a blank page to a shipped build across Unity, Unreal, Godot,
and web or custom engines. It orchestrates the whole lifecycle — concept, engine
choice, architecture, gameplay, art and audio, polish, optimization, and release
— and defers engine-specific detail to the files in `references/`. It does not
play or run existing games, and it is not a general software-architecture guide.

## When to Use

- The user wants to make a game and doesn't know where to start.
- The user is mid-project and needs help with a specific system or stage.
- The user is choosing between Unity, Unreal, Godot, or building from scratch.
- The user wants to take a prototype to a shippable, published game.

Do NOT use for:

- Playing or running existing games — see `gaming/pokemon-player`,
  `gaming/minecraft-modpack-server`.
- Pure 2D asset creation in isolation — see `creative/pixel-art`.
- Generic software architecture unrelated to games — see `code-wiki`.

## Prerequisites

- `git` for version control from day one: add the engine's `.gitignore` and use
  Git LFS for binary assets (textures, audio, models).
- One engine installed once chosen — Unity Hub, Unreal/Epic Launcher, or Godot
  (Godot needs no account and is the lightest to start).
- The `terminal` toolset to scaffold projects, run the engine, and make builds.
- The `web` toolset to fetch engine docs and asset-store info when needed.
- Platform SDKs for each target at build time (see `references/shipping.md`).

## How to Run

1. Identify the user's current stage from the Quick Reference — most sessions
   target one stage, not the whole list.
2. Work that stage, opening the matching reference for the engine-specific how.
3. Use `terminal` to scaffold, run, and build; `read_file` and `search_files`
   to navigate the project; `patch` to edit scripts; `web` to pull engine docs.
4. End every session with something runnable — code that compiles or a concrete
   artifact (design doc, system, or build).

Write the one-page design doc to a profile-safe path such as
`~/.hermes/gamedev/<project>/design.md` (or a project-relative `docs/design.md`)
— never a hardcoded absolute path.

## Quick Reference

Lifecycle and the reference to open at each stage:

```
1.  Concept & design ....... define, scope, and write down the game
2.  Engine selection ....... references/engine-selection.md
3.  Project architecture ... structure, version control, build pipeline
4.  Core systems ........... references/core-systems.md
5.  Gameplay programming ... references/{unity,unreal,godot}.md
6.  Art & audio pipeline ... import, animate, integrate assets
7.  Polish & game feel ..... juice, VFX, UI/UX
8.  Optimization ........... profile, frame budget, memory
9.  Testing & QA ........... playtest, bug tracking, test builds
10. Build & publish ........ references/shipping.md
11. Post-launch ............ patches, analytics, community
```

Engine at a glance:

| If… | Use |
|---|---|
| 2D, solo/indie, fast iteration, free & open | **Godot** |
| 3D, large ecosystem + asset store + mobile | **Unity** |
| High-end 3D visuals, big team | **Unreal** |
| Tiny web game, full control, learning fundamentals | **Web (HTML5/JS/TS)** |
| Pure logic prototype, no graphics yet | **Python + Pygame** |

Reference index:

| File | Read it when… |
|---|---|
| `references/engine-selection.md` | Choosing Unity vs Unreal vs Godot vs web/custom |
| `references/core-systems.md` | Building game loop, input, state, save/load |
| `references/unity.md` | Implementing gameplay in Unity (C#) |
| `references/unreal.md` | Implementing gameplay in Unreal (C++/Blueprints) |
| `references/godot.md` | Implementing gameplay in Godot (GDScript/C#) |
| `references/shipping.md` | Building and publishing to any platform/store |

## Procedure

**1 — Concept & design.** Lock down what the game *is* before any code. Nail the
core fantasy (what the player feels), the core loop (the 30-second action, in one
sentence), 2–3 reference games plus the one thing it does differently, an honest
scope check (solo or first game means *small*), and the win/lose/progression arc.
Produce a one-page design doc; cut features to a 5–8 item MVP.

**2 — Engine selection.** Match the engine to the game and the developer, not to
hype. See `references/engine-selection.md` for the full matrix. Once chosen,
scaffold the project and initialize `git` immediately with the engine's
`.gitignore`.

**3 — Project architecture.** Set up for maintainability before the codebase
grows: version control with Git LFS for binaries, a flat and predictable folder
layout (`scripts/`, `assets/`, `scenes/`, `prefabs/`), one naming convention, and
a known path to a runnable build.

**4 — Core systems.** Build the engine-agnostic foundations — game loop (fixed vs
variable timestep), input handling, the scene/state machine, an event/signal bus,
save/load, and data-driven stats. See `references/core-systems.md`. Build once,
reuse across projects.

**5 — Gameplay programming.** The engine-specific layer: Unity (C#) →
`references/unity.md`; Unreal (C++/Blueprints) → `references/unreal.md`; Godot
(GDScript/C#) → `references/godot.md`. Each reference ships ready-to-adapt
patterns for movement, camera, collision/damage, spawning, and enemy AI.

**6 — Art & audio pipeline.** Getting assets in correctly matters as much as
making them: sprite import settings and atlases for 2D; glTF/FBX, scale, and PBR
for 3D; animation state machines; and an audio manager with volume buses. Keep an
asset budget (texture sizes, poly counts) from the start.

**7 — Polish & game feel.** The gap between a prototype and a game: juice (screen
shake, hit-stop, particles, easing), feedback on every action, input buffering
and coyote time, and accessible UI/UX. Add feedback until it feels slightly too
much, then pull back.

**8 — Optimization.** Only after it works. Profile first with the engine's
profiler to find the real bottleneck, hold a frame budget (16.6 ms for 60 FPS),
and reach for object pooling, batching, atlasing, LODs, culling, and cached
lookups. Watch texture/audio memory and build size.

**9 — Testing & QA.** Playtest early with real people and watch silently. Track
bugs with repro steps and severity. Test *builds*, not just play-in-editor — bugs
hide in builds. Cover edge cases (alt-tab, resize, controller unplug, save
mid-action).

**10 — Build & publish.** Turn the project into a distributable game. See
`references/shipping.md` for per-platform builds (desktop, web, mobile) and store
setup (Steam, itch.io, mobile stores), signing, ratings, and the launch checklist.

**11 — Post-launch.** Hotfix critical bugs fast, track analytics for drop-off and
crashes, engage the community, and plan content updates from real player data
rather than assumptions.

## Pitfalls

- **Scope is the #1 killer.** Steer relentlessly toward a *finishable* game; a
  shipped small game beats an abandoned big one. Avoid MMOs and open worlds for a
  first project.
- **No version control from day one.** Add `git` and Git LFS immediately; binary
  assets bloat history fast and are painful to retrofit.
- **Premature optimization.** Don't optimize before it works or before you
  profile — you will spend the early game guessing at the wrong bottleneck.
- **Not playtesting with real people.** You are blind to your own game; if a
  tester is confused, the game is, not them.
- **Skipping test builds.** Bugs hide in builds that never appear in the editor.
- **Fighting the engine's idioms.** Use nodes/signals in Godot, MonoBehaviour in
  Unity, the Gameplay Framework in Unreal — don't reinvent them.

## Verification

- The game **runs from a build**, not just play-in-editor, on a target platform.
- The **core loop is playable start to finish** (start → succeed/fail → repeat).
- It has been **playtested by someone else** without you guiding them.
- The project is under **version control** with the engine's `.gitignore` and Git
  LFS for binaries.
- The profiler shows the frame is **within budget** for the target frame rate
  before shipping.
