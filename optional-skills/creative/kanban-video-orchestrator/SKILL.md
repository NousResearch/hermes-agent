---
name: kanban-video-orchestrator
description: Plan, set up, and monitor a multi-agent video production pipeline backed by Hermes Kanban. Use when the user wants to make ANY video — narrative film, product/marketing, music video, explainer, ASCII/terminal art, abstract/generative loop, comic, 3D, real-time/installation — and the work warrants decomposition into specialized profiles (writer, designer, animator, renderer, voice, editor, etc.) coordinated through a kanban board. Performs adaptive discovery to scope the brief, designs an appropriate team for the requested style, generates the setup script that creates Hermes profiles + initial kanban task, then helps monitor execution and intervene when tasks stall or fail. Routes scenes to whichever Hermes rendering / audio / design skill fits each beat (`ascii-video`, `manim-video`, `p5js`, `comfyui`, `touchdesigner-mcp`, `blender-mcp`, `pixel-art`, `baoyu-comic`, `claude-design`, `excalidraw`, `songsee`, `heartmula`, …) plus external APIs for TTS, image-gen, and image-to-video as needed.
version: 1.0.0
author: [SHL0MS, alt-glitch]
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [video, kanban, multi-agent, orchestration, production-pipeline]
    related_skills: [kanban-orchestrator, kanban-worker, ascii-video, manim-video, p5js, comfyui, touchdesigner-mcp, blender-mcp, pixel-art, ascii-art, songwriting-and-ai-music, heartmula, songsee, spotify, youtube-content, claude-design, excalidraw, architecture-diagram, concept-diagrams, baoyu-comic, baoyu-infographic, humanizer, gif-search, meme-generation]
    credits: |
      The single-project workspace layout, profile-config patching pattern,
      SOUL.md-per-profile model, TEAM.md task-graph convention, and
      `--workspace dir:<path>` discipline are adapted from alt-glitch's
      original multi-agent video pipeline at
      https://github.com/NousResearch/kanban-video-pipeline.
---

# Kanban Video Orchestrator

Wrap any video request — from a 15-second product teaser to a 5-minute narrative
short to a music video to an ASCII loop — in a Hermes Kanban pipeline that
decomposes the work to specialized agent profiles.

This skill does **not** render anything itself. It is a meta-pipeline that:

1. **Scopes** the request through targeted discovery
2. **Designs** an appropriate team (which roles, which tools per role) based on the style
3. **Generates** a setup script that creates Hermes profiles, project workspace, and the initial kanban task
4. **Hands off** to the director profile, which decomposes via the kanban
5. **Monitors** execution, helps intervene when tasks stall or fail

The actual rendering happens inside the kanban once it's running, via whichever
existing skills + tools fit the scenes — `ascii-video`, `manim-video`, `p5js`,
`comfyui`, `touchdesigner-mcp`, `blender-mcp`, `songwriting-and-ai-music`,
`heartmula`, external APIs, or plain Python with PIL + ffmpeg.

## When NOT to use this skill

- The video is one continuous procedural project that needs no specialists. Just write the code directly.
- The user wants a quick one-shot conversion (e.g. "convert this mp4 to a GIF") — use ffmpeg directly.
- The output is a static image, GIF, or audio-only artifact — use the matching specific skill (`ascii-art`, `gifs`, `meme-generation`, `songwriting-and-ai-music`).
- The work fits a single existing skill cleanly (e.g. a pure ASCII video — just use `ascii-video`).

## Workflow

```
DISCOVER  →  BRIEF  →  TEAM DESIGN  →  SETUP  →  EXECUTE  →  MONITOR
```

### Step 1 — Discover (ask the right questions)

The discovery process is **adaptive**: ask only what is actually needed. Always
start with three questions to identify the broad shape:

- **What is the video?** (one-sentence brief)
- **How long?** (5-30s teaser / 30-90s short / 90s-3min explainer / 3-10min film / longer)
- **What aspect ratio + target platform?** (1:1 / 9:16 / 16:9; X, IG, YouTube, internal, etc.)

From the answer, classify the style category. The style determines which
follow-up questions to ask. **Do not ask all questions at once.** Ask 2-4 at a
time, listen, then proceed. Make reasonable assumptions whenever the user
implies an answer.

For complete intake patterns and per-style question banks, see
**[references/intake.md](references/intake.md)**.

### Step 2 — Brief

Once enough is known, produce a structured `brief.md` using the template in
`assets/brief.md.tmpl`. Stages:

1. **Concept** — the one-sentence pitch + emotional north star
2. **Scope** — duration, aspect, platform, deadline
3. **Style** — visual references, brand constraints, tone
4. **Scenes** — beat-by-beat breakdown (durations, content, target tool)
5. **Audio** — narration / music / SFX / silent (per scene if needed)
6. **Deliverables** — file format, resolution, optional alternates (vertical cut, GIF, etc.)

Show the brief to the user for confirmation before designing the team. **The
brief is the contract** — every downstream task references it.

### Step 3 — Team design

Pick role archetypes from the library that fit this video. **Compose, don't
clone.** Most videos need 4-7 profiles. The director is always present; the
rest are picked by what the brief actually requires.

For the role library and per-style team compositions, see
**[references/role-archetypes.md](references/role-archetypes.md)**.

For mapping role → which Hermes skills + toolsets it loads, see
**[references/tool-matrix.md](references/tool-matrix.md)**.

### Step 4 — Setup

Generate a setup script (`setup.sh`) and run it. The script:

1. Creates the project workspace (`~/projects/video-pipeline/<slug>/`)
2. Copies any provided assets into `taste/`, `audio/`, `assets/`
3. Creates each Hermes profile via `hermes profile create --clone`
4. Writes per-profile `SOUL.md` (personality + role definition)
5. Configures profile YAML (toolsets, always_load skills, cwd)
6. Writes `brief.md`, `TEAM.md`, and `taste/` content
7. Fires the initial `hermes kanban create` task assigned to the director

Use `scripts/bootstrap_pipeline.py` to generate setup.sh from a brief +
team-design JSON. See **[references/kanban-setup.md](references/kanban-setup.md)**
for the setup script structure, profile config patterns, and the critical
"shared workspace" rule.

### Step 5 — Execute

Run `setup.sh`. Then provide the user with monitoring commands:

```bash
hermes kanban watch --tenant <project-tenant>     # live events
hermes kanban list  --tenant <project-tenant>     # board snapshot
hermes dashboard                                   # visual board UI
```

The director profile takes over from here, decomposing the work and routing
tasks to specialist profiles via the kanban toolset.

### Step 6 — Monitor and intervene

Stay engaged — the kanban runs autonomously but a stuck task or bad output
needs human (or AI) judgment.

Monitoring patterns: poll `kanban list` periodically, inspect any RUNNING task
that exceeds its expected duration with `kanban show <id>`, and check
heartbeats. When a worker's output fails review, the standard interventions are:

1. Comment on the worker's task with specific feedback (`kanban_comment`)
2. Create a re-run task with the original as parent
3. Adjust the brief's scope and let the director re-decompose

For diagnostic patterns, intervention recipes, and the "task is stuck"
playbook, see **[references/monitoring.md](references/monitoring.md)**.

## Reference: worked examples

Six concrete pipelines covering very different video styles — narrative film,
product/marketing, music video, math/algorithm explainer, ASCII video, real-time
installation — showing how the same workflow yields very different teams and
task graphs. See **[references/examples.md](references/examples.md)**.

## Critical rules

1. **Discovery before action.** Never start generating a brief or team without
   asking at least the three baseline questions. A bad brief cascades through
   the entire pipeline.

2. **Match the team to the video.** Don't reuse the same 4-profile setup for
   every job. A music video that doesn't have a beat-analysis profile will
   misfire. A narrative film that doesn't have a writer profile will produce
   incoherent scenes. See `references/role-archetypes.md`.

3. **One workspace per project.** All profiles for a given video share the same
   `dir:` workspace. Tasks pass artifacts via shared filesystem and structured
   handoffs. **Every** `kanban_create` call passes
   `workspace_kind="dir"` + `workspace_path="<absolute project path>"`.

4. **Tenant every project.** Use a project-specific tenant
   (`--tenant <project-slug>`). Keeps the dashboard scoped and prevents
   cross-pollination with other ongoing kanbans.

5. **Respect existing skills.** When a scene fits an existing skill, the
   relevant renderer should load that skill via `--skill <name>` on its task
   or `always_load` in its profile. Do not re-derive what a skill already
   provides.

6. **The director never executes.** Even with the full `kanban + terminal +
   file` toolset, the director's `SOUL.md` rules forbid it from executing
   work itself. It decomposes and routes only — every concrete task becomes
   a `hermes kanban create` call to a specialist profile. The
   `kanban-orchestrator` skill spells this out further.

7. **Don't over-decompose.** A 30-second product video does NOT need 20 tasks.
   Aim for the smallest task graph that still parallelizes well and exposes the
   right human-review gates.

8. **Verify API keys BEFORE firing.** External APIs (TTS, image-gen,
   image-to-video) need keys in `~/.hermes/.env` or the user's secret store.
   A worker that hits a missing-key error wastes a task slot. The setup
   script's `check_key` helper aborts cleanly if a required key is missing.

## File map

```
SKILL.md                            ← this file (workflow + rules)
references/
  intake.md                         ← discovery question banks per style
  role-archetypes.md                ← role library (writer, designer, animator, …)
  tool-matrix.md                    ← skill + toolset mapping per role
  kanban-setup.md                   ← setup script structure & profile config
  monitoring.md                     ← watch + intervene patterns
  examples.md                       ← six worked pipelines
assets/
  brief.md.tmpl                     ← brief skeleton
  setup.sh.tmpl                     ← setup script skeleton
  soul.md.tmpl                      ← profile personality skeleton
scripts/
  bootstrap_pipeline.py             ← generate setup.sh from brief + team JSON
  monitor.py                        ← polling + intervention helpers
```
