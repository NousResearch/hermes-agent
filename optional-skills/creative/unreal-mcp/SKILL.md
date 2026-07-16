---
name: unreal-mcp
description: "Use when the user wants to do anything in Unreal Engine through Epic's official editor-embedded MCP server (catalog entry: unreal-engine) — build/light/populate scenes, place and transform actors, author Blueprints, animate with Sequencer, create material instances, frame cameras, take screenshots, render, import assets, run PIE test sessions and automation tests, or automate the editor end-to-end from plain-English prompts with no Unreal knowledge required. Covers the tool-search discovery walk (list_toolsets/describe_toolset/call_tool), serial game-thread call discipline, ProgrammaticToolset batching, the Blueprint graph DSL loop, scene-craft numbers (physical light units, exposure, scale conventions), complete build recipes, save/undo hygiene, and extending the tool surface with custom Python toolsets."
version: 1.0.0
requires: Unreal Editor 5.8+ with the Unreal MCP plugin enabled and its server running
author: Hermes Agent
license: MIT
tags: [unreal, unreal-engine, ue5, 3d, mcp, scenes, cinematics, lighting, gamedev]
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [unreal, unreal-engine, ue5, 3d, mcp, scenes, cinematics, lighting, gamedev]
    related_skills: [blender-mcp]
---

# Unreal Engine MCP Skill

Companion skill for the `unreal-engine` entry in the Hermes MCP catalog. The
MCP server (Epic's official, experimental "Unreal MCP" plugin, internal id
`ModelContextProtocol`) runs INSIDE the Unreal Editor process and exposes
editor functionality as typed tools. This skill teaches how to drive it well:
discovering the live tool surface, sequencing calls safely, translating
plain-English asks into scenes that actually look good, and verifying work
visually. The user should never need to touch the editor beyond launching it.

## When to Use

Use when the user wants anything done in Unreal Engine: build or dress a
level, spawn/move/delete actors, set up lighting and atmosphere, create or
tune material instances, frame a camera shot, capture screenshots or renders,
import assets, inspect the scene or UI, run automation tests, or script the
editor. Works for single actions ("make the sun golden hour") and for
complete multi-step projects ("build me a moody forest clearing with a
campfire and render a shot of it").

Don't use for: DCC-style mesh modeling/sculpting (use `blender-mcp` and
import the result), or for editing Unreal C++ project source (that's normal
code work — use the terminal; this skill is about the live editor).

## Prerequisites

Two halves, in this order: the editor side must be up before Hermes connects.

### One-time, editor side

1. Unreal Editor **5.8+** with a project open. (macOS: full Xcode must be
   installed and its license accepted — the editor exits on first launch
   without it; see pitfalls.)
2. **Edit > Plugins** — enable **Unreal MCP** (its Toolset Registry
   dependency auto-enables). Restart the editor when prompted.
3. The typed toolsets ship separately from the server: also enable the
   **AllToolsets** plugin in the same Plugins browser. Unreal MCP ships NO
   tools itself — AllToolsets provides the shipped toolsets (SceneTools,
   ActorTools, MaterialInstanceTools, ObjectTools, …); skip it and the
   server connects but the agent has nothing to call.
4. **Edit > Editor Preferences > General > Model Context Protocol** — enable
   **Auto Start Server**. Default bind is `http://127.0.0.1:8000/mcp`
   (port/path configurable in the same panel; server name is `unreal-mcp`).
   To start manually instead, run `ModelContextProtocol.StartServer` in the
   editor console (backtick key).

### One-time, Hermes side

    hermes mcp install unreal-engine

This writes the `mcp_servers.unreal-engine` HTTP entry pointing at
`http://127.0.0.1:8000/mcp` and probes the live server for its tools. Run it
while the editor + server are up so the probe sees the real surface. If the
user changed port/path in Editor Preferences, edit the `url` in
`~/.hermes/config.yaml` under `mcp_servers.unreal-engine` to match.

Do NOT use `ModelContextProtocol.GenerateClientConfig` for Hermes — that
writes `.mcp.json`-style files for Claude Code/Cursor/etc. Hermes connects
from `config.yaml` via the catalog entry.

### Every session

1. Launch Unreal Editor, wait for the project to finish loading; confirm the
   server started (Output Log shows the bind address, or run
   `ModelContextProtocol.StartServer` manually).
2. Start the Hermes session. Tools register as `mcp_unreal_engine_*`. If
   they're missing: editor wasn't up first — start it, then open a new
   Hermes session.
3. Sanity check: call `mcp_unreal_engine_list_toolsets` and confirm toolsets
   come back.

## The Tool Surface: Discovery, Not a Fixed List

By default the plugin runs in **tool-search mode**: `tools/list` returns only
three meta-tools, and every real tool is reached through them. Through Hermes
they appear as:

| Hermes tool | Purpose |
|---|---|
| `mcp_unreal_engine_list_toolsets` | Names + descriptions of every registered toolset |
| `mcp_unreal_engine_describe_toolset` | Full JSON schemas for one named toolset's tools |
| `mcp_unreal_engine_call_tool` | Invoke a named tool with arguments, get the result |

The discovery walk, always in this order:

1. `list_toolsets` → see what capability groups this project actually has
   (the surface is project-dependent: enabled plugins, Game Feature Plugins,
   and any custom toolsets all contribute). Names come back FULLY QUALIFIED
   (`editor_toolset.toolsets.scene.SceneTools`,
   `EditorToolset.EditorAppToolset`) — use them verbatim as `toolset_name`.
2. `describe_toolset` on the group you need → read the real parameter
   schemas. Never guess parameter names — schemas are the contract.
3. `call_tool` with the qualified toolset name, the SHORT tool name
   (`find_actors`, not the dotted form), and arguments matching the schema.

Cache what you learn for the session; re-list only after the editor side
changes (new plugin enabled, toolset authored, `RefreshTools` run).

The alternative eager mode (`Enable Tool Search` off in Editor Preferences)
advertises every tool as its own `mcp_unreal_engine_<tool>` entry. Discovery
then happens at `hermes mcp install`/`configure` time instead. Tool-search
mode is the default and what this skill assumes; it also keeps schema tokens
out of every API call, so prefer it.

See `references/tool-surface.md` for the shipped toolset catalog, authoring
custom toolsets, and the full plugin configuration/console-command reference.

## Operating Loop

Every Unreal task follows the same loop:

1. **Inspect first.** List toolsets, then query the scene/level state before
   touching anything. Never assume an empty or default level. In an
   unfamiliar project, also check for project-registered Agent Skills
   (`call_tool` → `AgentSkillToolset.ListSkills`): a matching project skill's
   instructions override this skill's generic defaults.
2. **Act in small, single-purpose calls.** One logical step per `call_tool`.
   The server executes tools **serially on the game thread** — a big
   monolithic operation freezes the editor UI until it finishes and risks
   client timeouts. Exception: for loops over 5+ homogeneous operations,
   ONE `ProgrammaticToolset.execute_tool_script` call batches them
   server-side without breaking the serial rule
   (`references/advanced-workflows.md`).
3. **NEVER issue overlapping calls.** Do not batch multiple
   `mcp_unreal_engine_*` calls in one turn — Hermes runs batched calls
   concurrently, and parallel calls against the game thread deadlock or
   fail. Strictly one call, await result, next call. This overrides the
   general parallel-tool-calls guidance.
4. **Read every result.** Many tools (Blueprint compiles, material edits,
   widget creation) report success/failure in the response body with no
   protocol-level exception. Anything that isn't an explicit success is a
   stop-and-diagnose, not a shrug. After property writes, read the value
   back — several write paths silently no-op (see pitfalls).
5. **Verify visually and structurally.** After each milestone, confirm state
   by querying the actors/properties you changed, and capture a viewport
   screenshot when composition matters (see `references/tool-surface.md` for
   the capture options; `vision_analyze` the image — you are the art
   director, judge it).
6. **Save often.** Editor edits are in-memory until packages/levels are
   saved; an editor crash loses everything since the last save, and MCP
   edits are not reliably undoable. Save before AND after any bulk change,
   and after every milestone.
7. **Report concretely.** Actor labels, asset paths (`/Game/...`), file
   locations of captures/renders.

Rules of the world while you work:

- Units are **centimeters**; axes are **Z-up**, X-forward; rotations are
  degrees (Rotator: Roll around X, Pitch around Y, Yaw around Z). Human eye
  height ≈ 165 cm; a door ≈ 210×90 cm. Full tables in
  `references/scene-craft.md`.
- Content paths use long package names: `/Game/Folder/Asset.Asset` for
  project content, `/Engine/BasicShapes/Cube.Cube` for engine primitives.
- Actor **labels** (what you see in the Outliner, settable, non-unique) are
  not actor **names** (internal, unique). Prefer resolving actors by
  label/class queries, then hold on to whatever handle the tool returns.
- Prefer physically-plausible lighting values (lux/candela/Kelvin) over
  arbitrary brightness numbers — but FIRST read the existing sun's
  intensity to learn the scene's calibration convention; template worlds
  are often calibrated around `intensity: 10`, and physical values blow
  them out (`references/scene-craft.md` has the numbers,
  `references/pitfalls.md` #12b has the calibration rule).

## From Plain English to a Scene

The user gives intent, not specs. Translate before you build:

1. **Extract the brief.** Subject, mood, time of day, interior/exterior,
   style, deliverable (screenshot? render? playable level?). Ask at most one
   round of clarifying questions, then commit — you are the technical
   director; don't bounce Unreal jargon back at the user.
2. **Plan the build order.** The order that works: level/environment shell →
   blocking (major geometry/meshes in place) → lighting + atmosphere →
   materials → set dressing/detail → camera → capture/render. Post the plan
   as a todo list for multi-step builds.
3. **Build with the loop above**, one milestone at a time, screenshot at
   each milestone.
4. **Art-direct yourself.** Compare each screenshot against the brief:
   readable silhouette? believable light direction/intensity? horizon not
   dead-center? scale correct against a human-height reference? Fix before
   moving on.
5. **Deliver.** Screenshots/renders as files (`MEDIA:` path), plus a short
   summary of what exists in the level and where it was saved.

`references/recipes.md` has complete worked builds (exterior daylight scene,
moody interior, golden-hour cinematic + render, asset import & placement)
with the exact call sequences and values.

## Reference Files

Load on demand; keep SKILL.md-level rules in mind throughout.

| Reference | Contents |
|---|---|
| `references/tool-surface.md` | Shipped toolsets catalog, discovery protocol detail, plugin console commands/CVars/flags, screenshot & capture paths, MCP Inspector debugging, extending with custom Python/C++ toolsets |
| `references/advanced-workflows.md` | Sophisticated workflows, live-verified: ProgrammaticToolset batching, Blueprint DSL authoring loop (create→DSL→compile→spawn), PIE test sessions, Sequencer orientation (140 tools), LogsToolset self-debugging, automation testing, semantic asset search, config settings, per-situation decision table |
| `references/scene-craft.md` | Numeric cheat sheet: physical light intensities, color temperatures, exposure/EV100, fog densities, mood recipes (noon/golden hour/overcast/night/interior), scale tables, content path conventions |
| `references/recipes.md` | End-to-end worked builds with exact call sequences |
| `references/pitfalls.md` | Setup, runtime, and workflow pitfalls with fixes — read before your first session and whenever something misbehaves |

## Pitfalls (top of mind — full list in references/pitfalls.md)

- **Start order matters.** Editor + server up first, then the Hermes
  session. Missing `mcp_unreal_engine_*` tools = wrong order.
- **One call at a time.** Serial game thread; no batching, no overlap.
- **The editor UI freezes during each call.** That's by design (game-thread
  execution). Warn the user during long operations; keep calls small.
- **Modal dialogs block everything.** A tool call that opens (or collides
  with) a modal editor dialog stalls until a human dismisses it. If a call
  hangs indefinitely, tell the user to check the editor for a dialog.
- **Timeouts on long operations.** Hermes' per-call default is 120 s; asset
  imports, big level saves, and renders can exceed it. Raise
  `mcp_servers.unreal-engine.timeout` in `~/.hermes/config.yaml` for
  render/import-heavy sessions.
- **Stale tool schemas.** After authoring/hot-reloading toolsets or enabling
  a plugin, run `ModelContextProtocol.RefreshTools` in the editor console
  and re-`list_toolsets`. New C++ `UFUNCTION`s need a full editor restart —
  Live Coding won't surface them.
- **Experimental plugin.** APIs and tool shapes can change between engine
  versions; trust `describe_toolset` over memory, including this skill's
  examples. When docs and the live schema disagree, the live schema wins.
- **Don't expose the server beyond localhost.** Loopback-only, no auth, by
  design. Never suggest binding it wider.
- **Licensing note.** The server logs on start: data transmitted via the
  plugin to a connected LLM service is Licensed Technology under the UE
  EULA (§6(e)) — the user is responsible for ensuring their LLM provider
  doesn't train on it. Surface this if the user asks about data handling.

## Verification Checklist

- [ ] `list_toolsets` returns toolsets at session start (connection healthy)
- [ ] Scene state queried before first edit (never assumed empty)
- [ ] After each milestone: changed actors/properties re-queried and a
      screenshot reviewed against the brief
- [ ] Level/dirty packages saved after each milestone and at the end
- [ ] Deliverables exist on disk (screenshot/render paths confirmed) and are
      reported to the user with absolute paths
- [ ] Editor left in a clean state: no pending modal, no unsaved surprise,
      user told exactly what was created/changed and where
