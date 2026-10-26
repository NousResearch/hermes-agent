# Unreal MCP — Pitfalls & Lessons

Read before your first session; return whenever something misbehaves. Ordered
by when they bite: setup → calling discipline → editor state → content →
delivery.

## Setup & Connection

### 1. Start order: editor first, Hermes second

Hermes probes MCP servers at session start. If the editor (and its server)
isn't up yet, no `mcp_unreal_engine_*` tools exist in the session. Fix:
launch the editor, confirm the server bound (Output Log shows
`LogModelContextProtocol` with the address), then open a NEW Hermes session.
Tools don't hot-appear mid-session.

### 2. Server enabled but no tools advertised

The Unreal MCP plugin ships the SERVER, not the tools. If `list_toolsets`
returns nothing/near-nothing, the toolset provider plugin (AllToolsets) or
Toolset Registry isn't enabled in this project. Fix in Edit > Plugins,
restart the editor, restart the Hermes session.

### 3. macOS: full Xcode is required, not just Command Line Tools

On a Mac, the editor needs Xcode to compile shaders for Metal. Without it,
first launch dies with a modal "Xcode Not Found" dialog and the editor
exits as soon as it's dismissed (verified UE 5.8 behavior — the log shows
`RequestExit` right after the dialog). Fix: install full Xcode from the App
Store, open it once to accept the license / install components, and if it
lives anywhere other than `/Applications/Xcode.app`, point the toolchain at
it: `sudo xcode-select -s /path/to/Xcode.app`. Verify with
`xcode-select -p` (should print an Xcode path, not the bare CLT path).
Expect the first successful editor launch after that to spend a long time
compiling shaders.

### 4. Port 8000 conflicts

Common collisions: local dev servers, Jupyter, other MCP hosts. Symptom: the
server fails to bind (Output Log) or Hermes' probe times out. Fix: change
Server Port Number in Editor Preferences > Model Context Protocol AND the
`url` in `~/.hermes/config.yaml` (`mcp_servers.unreal-engine`), then restart
both sides. Verify: `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/mcp`
(non-000 means something is listening; whether it's Unreal is a different
question — check the Output Log).

### 4b. "Connection refused" mid-session

The editor was closed, crashed, or the server was stopped
(`ModelContextProtocol.StopServer`). Don't retry the tool in a loop — tell
the user, have them relaunch/restart the server, then reconnect (new session
if tools were lost).

### 5. GenerateClientConfig is not for Hermes

`ModelContextProtocol.GenerateClientConfig` writes config files for Claude
Code/Cursor/VSCode/Gemini/Codex into the project root. Hermes' connection
lives in `~/.hermes/config.yaml` via `hermes mcp install unreal-engine`.
Running GenerateClientConfig neither helps nor harms Hermes — just don't
mistake it for the Hermes setup step.

## Calling Discipline

### 6. One call at a time — never batch MCP calls

The server executes tool calls serially on the game thread and Epic
explicitly warns against overlapping calls. Hermes executes same-turn tool
calls concurrently — so batching two `mcp_unreal_engine_*` calls in one turn
IS issuing overlapping calls. Strictly sequential: call, await, then next.
This deliberately overrides the general "batch independent calls" guidance.

### 7. The editor freezes during every call — keep calls small

Game-thread execution means the editor UI hitches for the duration of each
tool call. A 30-second operation is 30 seconds of frozen editor. Split big
asks (e.g. "spawn 200 trees") into chunks so the user's editor stays
responsive and any failure loses only one chunk.

### 8. Modal dialogs deadlock the loop

Anything that pops a modal (some deletes, import options, save prompts,
experimental-plugin warnings) blocks the game thread — and your call —
until a human clicks. If a call hangs far beyond its normal duration, tell
the user to look at the editor for a dialog. Prefer tool paths/parameters
that avoid interactive prompts; save proactively so "unsaved changes"
prompts don't appear at bad times.

### 9. Timeouts: Hermes gives up before Unreal does

Hermes' default per-call timeout is 120 s. Asset imports, first-shader
compiles, big saves, and renders can exceed it — the call "fails" while the
editor happily finishes the work. Symptoms: timeout error, then the next
scene query shows the operation actually completed. Fixes: raise
`mcp_servers.unreal-engine.timeout` in config for heavy sessions; after any
timeout, RE-QUERY state before retrying, or you'll do the work twice
(duplicate actors are the classic case).

### 10. Stale schemas after editor-side changes

Toolsets are cached: after enabling a plugin, authoring a toolset, or Live
Coding, run `ModelContextProtocol.RefreshTools` in the editor console, then
re-run `list_toolsets`/`describe_toolset`. New C++ `UFUNCTION`s need a full
editor restart regardless. If a call fails with "unknown tool" that
`describe_toolset` just showed, refresh + reconnect.

### 10b. Ref-object vs plain-string params are inconsistent — the error schema is the tiebreaker

Most tools take object references as `{"refPath": ...}` objects, but some
take plain string paths — live-verified: `add_to_scene_from_class` takes
`actor_type` as a refPath OBJECT, while `add_to_scene_from_asset` takes
`asset_path` as a STRING. Don't pattern-match across tools. When a call
fails on params, the error text contains the complete input schema for
that exact function — read it and fix; it's faster and more authoritative
than re-describing the toolset.

### 10c. Blueprint DSL node IDs must come from find_node_types

DSL docs examples and intuition both produce wrong node IDs
(live-verified: `(event Tick)` fails — it's `EventTick`; `MakeRotator`
fails — it's `Math|Rotator|MakeRotator`; doc-style
`Utilities|Transformation|AddActorLocalRotation` fails — the registry says
`Transformation|AddActorLocalRotation`; there is no `(self)` node — omit
`:self` for the owning actor). Resolve EVERY node ID with
`find_node_types` against the target graph before writing DSL. Errors are
progressive (one failing node at a time, named exactly) — fix and rerun.

### 11. Experimental means drift

Tool names, parameters, and result shapes may change across engine versions.
The live schema from `describe_toolset` is the only contract. If this
skill's examples and the live schema disagree, the schema wins — and patch
this skill afterward.

## Editor & Scene State

### 12. Never assume a fresh level — and NEVER double-spawn environment actors

Query the scene before the first edit. Template levels (Open World, etc.)
ALREADY contain a DirectionalLight, SkyAtmosphere, SkyLight,
ExponentialHeightFog, and often VolumetricCloud. Spawning your own creates
duplicates that compound (double fog = whiteout, double sky = wrong
exposure) and are invisible in a screenshot until things look
inexplicably wrong. Live-verified failure: spawning a "golden hour kit"
into the default Open World template produced two of everything and a
254/255-luminance whiteout. Rule: `find_actors` for each environment class
FIRST; configure what exists; spawn only what's missing.

### 12b. Read the existing sun before imposing physical light values

Scene-craft tables give physical values (golden hour ≈ 10k lux), but a
template's world is calibrated as a SYSTEM — the default Open World sun is
`intensity: 10` (lux-ish units under default exposure), not 100,000. Setting
12,000 lux into that world blows the frame to pure white regardless of
small exposure tweaks. Live-verified rule: `get_properties` the existing
sun's intensity first. If the scene is calibrated low (single-digit sun),
work RELATIVE to it (e.g. golden hour ≈ 0.5–1× the template's noon value,
warm temperature, low pitch) and let auto-exposure adapt, or rebuild the
whole exposure chain deliberately (manual EV100 + physical values
everywhere). Mixing the two conventions is the #1 whiteout cause.
`ObjectTools.reset_properties` is the escape hatch — it restores per-project
defaults when you've painted into a corner.

### 12c. Verify exposure objectively, not just by eye

A capture can look "bright" in vision judgment while being unrecoverable.
Cheap objective check on any capture (editor-host filesystem):
`ffprobe -f lavfi -i "movie=<png>,signalstats" -show_entries
frame_tags=lavfi.signalstats.YAVG -of json` — YAVG > 250 means blown
white, < 5 means black. Use it whenever a lighting change should have
moved the histogram; it distinguishes "fog whiteout" from "exposure
whiteout" faster than iterating blind.

### 13. In-memory edits are lost on crash — save per milestone

Everything you do lives in unsaved packages until a save happens. The editor
is an application that can crash, especially mid-experimental-feature. Save
the level + dirty packages after every milestone (`AssetTools.save_assets`,
`SceneTools.save_actor`). Caveat: an UNTITLED level (`/Temp/Untitled_*`,
the state after File > New or launching without a map argument) may route a
save through the Save-As dialog — a modal that deadlocks the MCP loop
(pitfall 8). Prefer starting from a saved level (pass the map path on the
editor command line, or `SceneTools.load_level` a real `/Game/...` map)
before doing hours of work.

### 14. Label ≠ Name ≠ path — use the full path as the stable identifier

Outliner shows actor LABELS (settable, duplicable, human-friendly). Internal
NAMES are unique per level but auto-generated (`StaticMeshActor_3`). The only
identifier that survives renames and disambiguates duplicates is the full
object path: `/Game/Maps/Level.Level:PersistentLevel.BP_Character_C_0`.
Tools may accept label, name, or path — read the schema. When you create an
actor, immediately set a meaningful label, and record whatever handle the
tool returns for later operations.

### 14b. Asset path forms mean different things

| Form | Example | Loads |
|---|---|---|
| Package | `/Game/Foo/Bar` | The package (asset-registry queries) |
| Package.Asset | `/Game/Foo/Bar.Bar` | The primary asset (most load/assign args) |
| Package.Asset_C | `/Game/Foo/Bar.Bar_C` | A Blueprint's **generated class** |

"Class not found: /Game/Path/BP_Foo" almost always means the missing `_C`
suffix — spawning by Blueprint class needs the generated-class form.

### 14c. Property writes can silently no-op — round-trip verify

UPROPERTY names are PascalCase at the reflection layer; snake_case lookups
through some write paths silently change nothing and return no error. After
any property write that matters, READ THE VALUE BACK and compare (allowing
formatting normalization like `1` vs `1.000000`). If it didn't take, retry
with the exact PascalCase name shown by the property dump/schema.

### 15. Play In Editor changes the world (literally)

If the user hits Play, queries/edits may target the transient PIE world, and
edits to it evaporate when play stops. If results look inexplicably
transient or actor lists suddenly differ, ask whether PIE is running; do
edit work outside PIE.

### 16. Undo exists, but don't lean on it

Editor transactions power Ctrl+Z; tool-driven changes may or may not create
clean transaction boundaries depending on the tool's implementation. Treat
undo as the user's manual escape hatch, not your rollback mechanism — your
rollback is: query state, compute the inverse edit, apply it.

## Content & Assets

### 17. Long package names, not file paths

Assets are addressed as `/Game/Folder/Asset.Asset` (project content),
`/Engine/...` (engine content), `/Script/Module.Class` (native classes).
Windows-style or absolute filesystem paths are wrong everywhere except
import/export file arguments and screenshot output paths.

### 18. Filesystem results land on the EDITOR host

Screenshots, renders, and exports write to the machine running Unreal (e.g.
`<Project>/Saved/Screenshots/...`). If Hermes runs elsewhere (SSH backend,
container), `read_file` on that path reads the wrong filesystem. Same-machine
setups (the default here) can read captures directly.

### 19. Referenced ≠ loaded

Engine basics (`/Engine/BasicShapes/...`) are always available, but project
assets may need loading before use, and a typo'd asset path often fails
soft (empty mesh, default material) rather than loud. After assigning
meshes/materials, re-query the actor to confirm the reference stuck.

### 20. Material edits: instances, not parents

Editing a parent Material recompiles shaders (slow, global blast radius).
Create a Material Instance (Dynamic or Constant per the tool surface), set
scalar/vector/texture parameters on it, assign to the mesh. Parameter names
must match the parent's exposed parameters exactly — query/describe before
setting; a misnamed parameter usually no-ops silently.

### 20b. Shader/asset compilation is async — don't judge or proceed early

Material creation/edits kick off shader compilation that can run seconds to
minutes; Niagara compiles, DDC builds, and package saves are async too. A
screenshot taken mid-compile shows the old (or default-checkerboard) state.
After material work, wait for compilation before judging visuals (poll a
compile/errors predicate if the tool surface has one; otherwise screenshot
after a delay and re-check if it looks wrong). Same discipline after saves:
don't chain a disk-read straight after a write.

### 20c. Emissive needs intensity > 1 to bloom

Emissive at ≤1.0 looks self-lit but never blooms. 3–10 gives visible glow —
and the Post Process Volume must have Bloom enabled (default on).

### 20d. Crash patterns to avoid outright

Engine-level, any server: (a) deleting or transforming an ASSET while level
actors still reference it → `RegisteredElementType` assertion, editor down,
unsaved work gone — walk references first, swap actors to a replacement,
then delete; (b) spawn→delete→spawn the same actor in rapid succession can
corrupt the actor registry — don't tight-loop create/destroy cycles;
(c) Niagara/MetaSound assertion during PIE reverts to last on-disk save —
save BEFORE entering PIE when those subsystems are involved.

## Delivery

### 21. Screenshot judgment is part of the job

Don't declare a lighting/composition milestone done from numbers alone —
capture the viewport, `vision_analyze` it, and art-direct (silhouette,
exposure, horizon placement, scale against human height). The user is
non-technical; you are the one with eyes on both the brief and the frame.

### 21b. Editor sprite icons appear in captures — hide them at the source

Viewport captures include per-component editor sprites (light bulbs,
speaker icons, and the plain grey/blue billboard every empty Actor gets).
They are editor overlay, NOT scene content — and for video/hero work they
photobomb every frame. Do NOT try to remove them in post; a sprite
overlapping scene geometry defeats 2D cleanup (live-verified: three
inpainting strategies all either smeared letter edges or left outline
pixels).

Fix at the source, before capturing: sprites are real components on the
actor. `ActorTools.get_components` → find `Billboard`/`Sprite`/`Arrow`
components → `ObjectTools.set_properties` with `{"bVisible": false}` on
each (round-trip verify). `remove_component` fails on construction-time
default subobjects ("Could not find subobject handle") — hide, don't
remove. Sweep the whole scene in one ProgrammaticToolset script: iterate
`find_actors`, hide every matching component (verified: 148 actors
scanned, 13 sprites hidden, one round-trip). An editor capture is still
not proof a Niagara effect is emitting — verify effects via actor state or
a PIE capture.

### 21c. The viewport axis gizmo survives bShowUI=false — plan a post-crop

`CaptureViewport` with `bShowUI: false` hides menus and toolbars but the
bottom-left XYZ axis gizmo is still burned into every frame
(live-verified on 5.8; at 2027x1534 it occupies roughly the region below
y≈1350, x<300). For video/hero deliverables, compose with spare margin and
crop it out in post (e.g. a 16:9 punch-in via ffmpeg
`crop=2027:1140:0:180,scale=1920:1080`) — deterministic across every frame
of a sequence, no per-frame edits. Unlike the gizmo, actor sprite icons
CAN be removed at the source — hide their components before the shoot
(see 21b); do that instead of post-cleanup.

### 21d. Frame sequences: one client session, serial captures, resumable loop

For multi-frame virtual-camera moves (orbits, cranes) drive
`CaptureViewport` in a loop from ONE MCP session, strictly serial (the
game thread renders each), and write frames idempotently
(skip-if-exists) so a killed run resumes where it stopped. Budget
realistically: ~15-20 s/frame at editor resolution on a laptop — a 240-frame
10 s @ 24 fps shot is roughly an hour. Smoothstep-ease camera parameters;
constant angular velocity reads mechanical. Remove the VolumetricCloud
actor if its low-res gather smears blocky artifacts across sky frames.

### 22. Report package paths + file paths

The user needs: what actors/assets now exist (labels + `/Game/...` paths),
where the level was saved, and absolute filesystem paths of any
captures/renders (delivered as `MEDIA:` where appropriate).

## Sources

Grounded in Epic's UE 5.8 Unreal MCP documentation, Epic's own agent-facing
skill pack for this server (unreal-engine-skills-for-claude-code), and
engine-level field reports from the UE-via-MCP community (ue5-mcp field
manual). Engine behaviors (crash patterns, Lumen mobility, async compiles,
reflection casing) are server-agnostic; tool names and schemas remain
whatever the live `describe_toolset` says.
