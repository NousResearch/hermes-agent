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

### 3. Port 8000 conflicts

Common collisions: local dev servers, Jupyter, other MCP hosts. Symptom: the
server fails to bind (Output Log) or Hermes' probe times out. Fix: change
Server Port Number in Editor Preferences > Model Context Protocol AND the
`url` in `~/.hermes/config.yaml` (`mcp_servers.unreal-engine`), then restart
both sides. Verify: `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/mcp`
(non-000 means something is listening; whether it's Unreal is a different
question — check the Output Log).

### 4. "Connection refused" mid-session

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

### 11. Experimental means drift

Tool names, parameters, and result shapes may change across engine versions.
The live schema from `describe_toolset` is the only contract. If this
skill's examples and the live schema disagree, the schema wins — and patch
this skill afterward.

## Editor & Scene State

### 12. Never assume a fresh level

Query the scene before the first edit. The user's level may have existing
actors, a non-default sun, post-process volumes with exposure overrides —
your lighting changes can look wrong because of a pre-existing volume, not
your values.

### 13. In-memory edits are lost on crash — save per milestone

Everything you do lives in unsaved packages until a save happens. The editor
is an application that can crash, especially mid-experimental-feature. Save
the level + dirty packages after every milestone. Also: some operations
(level streaming, some asset moves) behave differently on unsaved assets.

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

### 21b. Editor screenshots show sprite icons that look like content

Viewport captures in the editor include per-component sprite icons (light
bulbs, speaker/Niagara icons). They are editor overlay, NOT your scene —
particles especially: an editor screenshot is not proof a Niagara effect is
emitting. Verify effects via the actor's active state or a PIE capture.

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
