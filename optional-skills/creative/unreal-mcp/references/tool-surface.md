# Unreal MCP — Tool Surface Reference

How Epic's editor-embedded MCP server organizes, advertises, and executes
tools, and how to extend the surface when the shipped tools run out.
Everything here is against UE 5.8's experimental plugin (id
`ModelContextProtocol`); expect drift between engine versions — the live
`describe_toolset` schema always outranks this file.

## Architecture in one paragraph

The **Unreal MCP** plugin hosts the HTTP server inside the editor process
(default `http://127.0.0.1:8000/mcp`, loopback-only, no auth, HTTP + SSE
only — no stdio/WebSocket). It implements the protocol but ships no tools of
its own. Tools come from **Toolsets** — classes deriving from
`UToolsetDefinition` (C++) or `unreal.ToolsetDefinition` (Python) — collected
at startup by the **Toolset Registry** subsystem (sibling plugin,
auto-enabled; the registry itself ships no toolsets either). The shipped
tools live in per-domain plugins under `Engine/Plugins/Experimental/
Toolsets/` — the workhorse is **EditorToolset** (core editor toolsets,
Python + C++) — and **AllToolsets** is a one-checkbox aggregator plugin
that depends on ~21 of them (verified from `AllToolsets.uplugin`, 5.8):
AIModule, AnimationAssistant, AutomationTest, ConfigSettings, Conversation,
DataRegistry, DataflowAgent, Editor, GameFeatures, GameplayTags, GAS,
MCPClient, Niagara, PCG, Physics, Plugin, SemanticSearch, SlateInspector,
StateTree, UMG, WorldConditions. Project plugins and Game Feature Plugins
can contribute more. Unreal MCP wraps every registered tool call as an MCP
Tool. Execution is **serialized onto the game thread** — one tool call at a
time, editor UI blocked while each runs.

## Tool-search mode (the default contract)

With `Enable Tool Search` on (default), `tools/list` advertises exactly three
meta-tools:

| Meta-tool | Args | Returns |
|---|---|---|
| `list_toolsets` | — | Registered toolset names + descriptions |
| `describe_toolset` | toolset name | JSON Schemas for every tool in that toolset |
| `call_tool` | toolset/tool name + arguments object | The tool's result, same turn |

Discipline:

- `list_toolsets` once per session; re-run only after `RefreshTools`, plugin
  changes, or reconnect.
- `describe_toolset` before first use of any toolset. Parameter names, types,
  and required fields come from the schema — never from memory or this file.
- Results: primitive results arrive wrapped as `{"result": ...}` (CVar
  `ModelContextProtocol.WrapPODToolResultsInObject`, default true).
  Structured results serialize with field-level schema.
- Errors come back as tool-call errors with the engine-side message — read
  them; they usually name the offending parameter or missing asset.

Eager mode (`Enable Tool Search` off) advertises every tool individually.
Under Hermes that means each tool becomes `mcp_unreal_engine_<tool_name>` at
session start, and `hermes mcp configure unreal-engine` can prune the list.
Schema payload grows with every registered toolset, and tool authors are told
NOT to rely on eager advertising — stay in tool-search mode unless a very
small fixed surface is wanted.

## call_tool dispatch semantics (live-verified, 5.8)

Verified against a running 5.8 server; these details are where naive
clients die:

- `list_toolsets` returns **fully-qualified** toolset names — Python:
  `editor_toolset.toolsets.scene.SceneTools`; C++:
  `EditorToolset.EditorAppToolset`. Epic's prose says "SceneTools"; the
  registry speaks qualified names. Use them verbatim in `describe_toolset`
  and `call_tool`'s `toolset_name`.
- `tool_name` must be the **short** name (`get_current_level`,
  `CaptureViewport`). Passing the fully-qualified tool name fails with
  "Unknown tool" even though `describe_toolset` displays qualified names.
- `call_tool` args: `{"toolset_name": ..., "tool_name": ..., "arguments":
  {...}}`; result returns on the same turn (the HTTP response blocks until
  the game thread finishes the call).
- **`TOptional` parameters must be passed explicitly as `null`** — omitting
  them errors with `input param "X" needs a default value`. E.g.
  `CaptureViewport` minimal call is `{"captureTransform": null,
  "annotations": null, "bShowUI": false}`.
- **Schema `required` is literal.** `find_actors` marks `name`, `tag`,
  `collision_channels` required even though they're semantically optional —
  pass `""` / `[]` to mean "any".
- **Property names are camelCase with UE's `b` prefix intact** at this
  reflection layer: `bUseTemperature`, `bAtmosphereSunLight`, `fogDensity`,
  `bRealTimeCapture`, `mobility`. Writing `useTemperature` does NOT error
  the whole call — the response names each property that could not be set
  (schema-in-error style; READ error text, it lists the exact failures and
  often the full input schema).
- **Object references travel as `{"refPath": "<soft object path>"}`**
  everywhere (actors, classes, components). Class refs use
  `/Script/Module.Class` (e.g. `/Script/Engine.PointLight`); actor refs are
  the full path (`/Temp/Untitled_1.Untitled_1:PersistentLevel.DirectionalLight_UAID_...`).
  Spawn/find tools RETURN refPaths — capture and reuse them.
- **`ObjectTools.set_properties` takes `values` as a JSON *string***, not
  an object: `{"instance": {"refPath": ...}, "values":
  "{\"intensity\": 10.0}"}`. `get_properties` likewise returns a JSON
  string inside `returnValue`. Double-encode/decode accordingly.
- Primitive results arrive wrapped as `{"returnValue": ...}` inside the
  text content block.

### HTTP wire behavior (for raw clients / debugging)

- `initialize` → plain JSON response + `Mcp-Session-Id` header you must
  echo on every subsequent request; `notifications/initialized` → 202
  empty; `tools/call` → **`text/event-stream`**: the result arrives as an
  `event: message` + `data: <jsonrpc>` frame only when the game thread
  finishes. A client that treats the response as plain JSON reads an empty
  body. Send `Accept: application/json, text/event-stream` always.

## Shipped toolsets

The registry is project-dependent; `describe_toolset` on the live server is
the only source of truth for schemas. The core surface below is verified
against EditorToolset's source in the 5.8 install (Python:
`.../EditorToolset/Content/Python/editor_toolset/toolsets/`; C++:
`EditorAppToolset.h`).

**EditorToolset plugin (the core), Python toolsets** (live-verified on 5.8;
qualified prefix `editor_toolset.toolsets.<module>.<Class>`):

| Toolset | Verified tools (subset) |
|---|---|
| `scene.SceneTools` | `load_level`, `get_current_level`, `find_actors` (by name/type/tag/bounds), `add_to_scene_from_class`, `add_to_scene_from_asset`, `remove_from_scene`, `save_actor`, `create_level_instance`, folders |
| `actor.ActorTools` | `get_label`/`set_label`, tags, `get_actor_transform`/`set_actor_transform` (`xform` fields optional = "don't change"), parenting, components |
| `primitive.PrimitiveTools` | `add_cube` (dimensions), `add_sphere` (radius), `add_cylinder`/`add_cone` (radius+height) — adds StaticMeshComponents with `local_transform` to a host actor: spawn `/Script/Engine.Actor`, then compose. The fastest blocking path, zero asset dependencies |
| `object.ObjectTools` | `list_properties` (returns full JSON schema of every property), `get_properties`/`set_properties` (JSON-string `values`), `reset_properties` (restore defaults — also your rollback), `get_class`, `search_subclasses` |
| `material_instance.MaterialInstanceTools` | `create`, `list_parameters`, `get/set_scalar_parameter`, `get/set_vector_parameter` |
| `asset.AssetTools` | `find_assets`, `load_asset`, `exists`, `save_assets`, `is_dirty`, `get_dependencies`/`get_referencers` (check before delete!), `delete`, `move`, `duplicate`, folders, `read_file`/`write_file` (project-scoped) |
| `blueprint.BlueprintTools` (+ dsl/layout/node) | Blueprint authoring |
| `material.MaterialTools`, `static_mesh.StaticMeshTools`, `texture.TextureTools`, `data_table.DataTableTools`, … | per-asset-type operations |
| `programmatic.ProgrammaticToolset` | **the batching escape hatch** — see below |

**`EditorToolset.EditorAppToolset` (C++, same plugin) — the agent's eyes
(full live list):** `CaptureViewport`, `CaptureEditorImage`,
`CaptureAssetImage`, `GetCameraTransform`/`SetCameraTransform`,
`GetSelectedActors`/`SelectActors`/`FocusOnActors`/`GetVisibleActors`,
`WorldPosToScreenCoords`/`ScreenCoordsToWorld`,
`GetSelectedAssets`/`SelectAssets`,
`GetContentBrowserPath`/`SetContentBrowserPath`, `OpenEditorForAsset`,
`GetOpenAssets`, `SearchCVars`, `StartPIE`/`StopPIE`/`IsPIERunning`.

`CaptureViewport` specifics (live-verified): args `{"captureTransform":
<transform-or-null>, "annotations": <config-or-null>, "bShowUI": false}`.
Returns base64 PNG (decode + save it yourself) plus camera
location/rotation/FOV. `captureTransform` captures from any pose WITHOUT
moving the user's viewport — use it as a virtual camera. Annotation config
`{"gridSpacingCm": 500, "gridExtentCm": 3000, "gridHeight": <ground Z>,
"labelActors": true}` overlays a projected ground grid and actor callouts;
**grid coordinate labels are in METERS** (world cm ÷ 100). Use annotated
captures for placement work, clean ones for beauty checks.

Also confirmed live: `ToolsetRegistry.AgentSkillToolset`,
`EditorToolset.LogsToolset` (read Output Log + set verbosity — useful for
self-debugging), `SemanticSearchToolset` (hybrid vector+BM25 asset search),
five `NiagaraToolsets.NiagaraToolset_*` groups, `PCGToolset` (+Spatial),
`UMGToolSet`, three `GASToolsets.*`, `AutomationTestToolset`,
`ConfigSettingsToolset` (read/write Project Settings & Editor Preferences
sections by schema — the remote path to exposure defaults, rendering
settings, etc.), `SlateInspectorToolset`, `PluginToolset`,
`animation_toolset.toolsets.sequencer.SequencerTools` + keyframing/
controlrig/outliner siblings, `aimodule_toolset` BehaviorTreeTools,
`state_tree_toolset` StateTreeTools, and more — 67 toolsets on a blank
project with AllToolsets enabled.

Known gap: no mesh-modelling tools — spawn/place/instance existing meshes,
yes; author new geometry, no. The supported route to parametric geometry is
a custom Python toolset wrapping **Geometry Script** (`UDynamicMesh`:
append box/cylinder/sphere, booleans, then `Create New Static Mesh Asset
from Mesh` to bake an `SM_` asset). For organic/sculpted meshes, model in
Blender (`blender-mcp` skill) and import.

First-session move: `list_toolsets`, then `describe_toolset` each group you
plan to use, and keep those schemas in working memory for the session.

## ProgrammaticToolset — sanctioned batching

The serial-call rule makes N-step edits slow over the wire. The shipped
answer is `ProgrammaticToolset` (verified in `programmatic.py`):

1. `get_execution_environment` — **mandatory first call** (the tool's own
   docstring requires it); returns the allowed modules, script constraints,
   and usage instructions.
2. `execute_tool_script(script)` — runs a **sandboxed** Python script that
   defines `run() -> dict`. Inside, you call other registered tools
   programmatically and glue them with logic — one MCP round-trip for a
   whole loop (e.g. spawn 20 actors with computed transforms).

Sandbox facts (from source): allowed imports are `json`, `math`,
`datetime`, `copy`, `re`, `time` only; `open()` is restricted to
project-contained paths; scripts run inside an editor **transaction scope**
(undo-friendly); it is tool orchestration, NOT general Python — arbitrary
`unreal.*` calls are not the contract. Data returns via `run()`'s dict.

Use it whenever a recipe loop exceeds ~5 homogeneous calls; keep one-off
edits as plain `call_tool`.

## Project Agent Skills (AgentSkillToolset)

Projects and plugins can register **Agent Skills** — named instruction
bundles for project-specific conventions and workflows (naming schemes,
folder layout, canonical multi-step sequences). They are NOT listed by
`list_toolsets`; reach them through `call_tool`:

1. `AgentSkillToolset.ListSkills` → names + descriptions of registered
   skills.
2. If one matches the task, `AgentSkillToolset.GetSkills` on it → full
   instructions, then FOLLOW THEM — a project skill exists precisely
   because the project's way differs from the obvious way, and it takes
   precedence over this skill's generic defaults.

Check at the start of unfamiliar work in any project, not just once ever.

## Seeing your work: screenshots and captures

An agent that can't see the viewport is flying blind. In order of preference:

1. **`EditorAppToolset.CaptureViewport`** (confirmed shipped) — returns the
   image through MCP as base64 PNG with camera metadata; supports capturing
   from an arbitrary transform without disturbing the user's viewport, and
   an optional annotation overlay (world-space meter grid + actor callouts)
   for spatial-placement work. This is the default verification tool.
2. **Console `HighResShot` via any console/exec tool** when you need
   resolutions beyond the viewport: `HighResShot 3840x2160` writes to
   `<Project>/Saved/Screenshots/<Platform>/` on the EDITOR host's
   filesystem; read the file back (same machine) with `vision_analyze`.
3. **Custom toolset escape hatch** for anything else (e.g. camera-actor
   framed captures with MRQ-quality settings).

Always `vision_analyze` the capture and art-direct against the brief before
declaring a milestone done.

## Plugin configuration reference

Editor Preferences > General > Model Context Protocol:

| Property | Default | Notes |
|---|---|---|
| Auto Start Server | `false` | Turn on for frictionless sessions |
| Server Port Number | `8000` | Change on conflict; mirror in Hermes config url |
| Server URL Path | `/mcp` | Same |
| Enable Tool Search | `true` | Keep on (see above) |

Console commands (editor console, backtick):

| Command | Effect |
|---|---|
| `ModelContextProtocol.StartServer [port]` | Start server (optional port override) |
| `ModelContextProtocol.StopServer` | Stop server, close all sessions |
| `ModelContextProtocol.RefreshTools` | Re-poll toolset providers — run after authoring/hot-reload/Game-Feature activation |
| `ModelContextProtocol.GenerateClientConfig <Client\|All>` | Write client config files (ClaudeCode/Cursor/VSCode/Gemini/Codex) — NOT used for Hermes |

Command-line flags for launching the editor pre-configured:
`-ModelContextProtocolStartServer` (force start regardless of preference),
`-ModelContextProtocolPort=N`.

Console variables:

| CVar | Default | Notes |
|---|---|---|
| `ModelContextProtocol.WrapPODToolResultsInObject` | `true` | Primitive results wrapped as `{"result": ...}` |
| `ModelContextProtocol.AudioResultOggFormat` | `false` | OGG instead of WAV for audio results |
| `ModelContextProtocol.ProgressIntervalSeconds` | `1.0` | Min interval between progress notifications |
| `ModelContextProtocol.PaginationPageSize` | `0` | 0 = no pagination of list results |
| `ModelContextProtocol.EnableAnalytics` | `true` | Epic telemetry gate |

## Debugging the connection

- **Output Log** at editor startup logs bind address/port/path — first stop
  when the server seems absent. Port-in-use and missing-dependency failures
  surface here.
- **Log verbosity:** `Log LogModelContextProtocol Verbose` in the editor
  console.
- **MCP Inspector** (`npx @modelcontextprotocol/inspector`, point at
  `http://127.0.0.1:8000/mcp`, transport "Streamable HTTP") lists every
  advertised tool with schemas and offers form-based invocation — isolates
  "server broken" from "agent calling it wrong".
- **After Live Coding / authoring:** connected clients can hold stale
  schemas. `ModelContextProtocol.RefreshTools`, then reconnect (new Hermes
  session) if schemas still look stale.

## Extending the surface: custom toolsets

When shipped tools don't cover an operation, the supported path is authoring
a project toolset — NOT trying to smuggle arbitrary code through unrelated
tools. Python toolsets are first-class and hot-loadable, so prefer them.

### Python toolset (recommended)

Any enabled plugin's `Content/Python/` directory (or the project's) can hold
toolset modules; the registry discovers them at startup. Shape (mirrors
Epic's shipped `ActorTools`):

```python
import unreal
import toolset_registry

@unreal.uclass()
class MySceneTools(unreal.ToolsetDefinition):
    """One-line toolset description — surfaces to the agent in list_toolsets."""

    @toolset_registry.tool_call
    @staticmethod
    def take_viewport_screenshot(filename: str, width: int, height: int) -> str:
        """Capture the active viewport to Saved/Screenshots.

        Args:
            filename: Base filename without extension.
            width: Output width in pixels.
            height: Output height in pixels.

        Returns:
            Absolute path the screenshot will be written to.
        """
        ...
```

Conventions that matter (they generate the schema the agent sees):

- `@unreal.uclass()` on the class; inherit `unreal.ToolsetDefinition`.
- Class docstring = toolset description; write it for an agent audience.
- Each advertised function: `@toolset_registry.tool_call` + `@staticmethod`.
  Functions without the decorator stay private.
- Type hints (`str`, `bool`, `list[str]`, `unreal.Actor`, dataclasses) drive
  the JSON Schema; Google-style docstrings (`Args:`/`Returns:`) become the
  parameter descriptions. Write them with API-surface care.
- Small, single-responsibility tools with structured return types beat
  mega-tools returning prose. Data leaves the tool via its RETURN VALUE —
  `print()`/stdout go to the UE log, not back over MCP.

After authoring: `ModelContextProtocol.RefreshTools` in the editor console,
then re-`list_toolsets` from Hermes. Users on Claude Code can scaffold with
the `create-toolset` skill from Epic's `unreal-mcp` plugin pack; the
conventions above still apply.

### C++ toolset

Derive from `UToolsetDefinition`, mark the class `UCLASS(BlueprintType,
Hidden)`, expose static `UFUNCTION(meta = (AICallable))` methods; doc
comments reflect into schemas. Use only when Python can't reach the API,
when reflected `USTRUCT` signatures are needed, or when the Python boundary
cost matters. Exclude a function with `meta = (AIIgnore)`. Live Coding
propagates edited function bodies, but NEW `UFUNCTION`s require a full
editor restart. There is also a direct-registration path
(`IModelContextProtocolTool` + `IModelContextProtocolModule::AddTool()`) for
runtime-shaped tools; caller owns deregistration.

## Runtime and cooked builds

The server is editor-hosted by default but not editor-only: runtime modules
can host it in cooked builds via `IModelContextProtocolModule::StartServer()`.
The Toolset Registry adapter (and the three tool-search meta-tools) are
editor-only, though — cooked-build tools must be registered explicitly
through `AddTool()` and are advertised eagerly. MCP Resources and Prompts are
not advertised by any shipping toolset.

## Known limitations (5.8, experimental)

- HTTP + SSE transports only; loopback-only listener; non-loopback `Origin`
  headers rejected; no auth layer. Not safe beyond the local machine.
- Serial game-thread execution: overlapping calls unsupported; editor UI
  blocks during each call.
- Feature-incomplete by Epic's own labeling; APIs and data formats subject
  to change without notice.
- Live Coding does not propagate new `UFUNCTION` declarations.
