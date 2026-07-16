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
auto-enabled). Epic's shipped toolsets are delivered by a separate toolset
provider plugin (**AllToolsets**); project plugins and Game Feature Plugins
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

## call_tool dispatch semantics

Confirmed against Epic's own agent-facing skill for this server:

- `call_tool` takes `toolset_name`, `tool_name`, and an `arguments` object
  matching the schema from `describe_toolset`. The result returns on the
  same turn.
- Tool identities are effectively dotted: `BlueprintTools.create`,
  `SequencerTools.create_level_sequence`,
  `LiveCodingToolset.CompileLiveCoding` — they are dispatched server-side
  and never appear as native MCP tools while tool search is on.
- Top-level dispatch (omitting `toolset_name`) is reserved for tools
  registered directly on the MCP server — and is rejected for `call_tool`
  itself.

## Shipped toolsets

The registry is project-dependent; treat this as orientation, not contract —
`describe_toolset` on the live server is the only source of truth for tool
names and schemas. Epic's plugin pack describes the shipped surface as
"hundreds of tools across 30+ toolsets": actors, blueprints, materials,
Niagara, Control Rigs, Sequencer, State Trees, widgets, Gameplay Ability
System, automation testing, Live Coding.

Toolset names confirmed in Epic's docs and Epic's own Claude Code skill pack:

| Toolset | Scope |
|---|---|
| `SceneTools` | Scene-level queries and edits |
| `ActorTools` | Inspect/modify actors: transforms, labels, parent-child relationships, components |
| `MaterialInstanceTools` | Create/configure material instances |
| `MaterialTools` | Material assets |
| `ObjectTools` | Generic UObject property inspection/editing |
| `BlueprintTools` | Blueprint creation/editing (e.g. `BlueprintTools.create`) |
| `StaticMeshTools` | Static mesh asset operations |
| `LevelTools` | Level operations |
| `SequencerTools` | Level Sequences (e.g. `create_level_sequence`) |
| `LiveCodingToolset` | C++ Live Coding recompile (`CompileLiveCoding` blocks until the compile finishes and surfaces compiler diagnostics) |
| `AgentSkillToolset` | Project-registered Agent Skills (see below) |
| `GASToolsets` (plugin, C++) | Gameplay Ability System attributes — ships disabled; enabling prompts an experimental-feature warning |

Known gap: the shipped toolsets contain **no mesh-modelling tools** — you
can spawn/place/instance existing meshes but not author new geometry. The
supported route to parametric geometry is a custom Python toolset wrapping
**Geometry Script** (`UDynamicMesh`: append box/cylinder/sphere, booleans,
then `Create New Static Mesh Asset from Mesh` to bake an `SM_` asset). For
organic/sculpted meshes, model in Blender (`blender-mcp` skill) and import.

First-session move: `list_toolsets`, then `describe_toolset` each group you
plan to use, and keep those schemas in working memory for the session.

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

1. **A shipped screenshot/viewport tool, if the registry advertises one** —
   check `list_toolsets`/`describe_toolset` output for viewport, screenshot,
   or thumbnail capture tools and use those (they return the image through
   MCP directly).
2. **Console command via any shipped console/exec tool**: `HighResShot 1`
   writes the current viewport to
   `<Project>/Saved/Screenshots/<Platform>/` on the EDITOR host's
   filesystem. `HighResShot 3840x2160` for fixed resolution,
   `HighResShot 2` for 2× viewport. Read the file back with `read_file`/
   `vision_analyze` (same machine) — remember paths resolve on the editor
   host.
3. **Custom toolset escape hatch** (below) exposing
   `unreal.AutomationLibrary.take_high_res_screenshot()` or viewport
   capture, when nothing shipped covers it.

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
