# Advanced workflows (live-verified on UE 5.8)

Everything here was executed against a running 5.8 editor unless explicitly
marked schema-verified. Where behavior diverged from Epic's docs, this file
records what the server actually did.

## ProgrammaticToolset — batching without breaking the serial rule

`editor_toolset.toolsets.programmatic.ProgrammaticToolset` is the sanctioned
way to do N operations in one MCP round-trip. It is ONE tool call on the
game thread, so the serial-call rule stays intact; you're just letting a
script make the sub-calls server-side.

Contract (verified):

1. Call `get_execution_environment` ONCE per session before the first
   script. It returns `instructions` (read them — they are authoritative),
   `supported_modules`, and `language`.
2. `execute_tool_script` takes `{"script": "<python>"}`. The script must
   define `run() -> Dict[str, Any]`.
3. Inside the script, `execute_tool(tool_name, json_input)` calls any
   registered tool. `tool_name` is FULLY QUALIFIED INCLUDING the tool
   segment (`"editor_toolset.toolsets.primitive.PrimitiveTools.add_cube"`)
   — unlike top-level `call_tool`, there is no separate toolset/tool split.
   `json_input` is a JSON **string** (use `json.dumps`).
4. `execute_tool` returns a dict-like object; unwrap results with
   `["returnValue"]`. It raises `RuntimeError` on failure — no manual error
   checking.
5. Allowed imports (5.8): `json`, `math`, `datetime`, `copy`, `re`, `time`.
   Nothing else — no `unreal`, no `os`, no file I/O.
6. The whole script's return value comes back as a JSON string in
   `returnValue`.

Worked example (verified — 12-column colonnade, 36 components, ONE
round-trip that would otherwise be 37 serial calls):

```python
import json, math

def add_cylinder(actor_ref, name, radius, height, x, y, z):
    return execute_tool(
        "editor_toolset.toolsets.primitive.PrimitiveTools.add_cylinder",
        json.dumps({"actor": actor_ref, "name": name, "radius": radius,
                    "height": height,
                    "local_transform": {"location": {"x": x, "y": y, "z": z}}}))

def run():
    spawn = execute_tool(
        "editor_toolset.toolsets.scene.SceneTools.add_to_scene_from_class",
        json.dumps({"actor_type": {"refPath": "/Script/Engine.Actor"},
                    "name": "Colonnade",
                    "xform": {"location": {"x": 75800, "y": 84900, "z": 44300}}}))
    host = spawn["returnValue"]
    n, ring_r = 12, 900.0
    for i in range(n):
        a = 2.0 * math.pi * i / n
        add_cylinder(host, "Shaft_%02d" % i, 40, 360,
                     ring_r * math.cos(a), ring_r * math.sin(a), 210)
    return {"colonnade": host["refPath"], "columns": n}
```

When to reach for it: any loop over 5+ homogeneous operations (placement
rings, grid scatter, bulk renames, bulk property sweeps). When NOT to:
operations where you need to see intermediate results to decide the next
step — the script can't ask you questions mid-run.

Failure surface: `print()` goes to the UE log, not the MCP return — return
diagnostics in the result dict instead. A script exception returns the
traceback as the tool error text.

## Blueprint authoring — the DSL loop

`editor_toolset.toolsets.blueprint.BlueprintTools` (53 tools) authors real
Blueprints. The graph surface is an s-expression DSL, and the workflow that
survives contact with the live server is:

1. **`create`** — `{"folder_path": "/Game/Blueprints", "asset_name":
   "BP_Spinner", "asset_type": {"refPath": "/Script/Engine.Actor"}}` →
   returns the Blueprint's refPath (`/Game/Blueprints/BP_Spinner.BP_Spinner`).
2. **`list_graphs`** — returns graph refPaths in colon form:
   `...BP_Spinner.BP_Spinner:EventGraph`,
   `...BP_Spinner.BP_Spinner:UserConstructionScript`.
3. **`get_graph_dsl_docs`** — pulls ~9k chars of grammar documentation off
   the live server. Read it before writing DSL; it covers `event`/`fn`,
   `bind`, `if`/`for`/`while`/`switch`, multi-exec continuation blocks
   (`(:then ...)`, `(:CastFailed ...)`), auto-generated underscore
   variables for data output pins, and quoted pin names.
4. **Resolve every node ID with `find_node_types` BEFORE writing DSL** —
   `{"graph": {"refPath": "<graph>"}, "type_id_filter": "MakeRotator",
   "context_pins": []}` → exact IDs. Node IDs are pipe-delimited category
   paths and must match the live registry exactly. Verified gotchas:
   - Engine events use K2 display names: `EventTick` (with `DeltaSeconds`
     param), `EventBeginPlay` — `(event Tick ...)` fails with
     "AddEvent|Tick does not exist".
   - `Math|Rotator|MakeRotator`, not bare `MakeRotator`.
   - `Utilities|Operators|Multiply` (wildcard operator), not
     `Multiply_FloatFloat`.
   - `Transformation|AddActorLocalRotation`, not
     `Utilities|Transformation|AddActorLocalRotation` — category prefixes
     in doc examples don't always match the live registry. The registry
     wins.
   - There is no `(self)` node; the target is implicit — omit `:self`
     entirely for calls on the owning actor.
5. **`write_graph_dsl`** — `{"graph": {"refPath": "<EventGraph>"}, "code":
   "<dsl>"}`. Returns `null` on success. On failure the error is an
   AssertionError naming the exact failing node and its enclosing form —
   fix ONE node at a time and rerun; the error moves to the next problem.
6. **`compile_blueprint`** — `{"blueprint": {"refPath": ...},
   "warnings_as_errors": false}`. Returns `null` on success.
7. **Spawn an instance** — `SceneTools.add_to_scene_from_asset` with
   `{"asset_path": "/Game/Blueprints/BP_Spinner.BP_Spinner", ...}`.
   NOTE: this tool takes `asset_path` as a plain STRING, not an `asset`
   refPath object — the error schema is the tiebreaker (see below). The
   spawned actor's class is `BP_Spinner_C` (the `_C` generated-class
   suffix, visible in the returned refPath).

Verified end-to-end: created `BP_Spinner`, wrote a Tick handler that yaws
the actor 90°/s (`(event EventTick (DeltaSeconds) (Transformation|AddActorLocalRotation
:DeltaRotation (Math|Rotator|MakeRotator :Roll 0.0 :Pitch 0.0 :Yaw
(Utilities|Operators|Multiply DeltaSeconds 90.0))))`), compiled clean,
spawned it, and attached a visible mesh via `PrimitiveTools.add_cube` on
the instance.

Variables, functions, dispatchers: `add_variable` (`type_name` strings),
`add_object_variable`/`add_struct_variable`, `add_function_graph` +
`add_function_param`, `add_event_dispatcher`, `set_variable_replication` —
same refPath discipline. `read_graph_dsl` round-trips existing graphs back
to DSL for inspection/editing.

## Schema-in-error is a first-class discovery mechanism

When a call is missing/mistyping a required param, the server returns the
COMPLETE input schema of the tool in the error text. This is faster than
re-running `describe_toolset` and is authoritative for the exact function
you called. Two verified cases where it corrected the surface:

- `add_to_scene_from_asset` — advertised conceptually as taking an asset
  reference; live schema requires `asset_path` (string).
- `StartPIE` — `{}` fails, and the error hands you the full
  `PIESessionOptions` schema.

Rule: on a param error, READ the schema in the error before anything else.

## PIE sessions (schema-verified)

`EditorAppToolset.StartPIE` requires an `options` object
(`FPIESessionOptions`):

- `bSimulate` (required): `true` = Simulate-In-Editor — world ticks,
  physics/AI run, no player pawn possessed. `false` = standard PIE with
  possession.
- `playMode` (required): `PlayMode_InViewPort`, `PlayMode_InEditorFloating`,
  `PlayMode_Simulate`, etc. Out-of-process modes (NewProcess, MobilePreview,
  VR, QuickLaunch) are silently downgraded to in-viewport — the tool needs
  in-process PIE for delegate-based completion tracking.
- `warmupSeconds` (required): extra settle time after the engine fires
  PostPIEStarted (BeginPlay has run) before the call returns. `0` = return
  as soon as PIE is up.
- `startTransform` (optional): spawn the pawn/reference at a specific
  transform instead of PlayerStart.

`IsPIERunning` returns a bare boolean. Runtime-state inspection during PIE
(actor transforms ticking, LogsToolset reads) plus `StopPIE` complete the
loop: start simulate → read state / logs → stop → judge.

The test loop this enables: compile Blueprint → StartPIE (simulate) →
sample an actor transform twice a few seconds apart → confirm your Tick
logic actually runs → StopPIE. Remember pitfall 15: PIE mutates world
state; take editor-world measurements before or after, not across, a PIE
session.

## Sequencer — orientation for a 140-tool surface

`animation_toolset.toolsets.sequencer.SequencerTools` is the largest
toolset (140 tools) and follows an open-sequence-implicit-target model:
`create_level_sequence` / `open_sequence` / `get_focused_sequence`, then
most calls operate on the focused sequence.

Capability map (names verified via describe; group by prefix):

- **Structure**: `add_actors` (possessables), `add_spawnable_from_class` /
  `add_spawnable_from_instance`, `create_camera` (returns a camera-cut
  ready binding), bindings CRUD (`get_bindings`, `find_binding_by_name`,
  `remove_binding`, `rebind_component`, `fix_actor_references`).
- **Tracks/sections**: `add_track_to_binding` / `add_track_to_sequence`,
  `add_section`, `set_section_range`/`set_section_blend_type`/ease in-out,
  `set_camera_cut_binding`.
- **Timing**: `set_playback_range`, `set_display_rate`,
  `set_tick_resolution`, `set_work_range`, marked frames.
- **Transport**: `play`, `pause`, `play_to`, `set_playhead_frame`,
  `force_evaluate`, `set_playback_speed`.
- **Keyframing** lives in the sibling
  `animation_toolset.toolsets.keyframing.SequencerKeyframingTools` (22
  tools): `get_channel_names` → `add_key_float`/`add_key_bool`/... →
  `get_keys`, `set_default_value`, `bake_channel_keys`, curve-editor
  control.
- **Baking/IO**: `bake_transform`, `import_export` sibling toolset (FBX
  etc.), `copy_tracks`/`paste_tracks`.
- **Runtime conditions / custom bindings / ControlRig**: dedicated sibling
  toolsets (`SequencerConditionTools`, `SequencerCustomBindingTools`,
  `SequencerControlRigTools`, `ControlRigTools`).

Minimal cinematic recipe skeleton: `create_level_sequence` →
`create_camera` → `add_actors` for subjects → keyframe camera transform
channels at frame A and B → `set_playback_range` → `play` → capture/judge.

## Editor self-debugging with LogsToolset

`EditorToolset.LogsToolset`: `GetLogCategories`, `Get/SetVerbosity`,
`GetLogEntries`. After any failed operation or suspicious silence, pull
recent log entries filtered to the relevant category (`LogBlueprint`,
`LogNiagara`, `LogModelContextProtocol`, ...) instead of guessing. This is
also how you see `print()` output from ProgrammaticToolset scripts and
Python toolset internals.

## Automation testing

`AutomationTestToolset.AutomationTestToolset`: `DiscoverTests` /
`ListTests` → `RunTests` or `RunTestsByFilter` → `GetTestStatus` (poll —
test runs are async on the editor) → `GetTestResults` → `StopTests` if
needed. This is the CI-shaped loop for "make a change, prove nothing
broke" inside a live editor session.

## Asset intelligence

- `SemanticSearchToolset`: `Search` (hybrid vector + BM25 over project
  assets) and `FindSimilar` — use for "find me a rusty metal material"
  style requests before falling back to `AssetTools.find_assets` name
  matching.
- `StaticMeshTools`: `import_file` (bring in external meshes),
  `set_nanite_enabled`, `generate_lods`/`set_lod_thresholds`,
  `generate_convex_collisions`, `get_triangle_count`/`get_bounds` — the
  optimization pass after any import.
- `ConfigSettingsToolset`: `ListContainers`/`ListCategories`/`ListSections`
  → `GetSectionSchema` → `SetSectionProperties` (saves to config). The
  remote path to Project Settings and Editor Preferences — rendering
  defaults, exposure defaults, auto-start flags — without touching ini
  files by hand.
- `ToolsetRegistry.AgentSkillToolset`: `ListSkills`/`GetSkills`/
  `CreateSkill`/`UpdateSkill` — project-embedded agent skills that ship
  with the .uproject. If a project has them, list them FIRST; they encode
  project-specific conventions that outrank this file's generic guidance.

## Choosing a strategy (decision table)

| Situation | Reach for |
|---|---|
| 5+ homogeneous ops (scatter, bulk edit) | ProgrammaticToolset script |
| Gameplay behavior, event logic | BlueprintTools DSL loop |
| Camera moves / animation over time | SequencerTools + KeyframingTools |
| "Does it actually behave at runtime?" | StartPIE (simulate) + transform/log sampling |
| "Find an asset like X" | SemanticSearch, then AssetTools |
| Imported mesh is heavy | StaticMeshTools nanite/LOD/collision pass |
| Change editor/project settings | ConfigSettingsToolset |
| Anything failed silently | LogsToolset GetLogEntries |
| Project has its own agent skills | AgentSkillToolset first |
