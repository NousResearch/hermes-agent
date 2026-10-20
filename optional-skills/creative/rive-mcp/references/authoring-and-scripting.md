# Authoring, data graphs and code

This is executable procedural guidance derived from official documentation and the captured live MCP schema. Operations below remain recipes until exercised on the intended authorized file. The live tool schema wins over stale examples. Official references: https://rive.app/docs/editor/ai/mcp.md and https://rive.app/docs/llms.txt.

## Artboard and geometry

1. Inspect current file/artboards. Create only the required artboard; leave stage x/y unset unless deliberately positioning it, because Rive auto-places new artboards.
2. Prefer `path_editor.createParametricShapes` for rectangles/ellipses/polygons/stars. For custom shapes provide at least one path, `moveTo`, drawing commands and `close` for fills. A shape without paths/commands can exist yet be invisible.
3. Parent coordinates and path coordinates differ: shape x/y is in parent space; path commands in shape-local space. Layout origins are top-left. Use a same-size child layout/container when aligning a shape by its center.
4. Query property keys and enum mappings before writes. For editor MCP, rotation is degrees and percent-displayed scale/opacity/trim use UI-style percentages. Numeric key values are runtime/editor schema details, not universal constants. Convert time to frames using the actual timeline FPS.
5. Colors in authoring payloads use documented ARGB `#aarrggbb`, not CSS `#rrggbbaa`. Resolve the project's token roles first and use a tested converter; avoid manual swatch copies becoming a second palette.
6. Gradients/trim are persistent objects. Query IDs and update existing effects rather than stacking duplicates. Boolean combine operations can flatten geometry/remove consumed shapes and destroy animation keys; prefer non-destructive grouping unless the destructive result is intended and baseline is recoverable.

Minimal shape payload after resolving a real artboard ID:

```json
{"command":"createParametricShapes","data":{"createParametricShapes":{"shapes":[{"primitive":"rectangle","name":"SignalPlate","parentId":"RESOLVE_ARTBOARD_ID","x":64,"y":64,"width":96,"height":32,"paints":[{"paintType":"fill","color":"RESOLVE_TOKEN_AS_ARGB"}]}]}}}
```

This template intentionally does not run until IDs/color are resolved. Read hierarchy plus exact x/y/dimensions/fill back; a create response alone is not verification.

## Timelines, state machines and interruption

- A new artboard in the captured editor has a pristine `Timeline` and default state machine already linked from Entry. List, rename and key the existing timeline. Only add another animation/machine for a genuine separate requirement; never leave an unused default next to duplicated named animations.
- Use real animation IDs and actual integer property keys in `modifyKeyFrames`. The smaller-frame key's interpolation controls the interval. Read keys back with `queryKeyFrames`.
- Define state meaning before graph complexity. A few purposeful states with tested cancellation beat hundreds of automatically generated transitions.
- Inspect with `queryStateMachine`, not only `listStateMachines`. Test entry, deliberate transitions, conflicting signals, reset, exit-time behavior and interrupted transitions. Beware any-state transitions that repeatedly retrigger or priority that masks another state.
- `simulateStateMachine` can drive View Model properties at frames and return state/transition/event trace plus resting states. Compare driven against undriven runs: the default Entry transition already happens unconditionally. It does not render pixels. It reverts written properties but may execute script/listener side effects, so run only a scoped harmless graph.
- Simulation refuses while a timeline/state machine is open for editing. Do not close/change user UI without permission. Runtime tests can separately exercise exported behavior.

## View Models and reusable contracts

Prefer View Model-based interaction for new assets, while preserving working legacy input contracts unless migration is needed. Distinguish:
- **Definition:** schema of typed properties.
- **Instance:** actual values for one use; do not accidentally share mutable instance state across independent visuals.
- **Binding:** path/direction/converter between an instance and an object property.
- **Host truth:** real domain state outside Rive; the illustration consumes validated display state.

Create/inspect enums before enum properties. Resolve nested paths explicitly and bind the correct instance to artboard/component. Bind string values to **Text Runs**, not Text containers. Prefer explicit model paths when repeated/nested components make names ambiguous. Inspect `listDataBinds`, directions, converters and relative flags after changes.

List/reuse converters; the captured MCP cannot delete them. Use conversion helpers for actual unit/format changes, not to hide a mismatched schema. Component Lists instantiate from a bound list at runtime; do not manually add child components. Stateful components have different ownership from manually assigned data contexts—inspect actual feature documentation and target support.

Listeners need a target component. An untargeted listener may not export. For artboard-wide pointer behavior, target a covering shape; View Model/event listeners may target the artboard by its actual name. New runtime signals should use typed View Model triggers/properties where appropriate; legacy events remain a separate contract. Never let an asset event invoke arbitrary host URLs, commands or privileged actions.

Custom Property Groups can expose keyframed/bound local properties but do not replace View Models for interaction conditions. Tags organize the editor, are not exported behavior and may be plan-gated.

## Luau: declarative first, code when needed

Official entrypoints: https://rive.app/docs/scripting/getting-started.md, https://rive.app/docs/scripting/creating-scripts.md, https://rive.app/docs/scripting/debugging/unit-testing.md.

Protocols include Node, Layout, Converter, Path Effect, Transition Condition, Listener Action and Tests; choose the actual protocol for the job. Keep authorization, network/data access and business decisions in the host. Scripts should own bounded illustration-local rules.

Agent loop:
1. `get_scripting_reference` for needed `rive/*` types, `viewmodel_definitions`, `example_node` or `example_test`.
2. Write a failing pure-helper Test case against the actual Tester API. Use seeded typed code; `manage_scripts.create` has no generic invented `protocol` parameter in the captured schema.
3. Create source with `manage_scripts` (`code_file_type: luau` or `wgsl`); read actual returned path/source with `text_editor.view`; verify reported language/protocol through `get_scripts`.
4. Run `script_diagnostics`, `recompile_all_scripts`, `run_tests` on the actual Tests path; preserve real fail→repair→pass evidence. Script execution is not read-only simply because it is a test.
5. Attach the node/layout asset where required; creating an asset does not instantiate it. If no verified MCP operation exists for attachment, report that exact gap and use approved UI rather than pretend.
6. Trigger scoped execution, then `read_console`. Console reads completed execution, not a command to play. Test protocol success does not prove scene drawing.

Use the editor execution-time cap where needed and verify actual callback units. Prevent unbounded advance loops and expensive per-frame allocations; define deterministic seeds if randomness affects tests. Do not invent a generic Luau CLI that understands Rive's embedded types.

## WGSL/GPU Canvas boundary

https://rive.app/docs/scripting/wgsl-shaders.md documents GPU Canvas, vertex/fragment WGSL and target compilation. It excludes compute, override declarations and other listed features; up to four bind groups are documented. Export target variants matter and the original source is not the runtime handoff.

**Check https://rive.app/docs/feature-support.md first.** Broad 'same as editor' renderer prose does not override individual feature cells. Verify GPU Canvas, scripting and semantics against the exact target and runtime release before selecting them. Do not install native/game toolchains until a project needs them.

## Export/source handoff

Use `export_file {format:'riv'|'rev', destination: EXISTING_ABSOLUTE_DIRECTORY, ...}` after exact-file check. `.rev` is editable backup; `.riv` is runtime output and can omit editor-only names/information. Check entitlement by format rather than assuming all export is paid. The desktop 0.9.9 MCP schema says runtime `.riv` works on every plan while `.rev` backup may require an upgrade. Public pricing/export pages can lag or disagree; the actual authorized operation is the proof. Runtime MIT licensing is separate from source, asset and export rights.

The tool snapshots current in-memory changes, does not overwrite and derives filename from document name. `.riv` can need network to resolve hosted assets; `.rev` embeds assets by default unless deliberately disabled. `inline_base64` is only for the tool's permission-denied fallback recipe, not a routine option or permission bypass.

Record editor file URL/ID, source revision, returned paths, export settings, asset/font rights and actual runtime schema. Independently reopen source and reload/render runtime when promised. New files may load on older runtimes while unsupported newer features are skipped; a successful load is not feature-complete fidelity. Sources: https://rive.app/docs/editor/exporting/exporting-for-runtime.md and https://rive.app/docs/editor/exporting/exporting-for-backup.md.
