---
name: rive-mcp
description: "Author, inspect, and integrate interactive Rive graphics."
version: 1.1.0
author: "Brooklyn Nicholson (OutThisLife), Chris Mish (cygnostik, prodyn.ai), Hermes Agent"
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    category: creative
    tags: [Rive, Animation, MCP, Data Binding, State Machines, Luau, Accessibility]
    related_skills: []
---

# Rive Skill

Author interactive graphics through Rive's official CLI/RML or desktop MCP,
then integrate and verify exported assets with the target runtime. This optional
skill includes a guarded MCP client and an original source template; Rive supplies
the compiler/editor. Neither route needs a new Hermes plugin.

## When to Use

- Inspect or author Rive shapes, animations, state machines, View Models,
  layouts, scripts, assets, and exports.
- Integrate a `.riv` into a web, native, or game project; debug data binding,
  rendering, lifecycle, accessibility, and feature compatibility.
- Build authored interactive illustrations, characters, explainers, or instruments.
- Do not replace ordinary HTML controls, simple CSS transitions, or static SVG
  with Rive without a reason. An explicitly artistic brief is a valid reason.

## Prerequisites

- For local source-controlled authoring, the official Rive CLI supports Apple
  Silicon macOS, Linux x86_64, and Windows. Read `references/official-cli.md`;
  install from official sources only with authorization. Local builds need no login.
- Official editor MCP requires the macOS or Windows desktop app, running on the
  **same machine as the MCP client**. Linux supports runtime work, not the
  official desktop editor. A remote Hermes backend's loopback is not the user's desktop.
- For editor MCP, an authorized file must be open before authoring. Confirm file identity and
  asset/font/export rights; signing in alone proves none of these.
- Editor endpoint: `http://127.0.0.1:9791/mcp`. Keep it loopback-only. The separate
  `https://rive.app/docs/mcp` searches documentation, not the editor.
- Hermes includes MCP support. The optional CLI helper needs Python 3.11+ and
  the dependencies in `scripts/requirements.txt`; see `references/cli.md`.
- Resolve this skill's actual directory with `skill_view(name="rive-mcp")`.
  Below, `{SKILL_DIR}` is that returned path, not a fixed home/profile directory.

## How to Run

Choose the route from the task: **CLI/RML** for local source-controlled scenes,
compilation, and headless verification; **MCP** for the currently open editor
file. Runtime-only integration needs neither authoring tool. Do not configure
MCP for a task fully served by CLI.

For CLI work, use `terminal` with `rive --version`, `rive docs`, and
`rive schema TYPE` before writing RML with `write_file`/`patch`. Work in an
isolated project directory: the CLI scans assets and all scripts/shaders there.
The verification loop is `rive PROJECT --verify`, `rive inspect PROJECT --json`,
`rive PROJECT --once`, then `rive PROJECT --screenshot=OUTPUT.png` and target
runtime reload. Inspect PNG output with `vision_analyze`. See
`references/official-cli.md` and `templates/minimal-rml/` for the runnable slice.

Keep CLI `--publish`, `--rev`, `login`, `push`, and `pull` behind their distinct
account/upload/overwrite authorization. Local script-free `.riv` builds do not
need publishing; web-bound scripts require signing. Disable CLI usage analytics
with its documented `RIVE_ANALYTICS=off` control for automation unless opted in.

For editor MCP, use `terminal` for these commands. Inspect the existing entry before changing
anything; never replace a different server configuration silently:

```text
hermes config get mcp_servers.rive
hermes mcp test rive
```

If unconfigured and setup is authorized, prefer the catalog entry when available:

```text
hermes mcp install rive
hermes config set mcp_servers.rive.sampling.enabled false
hermes config get mcp_servers.rive
hermes mcp test rive
```

The catalog starts with inspection tools selected. Use `hermes mcp configure rive`
to explicitly enable required authoring tools. Before catalog availability, the
manual setup in `references/agent-first.md` uses the same official endpoint.
Skill installation, editor installation, and MCP configuration are separate actions.

Discover exposed tools, read their schemas, then call `session_info` and
`open_file_editor` with `getCurrentFile` (enable it if needed). Use names exactly
as returned by discovery, not guessed tool prefixes. If tools are hidden, use the
available tool-search/schema tools; if Tool Slimmer is installed and hides a
required tool, request `tool_slimmer_request_full_tools`. A fresh MCP test proves
connectivity, not availability in this conversation. Use a new session or a
supported user-requested `/reload-mcp`; never manually rewrite cached prompts.

When native calls are unavailable, read `references/cli.md`, then use `terminal`
with the resolved interpreter and skill path:

```text
python "{SKILL_DIR}/scripts/rive_doctor.py" --json
python "{SKILL_DIR}/scripts/rive_mcp.py" list
python "{SKILL_DIR}/scripts/rive_mcp.py" schema session_info
python "{SKILL_DIR}/scripts/rive_mcp.py" call session_info --args "{SKILL_DIR}/scripts/empty-args.json"
```

## Quick Reference

| Work | Tool family / reference |
| --- | --- |
| Local authoring/build/test | Official `rive` CLI and RML; `references/official-cli.md` |
| File identity and scope | `session_info`, `open_file_editor`; `references/agent-first.md` |
| Geometry and property inspection | `path_editor`, `query_objects`, `query_property_keys`, `query_property_values` |
| Timelines and graph wiring | `animation_editor`; query the graph, not only names |
| Typed models and bindings | `viewmodel_editor`; `references/authoring-and-scripting.md` |
| Luau and WGSL | `get_scripting_reference`, `manage_scripts`, `text_editor`, diagnostics/tests |
| Visual feedback | `capture_artboard` when advertised; inspect returned PNG with `vision_analyze` |
| Runtime and source export | `export_file` with explicit `riv` or `rev` format |
| Host integration and accessibility | `references/runtime-integration.md` |
| Composition and interaction | `references/creative-direction.md` |
| Sources, license and evidence limits | `references/research.md`, `NOTICE.md` |

## Procedure

### 1. Define the visual contract

Write `cause → state change → visual consequence → stable result`. Specify the
artboard, state machine, View Model paths/types/ranges/defaults, layout, renderer,
asset rights, motion policy, and static fallback. Start from
`templates/asset-contract.json`; unresolved fields are not an authored asset.
Follow the project's own brand, not this skill's contributors' brand.

**Done:** the intended interaction and host/asset ownership are explicit.

### 2. Inspect, author, and read back

For CLI, author text RML using live `rive schema` types, unique IDs, and explicit
`main` artboard configuration. Use radians for RML rotation and ARGB hex without
`#`; editor MCP degrees/percentages are a separate contract. Run compile,
inspection, and actual screenshot checks; any one alone can miss invisible or
misbound art. Read back driven values: unknown `--data` paths can be ignored and
unknown artboard names can silently fall back. Details: `references/official-cli.md`.

For MCP, confirm the exact current file ID. Resolve object IDs and property keys from
live queries. Use one controller and bounded changes; verify the active file
immediately before each mutation, then read the affected objects/properties back.
A file check is not an atomic lock: do not allow concurrent tab switches.

Prefer parametric primitives; use freeform paths for custom geometry. Reuse
pristine default timelines/state machines instead of creating duplicates. Query
complete graph wiring. For new contracts prefer View Models; preserve working
legacy inputs unless migration is needed. Bind strings to Text Runs, not Text
containers. Reuse converters and set listener targets explicitly.

For scripting, fetch actual API definitions, create/read source, diagnose,
compile, run scoped tests, attach the asset, and verify execution. A script asset
existing does not prove scene attachment. Simulation and tests can execute side
effects; they are not read-only. See `references/authoring-and-scripting.md`.

**Done:** requested structure and values are read back; visual work is reviewed
through an actual capture or UI, not inferred from a hierarchy.

### 3. Export and independently reload

For MCP use `export_file` with explicit format and an approved existing destination.
For CLI use `--once` for a local unsigned build; `--publish` signs via Rive's API
and can watermark the output. `--rev` requires login. Do not use a cloud operation
as a routine local-build check.
Capture its returned path: filenames derive from the document, and exports may
include unsaved changes. `.riv` is runtime output; `.rev` is editable backup.
Check current entitlement for the requested format. Public pricing and export pages can disagree with the live editor schema;
verify the requested format rather than assuming universal entitlement.
A permitted runtime export does not substitute for promised editable source; never bypass a denied operation.

Reload `.riv` with the target runtime and reopen `.rev` when editable handoff is
promised. Loading a header alone does not prove feature fidelity.

**Done:** each promised deliverable exists and works independently, or its exact
permission/prerequisite blocker is stated separately.

### 4. Integrate behind a semantic host

Build meaningful HTML/native controls and fallback first. Keep business state
in the host; animation completion never establishes a real operation's success.
Validate actual typed properties after load. Verify their visible effect, not
only a JavaScript mirror. Pin matching runtime JS/WASM and check the current
feature matrix; editor shader support does not establish target support.

Own reduced motion, explicit play/pause, visibility, async generations, and
terminal cleanup separately. Static states must be useful composed poses, not
arbitrary frozen transitions. Keep the latest desired state during load; prevent
late completions from reviving disposed instances. Set asset-CDN and WASM-fallback
policies separately. See `references/runtime-integration.md`.

**Done:** the intended interaction, fallback, and lifecycle work in the actual host.

## Pitfalls

- `hermes mcp add` is interactive; cancellation can look like a successful CLI
  exit. Read saved configuration and test actual discovery.
- A closed editor is an unavailable dependency, not proof of a crashing server.
  Do not add autostart, background polling, or disable configuration unasked.
- A tool named like a read, or marked `readOnlyHint`, is not automatically safe.
  Newly discovered helper calls require explicit review/authorization.
- Read nested `isError`, `success:false`, and error envelopes even after HTTP 200.
  Never automatically retry an uncertain mutation.
- Public headings about files do not establish blank-file creation. Use the live
  schema or authorized UI; never opportunistically edit another open document.
- Rive documentation still mentions `End Prompt` in a client workflow. It is not
  a universal MCP transaction/commit tool; honor the actual editor confirmation UI.
- Official CLI/RML now supports editor-free authoring. Third-party RiveMCP
  is a separate product and license, not an official
  requirement, a fallback for denied exports, or included in this package.

## Verification

Match proof to scope: a small authoring slice needs exact-file checks, readback,
and visual review; production delivery needs the applicable full host checks.

- Prove typed binding effects with real pixels/geometry/text and relevant poses.
- Exercise missing/corrupt asset/WASM, invalid properties, cancellation/retry,
  live reduced motion, hidden/offscreen behavior, and unmount/remount.
- Check keyboard/focus, native/static content, layout and typography at relevant
  sizes. Experimental semantics and axe checks are not screen-reader certification.
- Reproduce with recorded dependencies and clean inputs. Separate documentary,
  synthetic transport, live editor, runtime, and native-device evidence.
- Report exact tested versions, actual artifacts, rights, and remaining blockers.

Maintained with contributions from [ProDyn](https://prodyn.ai). Original upstream
proposal by Brooklyn Nicholson (OutThisLife); lineage and MIT terms in `NOTICE.md`
and `LICENSE`. Attribution is documentation only—no telemetry or forced output branding.
