# Official editor MCP and scope

## Setup through Hermes

Use `terminal` to inspect `hermes config get mcp_servers.rive` before setup.
Change only the active profile and preserve existing values. On an authorized,
unconfigured profile without the catalog entry, use:

```text
hermes config set mcp_servers.rive '{url: http://127.0.0.1:9791/mcp, connect_timeout: 10, timeout: 120, sampling: {enabled: false}}'
hermes config get mcp_servers.rive
hermes mcp test rive
```

The mapping is one quoted argument (POSIX shells and PowerShell). On other
shells, use `hermes mcp add --help` and the interactive setup instead of guessing
quoting. `hermes mcp add` may prompt for authentication and tool selection;
cancellation can return exit zero without saving. Always read back the entry.
No credential is needed for the documented loopback server.

Catalog installation uses `hermes mcp install rive`, followed by explicit
sampling disablement and readback as shown in SKILL.md. The catalog's initial
selection exposes inspection tools; `hermes mcp configure rive` enables the
needed authoring tools. Those native tools do not inherit the Python helper's
file guards. Apply the same authorization, identity, and readback discipline.

The editor and MCP client must run on the same machine. On a remote backend,
127.0.0.1 means that backend—not the desktop displaying Hermes. Do not expose,
proxy, or tunnel this unauthenticated endpoint as a convenience workaround.
No startup service, autostart editor, polling job, or third-party plugin is needed.

A fresh `hermes mcp test rive` verifies discovery, not this session's tool
snapshot. Use supported tool discovery first. A new session or an explicitly
requested `/reload-mcp` can refresh tools; do not edit cached prompts. The
included helper provides an explicit invocation without persistent connection
configuration. Do not describe `lazy: true` as guaranteed strict on-demand mode:
cache-miss discovery and retries depend on the installed Hermes version.

## Discovery and file identity

Enumerate the live catalog completely. Current tool names, command enums,
property keys, and schema fields override examples in prose. Never infer trust
from names or `readOnlyHint` alone. The read-policy in `scripts/rive_mcp.py` is
reviewed independently and new tools default to mutation-class.

Before an edit, call `session_info` and `open_file_editor` → `getCurrentFile`.
Check the exact file ID/URL and artboard. Cloud documents are not filesystem
paths. No active file means authoring is blocked; ask for the intended file or
use approved UI bootstrap. Public wording about creating files is not proof of
a generic blank-file command. Do not use another open document opportunistically.

Resolve IDs from current readback. One controller owns a mutation sequence.
Capture the baseline, recheck file identity immediately before the call, make
one bounded change, and read the exact objects/properties back. This is not an
atomic editor transaction; a concurrent tab switch can race the check.

Rive's public guide still describes typing **End Prompt** in a client workflow.
Honor actual editor prompts, but do not invent an `End Prompt` MCP method or
assume edits are safely staged until that phrase is sent.

## Tool families

| Work | Live names / examples to rediscover |
| --- | --- |
| Session/artboards | `session_info`, `open_file_editor`, `list_artboards` |
| Scene reads | `get_artboard_hierarchy`, `find_objects`, `query_objects`, `query_property_keys`, `query_property_values`, `get_selection` |
| Scene changes | `set_property_values`, `rename_objects`, `select_objects`, `duplicate_objects`, `reorder_objects`, `reparent_objects`, `delete_objects`, `group_editor` |
| Geometry | `path_editor`, `mesh_rigging_tool` |
| Layout/reuse | `layout_editor`, `component_editor` |
| Data | `viewmodel_editor`, `property_group_editor`, `tag_editor` |
| Motion | `animation_editor`, `create_listeners` |
| Assets/export | `assets_tool`, `upload_asset`, `upload_rev`, `export_file` |
| Visual feedback | `capture_artboard` (advertised by desktop 0.9.9) |
| Code | `manage_scripts`, `text_editor`, `script_diagnostics`, `get_scripts`, `run_tests`, `recompile_all_scripts`, `grep`, `read_console`, `get_scripting_reference` |

Descriptions can reference retired tool names. `grep` above is a Rive MCP tool,
not a recommendation to replace Hermes `search_files` with a shell command.

`capture_artboard` renders PNG for review: explicit artboard ID/name, `longEdge`
(default 512, documented clamp 64–1536), optional `backgroundColor`. Transparent
pixels mean nothing drawn, not black. Query the live schema before use. Inspect
the returned image with `vision_analyze`; a discovered capture API is not proof
that any artboard has rendered. The guarded CLI treats this new render operation
as mutation-class pending execution-side-effect review.

## Failure diagnosis

- Closed port: establish whether the editor is running; do not mistake a QuickLook
  extension for the editor. TCP reachability alone does not establish MCP health.
- Handshake error: use the SDK, including initialize/initialized and transport
  session headers; do not fix protocol order by changing account settings.
- HTTP 200 with tool errors: inspect JSON-RPC, `isError`, structured-content and
  text-encoded JSON envelopes. Partial errors invalidate the affected writes.
- Timed-out mutation: outcome may be unknown. Inspect the editor before any
  manual retry; never replay automatically.
- View-only revision or permission/export denial: stop the denied operation.
  Resolve actual permission or deliver only the distinct permitted output with
  a clear limitation; do not promise editable source from runtime-only output.
- Repeated background warnings: inspect the installed Hermes lifecycle and
  desired user behavior separately. Do not disable the server or add autostart
  simply to hide warnings.

Removal is `hermes config unset mcp_servers.rive` only when requested, followed
by exact-entry readback. It is not app uninstall or permission to remove user data.
