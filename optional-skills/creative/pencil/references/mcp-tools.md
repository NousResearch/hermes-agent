# Pencil MCP tools reference

Pencil exposes the same tool surface through `pencil interactive` / a headless
`pencil start` session that its IDE/desktop MCP server exposes to other AI
hosts. Call them as `tool_name({ ...args })`. The file path is injected
automatically in headless mode — never pass it yourself.

> **This list drifts — discover the current surface at runtime.** Pencil's
> tools and `batch_design` schema change between releases, so treat this file as
> a primer, not a spec. The authoritative sources, in order:
> 1. `pencil interactive --help` — prints the live tool reference (names +
>    params) for the installed build.
> 2. `get_editor_state({ include_schema: true })` — returns the live document
>    schema **and** the full `batch_design` operation DSL.
> If anything here disagrees with those, they win.

> **Auth required:** the shell will not start without a `pencil login` session
> or `PENCIL_CLI_KEY` — there is no offline mode, even headless.

## Read / inspect

| Tool | Purpose |
| --- | --- |
| `get_editor_state({ include_schema: true })` | Document metadata + structure + the batch_design DSL schema. Call this first. |
| `batch_get({ patterns?, nodeIds?, parentId?, readDepth?, searchDepth?, resolveVariables? })` | Search and read nodes. No args → top-level children. `patterns: [{ reusable: true }]` → list design-system components. Keep `readDepth` ≤ 3 to avoid huge output. |
| `get_variables()` | Read design tokens / theme values (for syncing with CSS). |
| `snapshot_layout()` | Document structure with computed bounds (find overlaps / positioning issues). |

## Mutate

| Tool | Purpose |
| --- | --- |
| `batch_design({ operations })` | The workhorse: insert / update / replace / move / copy / delete nodes, set variables, and generate images. `operations` is a compact string DSL (see below). |

`batch_design` operation DSL (confirmed forms — verify the rest via
`get_editor_state`):

- **Insert:** `name=I(parent,{ ...props })` — inserts a node under `parent`
  (use `document` for top level) and binds the new node id to `name`. Example:
  `hero=I(document,{type:"frame",name:"Hero",x:0,y:0,width:1440,height:900,fill:"#0A0A0A"})`
  **Binding names are ephemeral — they only exist within this one
  `batch_design` call.** In a later call, reference the node by its real id
  (from the call output or `batch_get`), not by the binding name.
- **Generate image into a node:**
  - `G(nodeId,"ai",prompt)` — AI-generated image from a text prompt.
  - `G(nodeId,"stock",keywords)` — stock photo (Unsplash).
- **Set variables / themes:** written via a `SetVariables` operation inside
  `batch_design` (some CLI versions also expose a top-level `set_variables`
  tool for two-way CSS-token sync) — confirm the exact shape via
  `get_editor_state`.

Other operations referenced by Pencil's docs — copy, update, replace, move,
delete — follow the same `OP(...)` shape; read their exact letters/args from
the `get_editor_state` schema rather than guessing.

## Visual / export

| Tool | Purpose |
| --- | --- |
| `get_screenshot({ nodeId })` | Render a node to a PNG (verify visual output). Save it, then read it back with Hermes's `vision_analyze`/`read_file` to check the result. |
| `export_nodes({ ... })` | Export nodes to PNG / JPEG / WEBP / PDF. |

## Style / guidelines

| Tool | Purpose |
| --- | --- |
| `get_guidelines()` | List available guides/styles for working with `.pen` files. |
| `get_guidelines({ category: "guide", name: "Landing Page" })` | Load a specific guide before generating that kind of layout. |

## Session lifecycle (REPL)

- `save()` — write the document to the `--out` path (headless mode only; app
  mode applies changes live).
- `exit()` — leave the shell.

The `pencil_repl.py` wrapper appends `save()` (headless) and `exit()`
automatically, so you only supply the tool-call lines.
