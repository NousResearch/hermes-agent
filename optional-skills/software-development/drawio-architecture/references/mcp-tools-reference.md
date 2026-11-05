# MCP Tools Reference

The official draw.io MCP server (`npx @drawio/mcp`) exposes these tools. Mention
the tool name explicitly in prompts so the agent uses MCP rather than writing a
file by hand (*"Always use the draw.io MCP tools to create diagrams."*).

## `open_drawio_xml`

Opens the draw.io editor with native draw.io/mxGraph **XML**. This is the main
tool for architecture diagrams.

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `content` | string | Yes | Full `<mxfile>…</mxfile>` XML |
| `lightbox` | boolean | No | Read-only view (default `false`) |
| `dark` | string | No | `"auto"` / `"true"` / `"false"` (default `"auto"`) |
| `routing` | string | No | `"libavoid"` reroutes connectors around shapes (obstacle-avoiding orthogonal routing) before opening |

**Layout passes (MCP only):**
- `routing: "libavoid"` — keeps your node positions; re-routes edges around shapes.
- `postLayout: "elk"` — full ELK re-layout; add `direction: "horizontal"` for LR.
  Do **not** combine both.

## `open_drawio_mermaid`

Opens the editor with a **Mermaid.js** diagram. Best for standard typed
diagrams (flowchart, sequence, class, state, ER, gantt, mindmap, sankey…) with
no custom styling/icon needs.

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `content` | string | Yes | Mermaid.js syntax |
| `lightbox` | boolean | No | Read-only view |
| `dark` | string | No | `"auto"` / `"true"` / `"false"` |

Complex Mermaid flowcharts (≥ ~20 nodes, ≥ 3 diamonds, feedback edges, or ≥ 3
endpoints) need `postLayout: "elk"`.

## `open_drawio_csv`

Opens the editor with **CSV** converted to a diagram (org charts, tables).

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `content` | string | Yes | CSV content |
| `lightbox` | boolean | No | Read-only view |
| `dark` | string | No | `"auto"` / `"true"` / `"false"` |

## `search_shapes`

Searches the ~10,000-shape draw.io library (AWS, Azure, GCP, Cisco,
Kubernetes, P&ID, electrical, BPMN…) and returns ready-to-use `style` strings
for `open_drawio_xml`. Supplemented by the draw.io icon service for brand logos.

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `query` | string | Yes | Space-separated keywords, e.g. `aws lambda`, `cisco router` |
| `limit` | number | No | Max results (default 10, max 50) |

> Use `search_shapes` only for diagrams needing industry-specific/branded icons.
> Skip it for flowcharts/UML/ERD/org charts/mind maps.

## `list_pages` / `get_page` / `set_page`

Page-level access to a local multi-page `.drawio`/`.xml` file. Pages are
addressed by 0-based index, exact name, or id.

| Tool | Params | Result |
|------|--------|--------|
| `list_pages` | `path` | `[{index, id, name, approxSizeBytes}]` |
| `get_page` | `path`, `page` | The page's `<mxGraphModel>` XML |
| `set_page` | `path`, `page`, `content` | Replaces that page (a single `<mxGraphModel>`); other pages untouched |

## Local helper tools (this repo)

These are **not** MCP tools — they're Python scripts you run:

- `scripts/setup_drawio_mcp.py` — detect platform + write MCP config / generate
  browser viewer URL.
- `scripts/validate_drawio.py` — XML lint, readability score, PNG `-e` IEND
  repair.
