---
name: drawio-architecture
description: "Create/edit draw.io architecture diagrams and exports."
license: MIT
version: 1.0.2
author: afonsoft
platforms: [linux, macos, windows]
compatibility: MCP mode needs Node.js + `npx @drawio/mcp` (or a self-hosted draw.io
  instance via DRAWIO_BASE_URL). Local export mode needs the draw.io desktop CLI on
  PATH. Works on macOS/Linux/Windows.
metadata:
  hermes:
    tags: [Diagrams, Drawio, Architecture]
  version: 1.0.2
  visibility: public
  author: merged from Agents365-ai/drawio-skill, scarr05/claude-skills-pub, jgraph/drawio-mcp
  url: https://github.com/afonsoft/skills
  homepage: https://github.com/jgraph/drawio-mcp
  sources: https://github.com/Agents365-ai/drawio-skill, https://github.com/scarr05/claude-skills-pub,
    https://www.drawio.com/docs/manual/generate/drawio-mcp-server/, https://dev.to/rushier/how-to-use-claude-ai-drawio-to-create-architecture-diagrams-for-projects-17i1,
    https://github.com/jgraph/drawio-mcp
  openclaw: '{"requires":{"bins":["npx"]},"envVars":[{"name":"DRAWIO_BASE_URL","required":false,"description":"Optional
    self-hosted draw.io base URL."}]}'
---

# Draw.io Architecture Diagrams + MCP Integration

Create professional, editable architecture diagrams in draw.io's native `.drawio` XML format **and** integrate the official draw.io MCP server so diagrams open directly in the editor. This skill merges two proven sources:

1. **Authoring knowledge** — how to write correct `.drawio` XML for architecture, network, cloud, flowchart and ER diagrams (from `Agents365-ai/drawio-skill` and `scarr05/claude-skills-pub`).
2. **MCP integration** — how to configure and call the official `@drawio/mcp` server so the agent opens diagrams inline/editor instead of dumping XML to a file (from `jgraph/drawio-mcp`, the vendor's own reference).

## Security and Trust Boundaries

- **Pin the MCP server version**: `npx` can download and execute remote code. Do not run bare `npx -y @drawio/mcp`. Use `npx -y @drawio/mcp@<VERSION>` with an explicit version verified on [npm](https://www.npmjs.com/package/@drawio/mcp) or the project's lockfile. Verify the package name and publisher (JGraph / `drawio`) before installing.
- **Prefer local stdio or self-hosted**: The recommended integration runs `@drawio/mcp` as a local stdio process. If you use `DRAWIO_BASE_URL`, point it to a draw.io instance you control and trust.
- **Hosted endpoint caution**: `https://mcp.draw.io/mcp` is a remote MCP Apps endpoint operated by the draw.io vendor. It receives diagram XML and renders inline. Only use it when the vendor, TLS channel, and data sensitivity are acceptable for your diagrams. Do not send confidential or regulated architecture data to the hosted endpoint.
- **SVG/PNG exports are local**: The CLI export path (`drawio -x ...`) runs the desktop application locally and does not upload diagrams unless you explicitly open a browser URL.

## When to use

- The user asks for an **architecture / system / service / network / cloud diagram**, a **flowchart**, **ER diagram**, **sequence / UML class**, **C4 model**, or any draw.io/diagrams.net visualization.
- The user wants diagrams to **open in draw.io automatically** (MCP) rather than be saved as a file for manual import.
- You are setting up the **draw.io MCP server** for the current agent/IDE and need the exact config block per platform.
- You want the agent to **search the official shape library** (AWS/Azure/GCP/Cisco/K8s/brand logos) without guessing `shape=` strings.

- User asks or mentions this skill in English (e.g., "use /drawio-architecture", "run drawio-architecture").
- O usuário pede ou menciona esta skill em português (ex.: "use /drawio-architecture", "execute drawio-architecture").

## When NOT to use

- A casual hand-drawn / whiteboard look → **excalidraw** or **tldraw**.
- Diagrams-as-code that live in git and render in Markdown → use **mermaid-architecture** (general) or **plantuml** (UML).
- Freeform infinite-canvas sketching → **tldraw**.

## Two delivery paths (pick one, or combine)

| Path | What it does | When to use |
|------|--------------|-------------|
| **A. MCP (recommended for chat agents)** | Agent calls an MCP tool (`open_drawio_xml`) → draw.io editor opens in a browser tab with the diagram. No file on disk required. | Agent is an MCP client (Claude Desktop/Code, VS Code Copilot, Cursor, OpenCode, Windsurf). Best for "show me the diagram" flows. |
| **B. Local CLI** | Agent writes a `.drawio` file, then `drawio -x -f png ...` exports PNG/SVG/PDF locally. | Headless/CI, need image deliverables, or no MCP available. Requires draw.io desktop CLI. |

Both paths share the **same XML authoring rules** in this skill — only the delivery step differs. You can generate XML with the rules below and feed it to *either* `open_drawio_xml` (path A) *or* a local file + CLI export (path B).

---

# PATH A — MCP server: configure + use

## A.1 Configure the server

The server is distributed as `npx @drawio/mcp` (stdio). Pin an explicit version and run it locally:

```bash
npx -y @drawio/mcp@<VERSION>
```

Replace `<VERSION>` with the latest stable release verified on [npm](https://www.npmjs.com/package/@drawio/mcp). Do not run bare `npx -y @drawio/mcp` because it resolves to the latest remote version at runtime.

Add it to your client's MCP config under `mcpServers.drawio`. For the concrete JSON block per platform plus self-hosting, see **`references/mcp-config.md`** (Claude Desktop, Claude Code, VS Code `.vscode/mcp.json`, Cursor `~/.cursor/mcp.json`, OpenCode, Windsurf, and the `DRAWIO_BASE_URL` env for self-hosted instances).

There is also a **hosted** alternative (`https://mcp.draw.io/mcp`) that renders diagrams *inline* via the MCP Apps protocol (Claude.ai, VS Code, Cursor) — no install, but it is a *different* server type than the stdio one above and sends your diagram XML to the draw.io vendor's servers. Only use it for non-sensitive diagrams and when you trust the vendor endpoint.

### Automated setup helper

Run the bundled helper to detect the agent/platform and append the correct config automatically:

```bash
python3 scripts/setup_drawio_mcp.py --detect
python3 scripts/setup_drawio_mcp.py --target claude-code --dry-run
python3 scripts/setup_drawio_mcp.py --target vscode --global
```

See **`scripts/setup_drawio_mcp.py`** for all `--target` values (`claude-desktop`, `claude-code`, `vscode`, `cursor`, `opencode`, `windsurf`, `raw`) and flags (`--dry-run`, `--global`, `--force`).

## A.2 MCP tools reference

The server exposes these tools. Mention the tool name explicitly in prompts so the agent uses MCP rather than hand-writing a file ("Always use the draw.io MCP tools to create diagrams.").

### `open_drawio_xml` — the main one for architecture diagrams

Opens the editor with native draw.io/mxGraph **XML**. Parameters:

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `content` | string | Yes | Full `<mxfile>…</mxfile>` XML (author it per this skill) |
| `lightbox` | boolean | No | Read-only view (default `false`) |
| `dark` | string | No | `"auto"` / `"true"` / `"false"` (default `"auto"`) |
| `routing` | string | No | `"libavoid"` reroutes connectors around shapes (obstacle-avoiding orthogonal routing) before opening |

### `open_drawio_mermaid`

Opens the editor with a **Mermaid.js** diagram. Use for standard typed diagrams (flowchart, sequence, class, state, ER, gantt, mindmap, sankey…) where you don't need custom styling/icons — write Mermaid text, let draw.io render it editable. Params: `content` (yes), `lightbox`, `dark`.

### `open_drawio_csv`

Opens the editor with **CSV** converted to a diagram (org charts, tables). Params: `content` (yes), `lightbox`, `dark`.

### `search_shapes`

Searches the ~10,000-shape draw.io library (AWS, Azure, GCP, Cisco, Kubernetes, P&ID, electrical, BPMN…) and returns ready-to-use `style` strings for `open_drawio_xml`. Supplemented by the draw.io icon service for brand logos (`react`, `slack`, `shopping cart`). Params: `query` (space-separated keywords, yes), `limit` (default 10, max 50).

> Use `search_shapes` only for diagrams needing industry-specific/branded icons (cloud architecture, network topology, P&ID, K8s, BPMN specifics, brand logos). Skip it for flowcharts/UML/ERD/org charts/mind maps that use basic shapes.

### `list_pages` / `get_page` / `set_page`

Page-level access to a local multi-page `.drawio`/`.xml` file. Address pages by 0-based index, exact name, or id.

| Tool | Params | Result |
|------|--------|--------|
| `list_pages` | `path` | `[{index, id, name, approxSizeBytes}]` |
| `get_page` | `path`, `page` | The page's `<mxGraphModel>` XML |
| `set_page` | `path`, `page`, `content` | Replaces that page (a single `<mxGraphModel>`); other pages untouched |

## A.3 Layout & routing passes (MCP `open_drawio_xml`)

You declare **logical structure** (nodes, edges, labels, containers). draw.io's router + optional post-layout handle placement. Two opt-in passes on `open_drawio_xml`:

- **`routing: "libavoid"`** — keeps your node positions, re-routes **edges** orthogonally *around* shapes. Use for deliberately-laid-out architecture/network/deployment/swimlane diagrams where wires shouldn't cut through boxes.
- **`postLayout: "elk"`** — **full re-layout** (ELK `layered`); nodes are re-placed and edges routed. Best for directional/hierarchical flows (flowcharts, pipelines, decision flows). Set `direction: "horizontal"` for left-to-right. Do **not** combine with `routing` — pick one.

For Mermaid: complex flowcharts (≥ ~20 nodes, ≥ 3 diamonds, feedback edges, or ≥ 3 endpoints) need `postLayout: "elk"`; simple flowcharts and all non-flowchart Mermaid types need none.

---

# PATH B — Local CLI export (fallback / deliverables)

Resolve the binary name first (`drawio` is canonical on Homebrew/Linux `.deb`/`.rpm`/AUR; `draw.io` on older builds; full path on macOS `.app`/Windows `.exe`). Store every diagram artifact under `docs/architecture/`:

```bash
# Ensure the architecture docs folder exists
mkdir -p docs/architecture

# Preview PNG (NO -e; required for vision self-check; width-capped under 2576px)
drawio -x -f png --width 2000 -o docs/architecture/diagram.png docs/architecture/diagram.drawio

# Final PNG (WITH -e; double extension keeps it editable; run repair_png after)
drawio -x -f png -e -s 2 -o docs/architecture/diagram.drawio.png docs/architecture/diagram.drawio

# SVG / PDF (final, -e safe)
drawio -x -f svg -e --embed-svg-images -o docs/architecture/diagram.svg docs/architecture/diagram.drawio
drawio -x -f pdf -e -o docs/architecture/diagram.pdf docs/architecture/diagram.drawio
```

After every `-e` PNG export, fix draw.io's truncated IEND chunk:
```bash
python3 scripts/validate_drawio.py docs/architecture/diagram.drawio.png --repair-iend
```

If the CLI is unavailable, fall back to a browser URL (no upload — XML lives in the `#` fragment):
```bash
python3 scripts/setup_drawio_mcp.py --viewer-url docs/architecture/diagram.drawio        # read-only
python3 scripts/setup_drawio_mcp.py --viewer-url --edit docs/architecture/diagram.drawio # editable editor URL
```

> See `references/mcp-config.md` for the full Linux headless (`xvfb-run`, `--no-sandbox`, `--disable-gpu`, `HOME`) guidance and the fallback chain.

---

# AUTHORING — `.drawio` XML rules (shared by both paths)

These rules come from the vendor's own `shared/xml-reference.md` (the single source of truth for MCP prompts) plus the Agents365/claude-skills-pub authoring guides. Follow them whether the XML goes to `open_drawio_xml` or a local file.

## File skeleton

```xml
<mxfile host="app.diagrams.net" modified="2026-01-01T00:00:00" agent="agent" version="24.0.0" type="device">
  <diagram name="Page-1" id="page1">
    <mxGraphModel dx="1200" dy="800" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1100" pageHeight="850" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
```

**Rules:** `id="0"` and `id="1"` are required root cells — never omit. User shapes start at `id="2"` with unique ids. Top-level shapes use `parent="1"`. Set `type="device"` when the file is meant to be opened from disk.

## Rigid grid (use for every diagram)

- Column `x = col*180 + 40`  (col 0 = 40, col 1 = 220, …)
- Row `y = row*120 + 40`     (row 0 = 40, row 1 = 160, …)
- Node sizes: rectangle `140×60`, diamond `140×80`, circle `60×60`, document `120×80`, cylinder `100×70`

Place each node at a `(col,row)`; the router handles spacing. **Do not** hand-add `<Array as="points">` waypoints or `exitX/entryY` overrides unless you have specific geometric intent.

## Core shapes (vertex)

```xml
<!-- Rounded rectangle — services, modules -->
<mxCell id="2" value="Label" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;" vertex="1" parent="1">
  <mxGeometry x="100" y="100" width="140" height="60" as="geometry"/>
</mxCell>

<!-- Diamond — decision -->
<mxCell id="3" value="Condition?" style="rhombus;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;" vertex="1" parent="1">
  <mxGeometry x="100" y="200" width="140" height="80" as="geometry"/>
</mxCell>

<!-- Cylinder — database -->
<mxCell id="4" value="DB" style="shape=cylinder3;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#666666;" vertex="1" parent="1">
  <mxGeometry x="350" y="100" width="100" height="70" as="geometry"/>
</mxCell>
```

## Edges (connectors)

**CRITICAL:** every edge `mxCell` needs an expanded `<mxGeometry relative="1" as="geometry" />` child. Self-closing edge cells do **not** render.

```xml
<mxCell id="e1" value="HTTP" style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;" edge="1" parent="1" source="2" target="3">
  <mxGeometry relative="1" as="geometry"/>
</mxCell>
```

- **Do not hand-route** — just declare `source`/`target`. For clean orthogonal wires around boxes, set `routing:"libavoid"` (MCP) or apply the same spirit in CLI files.
- **Consistent edge style per diagram:** ER → `entityRelationEdgeStyle`; UML class → straight (no `edgeStyle`); mind maps → `curved=1`; flowchart/architecture/network → `orthogonalEdgeStyle`.
- Keep edge labels short (1–3 words: `Yes`, `async`, `reads`). Push longer detail into node text or a legend node.
- Animated data-flow: add `flowAnimation=1;`.

## Containers & nested architecture

Use real parent-child containment (not shapes placed on top of bigger shapes).

| Type | Style | When |
|------|-------|------|
| Group (invisible) | `group;pointerEvents=0;` | No border, no connections |
| Swimlane (titled) | `swimlane;startSize=30;` | Visible title bar, or container itself connects |
| Custom container | add `container=1;pointerEvents=0;` | Any shape as container |

```xml
<mxCell id="vpc" value="VPC" style="swimlane;startSize=24;fillColor=#dae8fc;strokeColor=#6c8ebf;html=1;" vertex="1" parent="1">
  <mxGeometry x="0" y="0" width="720" height="360" as="geometry"/>
</mxCell>
<mxCell id="az1" value="AZ us-east-1a" style="swimlane;startSize=24;fillColor=#fff2cc;strokeColor=#d6b656;html=1;" vertex="1" parent="vpc">
  <mxGeometry x="20" y="36" width="320" height="300" as="geometry"/>
</mxCell>
<mxCell id="web1" value="web-1" style="rounded=1;whiteSpace=wrap;html=1;" vertex="1" parent="az1">
  <mxGeometry x="30" y="40" width="120" height="60" as="geometry"/>
</mxCell>
```

**Rules:** children use coordinates **relative to the parent**. Edges between cells in **different** containers must have `parent="1"` (else they render inside a container and get clipped). Every container gets `pointerEvents=0;` unless it itself must be connectable (then `swimlane` handles it).

## Layered architecture (top → bottom) and left-to-right flow

- **Layered (TB):** Users/Clients → API/Interface → Business Logic → Data/Storage. Each layer = a swimlane or rounded container; components inside; arrows show flow; add a legend; add a metadata footer.
- **Left-to-right (LR):** `Source → Process → Target`. Use `postLayout:"elk"` + `direction:"horizontal"` (MCP) or place columns left-to-right with the rigid grid.

See **`references/architecture-patterns.md`** for full worked XML of: layered service architecture, microservices with an event bus, client/API/DB, and a C4-lite context diagram.

## Color palette (semantic)

Use coordinated fill/stroke pairs. When 3+ roles appear, auto-generate a legend (see `references/style-guide.md`).

| Role | fillColor | strokeColor |
|------|-----------|-------------|
| Service / client | `#dae8fc` | `#6c8ebf` |
| Success / database | `#d5e8d4` | `#82b366` |
| Queue / decision | `#fff2cc` | `#d6b656` |
| Gateway / API | `#ffe6cc` | `#d79b00` |
| Error / alert | `#f8cecc` | `#b85450` |
| External / neutral | `#f5f5f5` | `#666666` |
| Security / auth | `#e1d5e5` | `#9673a6` |

## Cloud provider icons

For AWS/Azure/GCP/Cisco/K8s, prefer official icons. With MCP, call `search_shapes` and paste the returned `style`. Hand-authored cheatsheet (AWS `shape=mxgraph.aws4.resourceIcon;resIcon=…`, Azure `image=img/lib/azure2/…svg`, GCP paths) is in **`references/cloud-icons.md`**.

## HTML labels, dark mode, tags, layers, metadata

- **Always add `html=1`** to every cell style — plain text is unaffected, but HTML (`<b>`, `<br>`, `<font>`) renders only with it.
- Line breaks: `&#xa;` (works with or without `html=1`) or `&lt;br&gt;` (needs `html=1`). Never `\n`.
- XML-escape attribute values: `&amp;` `&lt;` `&gt;` `&quot;`.
- **Dark mode:** set `adaptiveColors="auto"` on `<mxGraphModel>`; `strokeColor/fillColor/fontColor="default"` auto-adapt. Use `light-dark(light,dark)` only when inverse is wrong.
- **Tags** (cross-cutting filters): wrap cell in `<object id=… tags="critical v2">`; `label` replaces `value`.
- **Layers** (toggle visibility): `mxCell parent="0"` with no `vertex`/`edge`.
- **Metadata + placeholders:** `<object … component="X" status="Active" placeholders="1">` with `label="%component% — %status%"`.

## CRITICAL: XML well-formedness

- **NEVER include XML comments (`<!-- -->`)** in diagram output — they waste tokens and can cause parse errors.
- Escape special chars in attribute values; always use unique `id`s.
- Validate before delivery: `python3 scripts/validate_drawio.py docs/architecture/diagram.drawio`.

## Workflow (recommended)

1. **Clarify** (1–3 questions if missing): diagram type, output mode (MCP open vs file+CLI vs image), scope/fidelity, specific technologies.
2. **Configure MCP** if not already (path A.1 / helper) — or resolve the draw.io CLI binary (path B).
3. **Plan** shapes, relationships, layout (LR/TB), grouping (tier/container), icon needs.
4. **Author** the XML with the rules above (or write Mermaid for a standard typed diagram).
5. **Deliver:**
   - **MCP** → call `open_drawio_xml`/`_mermaid`/`_csv`.
   - **CLI** → write the source `.drawio` to `docs/architecture/<diagram-name>.drawio`, then export PNG/SVG/PDF to the same folder, optionally vision self-check.
6. **Iterate** with targeted XML edits (change `fillColor`, move `x/y`, add/remove node/edge) until approved.

## Common mistakes

| Symptom | Fix |
|---------|-----|
| Edge doesn't render | Edge cell is self-closing; add `<mxGeometry relative="1" as="geometry" />` |
| Vision API 400 "Could not process image" | Exported with `-e`; re-export preview **without** `-e`. (`-e` PNG has truncated IEND — run repair) |
| Wire cuts through a box | Set `routing:"libavoid"` (MCP) or add a waypoint / increase spacing |
| Blank box instead of icon | Wrong `shape=mxgraph.*` name — use `search_shapes` / `references/cloud-icons.md` |
| Stacked edges at a node | Distribute `exitX/exitY`/`entryX/entryY` over the side |
| HTML shows as literal text | Missing `html=1` in style |
| Cross-nested container clip | Edges between different containers need `parent="1"` |
| CJK/URL opens with "URI malformed" | Browser fallback must `encodeURIComponent` the XML (helper does this) |

## References (this skill)

- `architecture` — umbrella skill that owns `docs/architecture/` and routes diagram work to this engine (the orchestrator calls `/architecture`, which delegates here for editable `.drawio` deliverables)
- `references/mcp-config.md` — exact MCP config JSON per platform + self-host + headless/CLI fallback + automated setup + verify steps
- `references/mcp-tools-reference.md` — full MCP tools reference with parameters and examples (en) / `mcp-tools-reference.pt-br.md` (pt-BR)
- `references/usage-guide.md` — how to drive the skill once configured (en) / `usage-guide.pt-br.md` (pt-BR)
- `references/usage-examples.md` — end-to-end usage examples (en) / `usage-examples.pt-br.md` (pt-BR)
- `references/architecture-patterns.md` — full worked XML for layered / microservices / client-API-DB / C4-lite
- `references/cloud-icons.md` — AWS / Azure / GCP icon cheatsheet
- `references/style-guide.md` — palette, typography, effects, legend generation
- `scripts/setup_drawio_mcp.py` — detect platform + write MCP config / generate viewer URL
- `scripts/validate_drawio.py` — XML well-formedness + structural lint + IEND repair

## External references (sources analyzed)

- draw.io MCP docs: https://www.drawio.com/docs/manual/generate/drawio-mcp-server/
- Vendor repo (4 integration modes + XML reference): https://github.com/jgraph/drawio-mcp
- MCP tool-server README (config blocks): https://github.com/jgraph/drawio-mcp/blob/main/mcp-tool-server/README.md
- XML reference (source of truth): https://github.com/jgraph/drawio-mcp/blob/main/shared/xml-reference.md
- Style reference: https://github.com/jgraph/drawio-mcp/blob/main/shared/style-reference.md
- Skill (authoring): https://github.com/Agents365-ai/drawio-skill
- Skill (authoring): https://github.com/scarr05/claude-skills-pub
- Workflow article: https://dev.to/rushier/how-to-use-claude-ai-drawio-to-create-architecture-diagrams-for-projects-17i1
