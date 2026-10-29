---
name: diagram-design
description: "Editorial HTML/SVG diagrams, 45 types, upstream-maintained."
version: 2.6.0
author: Cathryn Lavery (cathrynlavery / LittleMight)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [diagram, svg, html, chart, architecture, flowchart, sequence, sankey, gantt, mermaid, drawio, excalidraw]
    category: creative
    related_skills: [archify, architecture-diagram, concept-diagrams, excalidraw]
    upstream:
      repo: cathrynlavery/diagram-design
      path: skills/diagram-design
---

# Diagram Design (upstream-maintained)

> **Catalog stub.** This entry is maintained upstream at
> [cathrynlavery/diagram-design](https://github.com/cathrynlavery/diagram-design):
> the project ships a self-contained skill directory (`skills/diagram-design/`)
> with the SKILL.md, ~60 type and primitive references, HTML templates, ~200
> rendered examples and six import/export scripts. `hermes skills install
> official/creative/diagram-design` pulls the current tree live from that repo
> (quarantined and scanned like any hub install). This directory holds only the
> catalog metadata, so the vendored copy can never go stale; upstream lands
> dozens of commits a week.

Diagram Design draws diagrams as self-contained HTML files with inline SVG and
an editorial design system: one paper/ink/accent palette, a 4/10 density
target, one or two accent nodes per figure, and a pre-output "taste gate"
checklist. It covers 45 visual types: architecture (plus architecture delta and
IT current-state), flowchart, sequence, state machine, ER and database schema,
timeline, swimlane, Gantt, quadrant, radar, polar, loop/flywheel, tree, org
chart, layer stack, exploded and axonometric views, Venn, pyramid/funnel,
treemap, marimekko, heatmap, bar/dumbbell, waterfall, line families
(slopegraph, ridgeline, streamgraph, bump), scatter/bubble/beeswarm, Sankey,
fishbone, Wardley map, kanban, user journey, service blueprint, deployment,
dependency graph, UML class, story map and data-platform layouts. It also
redraws existing `.drawio`, Mermaid and `.excalidraw` files in the same style,
and can save brand profiles (`~/.diagram-design/profiles/`) taken from a
website, a design-system folder or pasted tokens.

## When to prefer it

| Need | Use |
|---|---|
| Charts and business/editorial figures (Sankey, waterfall, Gantt, Wardley, heatmap, slopegraph) or a branded house style across many diagrams | **diagram-design** |
| Validated architecture/workflow/sequence diagrams with receipts, pan/zoom, Mermaid conversion | `archify` |
| One quick dark-themed architecture page, zero dependencies | `architecture-diagram` (bundled) |
| Hand-drawn whiteboard look | `excalidraw` |

## Prerequisites

- Nothing for the core path: the agent writes HTML/SVG from the templates.
- `uv run python3` (stdlib only) for the import and export scripts:
  `drawio_extract.py`, `mermaid_extract.py`, `excalidraw_extract.py`,
  `export_svg.py` and `self_check.py`.
- PNG export (`scripts/export_png.py`) needs Playwright with Chromium
  (`uv run --with playwright python3 -m playwright install chromium`).
- Generated pages load Instrument Serif and other fonts from Google Fonts when
  a browser opens them, and PNG export makes the same request. Upstream's
  privacy policy states the skill itself has no server, analytics or telemetry.
- Installs pull ~290 files (~5 MB, mostly rendered example pages) from GitHub.
  Set `GITHUB_TOKEN` or sign in with the `gh` CLI first if the anonymous API
  limit is already spent.

## Running it in Hermes

- On the first diagram in a project, upstream pauses to ask whether to brand
  the style guide (website URL, design-system folder, pasted tokens, saved
  profile, or keep defaults). In cron or other headless runs, keep defaults and
  say so in the reply.
- Upstream's `commands/` (`/doctor`, `/export-diagram`, `/import-*`,
  `/profile`) are not part of the installed skill. Ask for the same action in
  plain words; the matching `references/*.md` file carries the procedure.
- Open the result with `browser_navigate` on the `file://` path and check it
  with `browser_vision` before delivering.

Full documentation: https://github.com/cathrynlavery/diagram-design#readme
