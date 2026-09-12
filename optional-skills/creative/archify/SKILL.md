---
name: archify
description: Validated interactive HTML diagrams, upstream-maintained.
version: 2.17.0
author: tt-a1i
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [diagram, architecture, workflow, sequence, dataflow, state-machine, mermaid, html, svg]
    category: creative
    related_skills: [architecture-diagram, excalidraw, concept-diagrams]
    upstream:
      repo: tt-a1i/archify
      path: archify
---

# Archify (upstream-maintained)

> **Catalog stub.** This entry is maintained upstream at
> [tt-a1i/archify](https://github.com/tt-a1i/archify): the project ships a
> self-contained skill directory (`archify/`) with the Node CLI, schemas,
> renderers, examples and references. `hermes skills install
> official/creative/archify` pulls the current tree live from that repo
> (quarantined and scanned like any hub install) — this directory holds only
> the catalog metadata, so the vendored copy can never go stale.

Archify turns a small typed JSON spec into a self-contained, explorable HTML
diagram: `architecture`, `workflow`, `sequence`, `dataflow` and `lifecycle`
(state machine) types, dark/light themes, pan/zoom, search, relationship
tracing, optional trace motion, and PNG/JPEG/WebP/SVG/WebM export. It accepts
plain-language requirements or pasted Mermaid (`flowchart`, `sequenceDiagram`,
`stateDiagram`) and can read repository evidence when the diagram must reflect
real code. Every candidate goes through a 9-check `validate` / `deliver`
receipt, so the output is verifiable rather than eyeballed.

## Prerequisites

- Node.js 18+ on `PATH` (`node bin/archify.mjs doctor` confirms the install;
  no `npm install` is needed inside the skill package).
- Installs pull ~200 files (~8 MB, mostly upstream tests and examples) from
  GitHub; the fetch is pinned to one tree SHA, recorded in the bundle metadata.
- Upstream's "Update awareness" step runs `scripts/check-update.mjs`, which
  reads a release manifest from GitHub and only prints a notice — it never
  downloads or installs anything. Skip it if outbound calls are unwanted.

After install, the bundled `architecture-diagram` skill remains the
zero-dependency fallback (it ports the same Cocoon AI lineage archify grew out
of); prefer archify when the user wants validated receipts, Mermaid conversion,
sequence/lifecycle types, or export beyond a single HTML file.

Full documentation: https://github.com/tt-a1i/archify#readme
