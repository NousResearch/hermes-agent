---
name: reladraw
description: Text diagrams with stated placement to SVG, upstream-kept.
version: 0.13.0
author: reladraw contributors
license: Apache-2.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [diagram, architecture, dataflow, pipeline, deployment, svg, text-to-diagram, mermaid-alternative]
    category: creative
    related_skills: [architecture-diagram, excalidraw, archify, concept-diagrams]
    upstream:
      repo: reladraw/reladraw
      path: .claude/skills/reladraw
---

# reladraw (upstream-maintained)

> **Catalog stub.** This entry is maintained upstream at
> [reladraw/reladraw](https://github.com/reladraw/reladraw): the project ships
> its own agent skill directory (`.claude/skills/reladraw/`, a `SKILL.md` plus
> the full syntax reference). `hermes skills install official/creative/reladraw`
> pulls the current tree live from that repo (quarantined and scanned like any
> hub install) — this directory holds only the catalog metadata, so the
> language reference can never lag the CLI you actually run.

reladraw is a text language for node-and-line diagrams where the source states
the arrangement — `right of api`, `below app.ui`, `level with store`,
`between web and worker` — and the compiler works out only the distances.
Mermaid, Graphviz and D2 hand placement to a layout engine; draw.io and
Excalidraw make you pick coordinates. reladraw sits in between: relative
positions you can read back out of the source without looking at the picture,
which is exactly the property an agent that cannot see its own SVG needs.
Anything ambiguous is refused with a line-numbered error instead of guessed.

Source in, standalone SVG out (`reladraw diagram.reladraw -o diagram.svg`;
`-o -` for stdout). A `<reladraw-diagram>` web component renders the same
source inside a page.

## Prerequisites

- Node.js 18+ on `PATH`. `npx -y reladraw --help` works without installing
  (`-y` skips the interactive install prompt in non-TTY agent shells);
  `npm install -g reladraw` installs the command. The package has no runtime
  dependencies (~700 KB installed).
- The upstream skill is `SKILL.md` + `reference/syntax.md` (~120 KB total,
  no scripts); the fetch is pinned to one tree SHA, recorded in the bundle
  metadata.
- Nothing in the skill makes network calls; the CLI reads the `.reladraw`
  file and writes the SVG.

## When to prefer it

- The user has a specific picture in mind (this box under that one, the
  database off to the right) and auto-layout keeps rearranging it.
- A diagram will be edited later by re-reading its source: the arrangement is
  in the text, so a change to one statement moves one thing.
- Not for data charts (bar, line, pie) and not for hand-drawn or interactive
  output — use `excalidraw` for the sketch look, `archify` for validated
  pan/zoom HTML with Mermaid import, and the bundled `architecture-diagram`
  skill for zero-dependency dark-themed SVG-in-HTML.

Full documentation: https://github.com/reladraw/reladraw#readme
