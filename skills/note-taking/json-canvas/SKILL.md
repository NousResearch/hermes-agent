---
name: json-canvas
description: Create and edit .canvas files with nodes and edges.
platforms: [linux, macos, windows]
version: 1.1.0
author: Harish Kukreja (counterposition), Hermes Agent (adapted from Steph Ango's official Obsidian skills)
license: MIT
metadata:
  hermes:
    tags: [obsidian, json-canvas, canvas, note-taking, diagrams]
    related_skills: [obsidian, obsidian-markdown, obsidian-cli, obsidian-bases]
---

# JSON Canvas Skill

Create and edit JSON Canvas files (`.canvas`) — the open format behind
Obsidian's Canvas: mind maps, flowcharts, and visual boards built from
nodes, edges, and groups per the
[JSON Canvas Spec 1.0](https://jsoncanvas.org/spec/1.0/). Adapted from
Steph Ango's MIT-licensed official Obsidian skills.

## When to Use

- Creating or editing `.canvas` files
- The user asks for a visual canvas, mind map, flowchart, or project board
  in Obsidian

Route elsewhere for broad vault work (`obsidian`), Markdown notes
(`obsidian-markdown`), or `.base` database views (`obsidian-bases`).

## Prerequisites

None — `.canvas` files are plain JSON and no running Obsidian app is
required. Resolve the vault path first (see the `obsidian` skill).

## How to Run

Read canvases with `read_file`, create them with `write_file`, and make
targeted edits with `patch`. After any change, validate with the bundled
checker (run via `terminal`, from this skill's directory):

```bash
python scripts/validate_canvas.py /absolute/path/to/file.canvas
```

It checks JSON validity, ID uniqueness, edge reference integrity, required
per-type fields, and enum values, and exits non-zero with per-error
messages on failure.

## Quick Reference

A `.canvas` file is a JSON object with two optional top-level arrays:

```json
{
  "nodes": [],
  "edges": []
}
```

### Generic node attributes

Array order determines z-index: first node = bottom layer.

| Attribute | Required | Type | Description |
|-----------|----------|------|-------------|
| `id` | Yes | string | Unique string (convention: 16-char hex) |
| `type` | Yes | string | `text`, `file`, `link`, or `group` |
| `x`, `y` | Yes | integer | Position in pixels (top-left corner) |
| `width`, `height` | Yes | integer | Size in pixels |
| `color` | No | canvasColor | Preset `"1"`-`"6"` or hex (e.g., `"#FF0000"`) |

### Per-type attributes

| Node type | Required | Optional |
|-----------|----------|----------|
| `text` | `text` (Markdown string) | — |
| `file` | `file` (vault path) | `subpath` (starts with `#`) |
| `link` | `url` | — |
| `group` | — | `label`, `background`, `backgroundStyle` (`cover`/`ratio`/`repeat`) |

```json
{
  "id": "6f0ad84f44ce9c17",
  "type": "text",
  "x": 0,
  "y": 0,
  "width": 400,
  "height": 200,
  "text": "# Hello World\n\nThis is **Markdown** content."
}
```

### Edge attributes

| Attribute | Required | Default | Description |
|-----------|----------|---------|-------------|
| `id` | Yes | - | Unique identifier |
| `fromNode`, `toNode` | Yes | - | Source / target node IDs |
| `fromSide`, `toSide` | No | - | `top`, `right`, `bottom`, or `left` |
| `fromEnd`, `toEnd` | No | `none` / `arrow` | `none` or `arrow` |
| `color` | No | - | Line color (canvasColor) |
| `label` | No | - | Text label |

```json
{
  "id": "0123456789abcdef",
  "fromNode": "6f0ad84f44ce9c17",
  "fromSide": "right",
  "toNode": "a1b2c3d4e5f67890",
  "toSide": "left",
  "label": "leads to"
}
```

### Colors

`canvasColor` is either a hex string (`"#FF0000"`; 3-, 4-, 6-, and 8-digit
forms are accepted) or a preset: `"1"` red, `"2"` orange, `"3"` yellow,
`"4"` green, `"5"` cyan, `"6"` purple. Preset rendering is
application-defined.

### Layout guidelines

- Coordinates can be negative; `x` increases right, `y` increases down
- Any unique string is a valid ID; generate 16-character lowercase hex
  strings (64-bit random) to match Obsidian's convention
- Space nodes 50-100px apart; leave 20-50px padding inside groups
- Align to multiples of 10 or 20 for cleaner layouts
- Suggested sizes: small text 250x100, large text 500x400, file preview
  400x300, link preview 300x150

## Procedure

Creating a new canvas:

1. Start from the base structure `{"nodes": [], "edges": []}`.
2. Generate a unique 16-char hex ID for each node.
3. Add nodes with the required fields (`id`, `type`, `x`, `y`, `width`,
   `height`) plus the per-type required field.
4. Add edges referencing valid node IDs via `fromNode`/`toNode`.
5. Write the file with `write_file`, then validate (see Verification).

Editing an existing canvas:

1. Read the file with `read_file` and locate the target node/edge by `id`.
2. For a new node, pick a position that avoids overlapping existing nodes
   and an ID that collides with nothing.
3. Group nodes visually by adding a `group` node whose bounds enclose them.
4. Apply the change with `patch` (or rewrite with `write_file` when the
   change is structural), then validate.

## Pitfalls

- **Newlines in text nodes**: use `\n` inside the JSON string. A literal
  `\\n` renders as the characters `\` and `n` in Obsidian.
- Duplicate IDs and dangling `fromNode`/`toNode` references break rendering
  silently — always validate after editing.
- `file` node paths are vault-relative; a wrong path shows a broken
  embed only when Obsidian opens the canvas.
- Do not add trailing commas or comments — `.canvas` is strict JSON.

## Verification

Run `python scripts/validate_canvas.py <file.canvas>` via `terminal`; it
must exit 0. It enforces: valid JSON; unique `id` values across nodes and
edges; every edge endpoint resolves to a node; required per-type fields
present; `type`, side, and end values within their enums; colors are
presets `"1"`-`"6"` or hex. For a final visual check, open the canvas in
Obsidian if it is available.

## References

- Complete worked examples (mind map, project board, flowchart):
  [references/EXAMPLES.md](references/EXAMPLES.md)
- [JSON Canvas Spec 1.0](https://jsoncanvas.org/spec/1.0/)
- [JSON Canvas GitHub](https://github.com/obsidianmd/jsoncanvas)
- Source: [kepano/obsidian-skills](https://github.com/kepano/obsidian-skills) by Steph Ango,
  MIT — full license in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)
