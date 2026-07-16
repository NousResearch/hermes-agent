---
name: canvas
description: Create, update, render, and remove durable Hermes Desktop Canvas reports.
---

# Hermes Desktop Canvas

A Canvas is a durable report pair: `<id>.canvas.json` is its structured state and `<id>.canvas.html` is its optional rendered companion. They live under `~/.hermes/profiles/<profile>/` and are not generic artifacts.

Before asking for clarification, list existing `*.canvas.json` files for the active profile. Reuse the matching Canvas; never replace it with an unrelated HTML artifact.

Keep execution quiet: inspect once, write once, validate once. Do not narrate reasoning or tool steps.

The JSON preserves `id`, `title`, `intent`, `source`, `updatedAt`, raw data, and a free-form `document`. The document contains sections with generic elements only: `text`, `kpi`, `table`, `list`, `chart`, `image`, `divider`, and `callout`. Charts declare type (`bar`, `line`, `area`, `pie`, `donut`), labels, and series. Choose structure from the user's intent; do not require domain-specific fields.

For updates, read the existing manifest, refresh only declared sources, revise data and document as needed, set `updatedAt`, validate JSON, and atomically replace the manifest. Update the `.canvas.html` companion only when it already exists or the user requests it. Do not claim success unless the write completed.

Final success evidence:

`CANVAS_WRITE_OK path=<path> updatedAt=<ISO-8601> sha256=<64-hex>`

On failure, leave the existing manifest intact and respond:

`CANVAS_WRITE_FAILED reason=<reason>`
