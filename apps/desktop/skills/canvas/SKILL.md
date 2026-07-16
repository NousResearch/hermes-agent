---
name: canvas
description: MUST be used whenever the user says Canvas or Canvases to create, update, render, list, or remove durable Hermes Desktop Canvas reports.
---

# Hermes Desktop Canvas

A Canvas is a durable report pair: `<id>.canvas.json` is its required structured state and `<id>.canvas.html` is its optional rendered companion. They live under `~/.hermes/profiles/<profile>/` and are not generic artifacts.

Hard requirement: write and validate `<id>.canvas.json` first. A standalone `.canvas.html` is a failed Canvas operation and must never be reported as success. Do not delegate Canvas creation to a generic HTML, artifact, dashboard, or web-design skill.

Before asking for clarification, list existing `*.canvas.json` files for the active profile. Reuse the matching Canvas; never replace it with an unrelated HTML artifact.

Keep execution quiet: inspect once, write once, validate once. Do not narrate reasoning or tool steps.

The JSON preserves `schema`, `id`, `title`, `profile`, `intent`, `source`, `updatedAt`, raw `data`, and a free-form `document`. The document is the report page: design its sections, hierarchy, and elements from the user's request. Use any composition of `text`, `kpi`, `table`, `list`, `chart`, `image`, `divider`, and `callout`. Charts declare type (`bar`, `line`, `area`, `pie`, `donut`), labels, and series. Do not use domain templates, fixed report blocks, or empty placeholder sections.

Minimum valid shape:

```json
{
  "schema": "hermes.canvas/v1",
  "id": "stable-slug",
  "title": "Human title",
  "profile": "active-profile",
  "intent": "What this Canvas communicates",
  "source": { "prompt": "Original user request", "instructions": "Refresh rules" },
  "updatedAt": "ISO-8601",
  "data": {},
  "document": { "sections": [{ "title": "Chosen section", "elements": ["chosen visual elements"] }] }
}
```

Keep every remote manifest below 460,000 UTF-8 bytes. For a large source, choose the useful summaries and a bounded relevant table/list; preserve the full-data retrieval instructions in `source.instructions`. Never embed an unbounded database dump in a Canvas manifest.

For creation, choose the Canvas structure yourself from the request, write the required JSON to `~/.hermes/profiles/<active-profile>/<id>.canvas.json`, parse it back, measure its byte size, and verify the exact final path exists. For updates, read the existing manifest, keep its identity and sources, but freely revise both `data` and `document` when the user changes the requested content or design. Update the `.canvas.html` companion only when it already exists or the user requests it. Do not claim success unless the required JSON write completed.

Final success evidence:

`CANVAS_WRITE_OK path=<path> updatedAt=<ISO-8601> sha256=<64-hex>`

On failure, leave the existing manifest intact and respond:

`CANVAS_WRITE_FAILED reason=<reason>`
