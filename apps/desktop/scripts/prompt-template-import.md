# Prompt template inbox (Desktop)

Running Desktop watches `$HERMES_HOME/prompt-templates-inbox/*.json` and merges
jobs into the live `hermes.desktop.prompt-templates` store (no rebuild).

## Job shape

```json
{
  "v": 1,
  "op": "merge",
  "folder": "Group",
  "templates": [
    { "label": "Name", "description": "", "text": "…prompt body…" }
  ]
}
```

- `op`: `merge` (default) or `replace` (wipe store first; `templates: []` clears).
- Exact duplicate **text** (normalized) is skipped; near-duplicate labels/text get a mark suffix for review.
- CLI helper: `apps/desktop/scripts/prompt-template-import`.

Depends on prompt-templates store (folders + CRUD).
