# Search and AI

Run via `terminal`. Prefer server-side Box AI over downloading file bodies.

## Full-text search

```bash
box search "invoice ACME" --json --limit 25
box search "type:pdf contract" --json --limit 25
```

Search returns only content the **current actor** can see. Wrong actor → empty or incomplete results.

Apply filters early (type, ancestor folder, owner) to reduce noise. Return file/folder IDs before downloading.

Docs: https://developer.box.com/guides/search/

## Metadata query

For structured metadata templates (enterprise metadata):

```bash
box metadata-query --help   # confirm flags for your CLI version
```

Use when files are tagged with metadata templates rather than searchable filenames.

## Box AI

Space calls **1–2 seconds apart** to avoid rate limits.

```bash
box ai:ask --help
box ai:extract --help
box ai:text-gen --help
```

Preference order for understanding document content:

1. **Box AI ask/extract** — content stays on Box servers
2. **Metadata** already on the file
3. **Download + local analysis** — only when AI is unavailable or insufficient

Before answering user questions about document content:

1. Search or list to get file IDs
2. Run Box AI on those IDs
3. Cite file IDs in the response

Docs: https://developer.box.com/guides/box-ai/

## Retrieval quality

- Filter by ancestor folder when the user names a location
- Confirm actor can see the folder (collaboration or workspace model)
- Do not download entire folders when search + AI suffices

## Troubleshooting empty search

See `references/troubleshooting.md` — usually wrong actor or content not shared with service account.
