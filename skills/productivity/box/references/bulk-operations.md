# Bulk Operations

Use when moving, creating, or tagging **more than a handful** of items. Read `references/auth-and-setup.md` if the acting identity is unclear.

## Constraints

- **Serial CLI** — one `box` command at a time; parallel CLI breaks auth.
- **Rate limits** — `429` responses include `Retry-After`; wait and retry.
- **Name uniqueness** — duplicate folder names in the same parent return `409`; look up existing folder ID instead of failing.

## Workflow

```
Inventory → Classify (optional) → Plan → Execute (serial) → Verify
```

Skip classify when sorting by filename, extension, or existing metadata alone.

## Step 1 — Inventory

```bash
box folders:items <FOLDER_ID> --json --max-items 1000 --fields id,name,type
```

Paginate until all items captured. Record `id`, `name`, `type` for each.

## Step 2 — Classify (optional)

Preference order:

1. Box AI ask/extract (server-side)
2. Existing metadata
3. Filename/extension rules
4. Download + local analysis (last resort)

Sample-first: classify 5–10 files, derive category set, use cheap rules for the rest. Pace AI calls 1–2s apart.

```bash
box ai:ask --items=id=<FILE_ID>,type=file \
  --prompt "Document type? One of: invoice, receipt, contract, report, other." \
  --json --no-color
```

## Step 3 — Plan

Map each file ID → target folder path. Note which folders exist vs need creation. Confirm with user before large or ambiguous batches.

## Step 4 — Create folders (serial)

```bash
box folders:create <PARENT_ID> "Category" --json --fields id,name
```

Create parents before children. On `409`, list parent to reuse existing folder ID.

## Step 5 — Move files (serial)

```bash
box files:move <FILE_ID> <TARGET_FOLDER_ID> --json
```

Log successes and failures; continue on individual errors. Pause 200–500ms between ops for large batches.

## Step 6 — Verify

List each target folder; confirm counts and IDs. List source folder for stragglers.

```bash
box folders:items <TARGET_FOLDER_ID> --json --max-items 1000 --fields id,name
```

## Rate limits

On `429`: read `Retry-After`, wait, retry the same request. Do not blast parallel requests during cooldown.

## Recovery

Keep a completed-ID log. Retry pass = inventory minus completed. Re-moving to the same parent is safe.

## CLI vs REST

Default to CLI for agent bulk work. Use `references/rest-api.md` only when CLI is unavailable and user confirms.
