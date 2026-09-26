# Bulk operations

Use this workflow for more than a handful of files. Choose the current OAuth actor before inventorying; it can only process content that identity can access.

## Workflow

```
Inventory → classify if needed → plan → confirm → execute → verify → report
```

## Inventory and plan

```bash
box folders:items <FOLDER_ID> --json --max-items 1000 --fields id,name,type,parent
```

Paginate until every item is accounted for. Record IDs, names, types, target folder IDs, and a completed-ID log. Before broad moves, access changes, or AI use, present the scope and ambiguous cases for approval.

## Classify content

Prefer deterministic filename, extension, and existing-metadata rules. For semantic classification, use Box AI rather than downloading file bodies:

```bash
box ai:ask --items=id=<FILE_ID>,type=file \
  --prompt "Classify as invoice, receipt, contract, report, or other." --json
```

For known fields, use `ai:extract-structured`; for variable fields, use `ai:extract`. Sample a small representative set before processing a large batch. Disclose Box AI unit use and obtain confirmation before a material AI batch.

## Execute and recover

```bash
box folders:create <PARENT_ID> "Category" --json --fields id,name
box files:move <FILE_ID> <TARGET_FOLDER_ID> --json --fields id,name,parent
```

Process ordered CLI mutations serially and log each success or failure. On `409`, find and reuse the existing target. On `429`, honor `Retry-After` and retry the same request. Resume from `inventory minus completed IDs`; do not restart blindly.

Use a documented `--bulk-file-path` workflow when the relevant command supports it. Use bounded SDK concurrency only when the application owns retries, idempotency, and rate-limit handling.

## Verify and report

List each destination and the source folder, then compare IDs and counts with the plan. Report links to the source folder, destination folders, and exceptions. Do not dump hundreds of item links unless the user asks for a manifest.
