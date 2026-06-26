# CLI Guide

Use Box CLI via the `terminal` tool. Default flags: `--json` plus `--fields` to minimize token output.

## Safe checks

```bash
command -v box
box --version
box users:get me --json --fields id,name,login
```

## Output control

```bash
# Minimal fields
box folders:items 0 --json --max-items 20 --fields id,name,type

# Pagination — repeat with offset or use --max-items
box search "contract" --json --limit 25
```

Run `box <command> --help` before first use of an unfamiliar subcommand.

## Serial execution

**One `box` command at a time.** Parallel CLI processes against the same environment cause auth conflicts and dropped operations.

## Common commands

```bash
# Browse
box folders:get <ID> --json --fields id,name,item_collection
box folders:items <ID> --json --max-items 100 --fields id,name,type

# Write
box folders:create <PARENT_ID> "Project-Alpha" --json
box files:upload ./report.pdf --parent-id <FOLDER_ID> --json
box files:download <FILE_ID> --destination . --save-as report.pdf

# Edit (metadata + content)
box files:update <FILE_ID> --name "Q1-Report.pdf" --description "Final" --json
box files:versions:upload <FILE_ID> ./report-v2.pdf --json
box files:upload ./report-v2.pdf --parent-id <FOLDER_ID> --overwrite --json
box files:move <FILE_ID> <NEW_PARENT_ID> --json

# Share
box shared-links:create <FILE_ID> file --access company --json
box collaborations:create <FOLDER_ID> folder --role editor --login user@example.com --json

# Search
box search "quarterly review" --json --limit 20
box metadata-query --help  # requires template scope/key and ancestor folder ID

# Admin-ish (scope-dependent)
box users:get me --json
box webhooks:list --json
```

Folder **`0`** is the current actor's root.

## `box request` — REST escape hatch

When no dedicated subcommand exists, call any API path:

```bash
box request /files/<FILE_ID> --json
box request /files/<FILE_ID> -X PUT --body '{"name":"renamed.pdf"}' --json
box request /folders -X POST --body '{"name":"New","parent":{"id":"0"}}' --json
```

Prefer `box request` over hand-written curl when CLI is installed — same auth, same environment.

## Actor controls

- **`--as-user <USER_ID>`** — impersonate another user (requires app scopes and access level).
- **Separate CCG environment with `--ccg-user`** — persistent impersonation; see `references/auth-and-setup.md`.

## When to use REST instead

Only when `box` is not installed or user declines CLI setup. See `references/rest-api.md`.

## Bulk work

For more than a handful of moves, follow `references/bulk-operations.md` (inventory → plan → serial execute → verify).
