# Route-A Migration Closeout Pattern

Use this after proving that project-local CloudBase MCP can replace `tcb` for production operations.

## What to prove once

For each major project class, validate one low-risk code-only write:

- preflight passes;
- whole-project function diff is clean;
- `manageFunctions action=updateFunctionCode` succeeds for one function;
- post-write diff remains clean;
- function status is `Active / Available`;
- health/read-only verification passes when available;
- rollback artifact exists but is not needed.

After this succeeds in representative projects, do not keep testing higher-risk write types merely to prove MCP. Code-only deployment proves the transport, project isolation, and write gate. Treat config/env/trigger/DB/storage writes as separate real operations with their own runbook.

## Closeout audit

Before declaring the migration done:

1. Check which directories are actually git repositories. Do not assume the workspace root is a repo.
2. Inspect status/diff for each real repo touched.
3. Scan generated docs/reports for literal secrets; redact values, but key names such as `SecretKey` or `VIBE_PHOTOING_API_TOKEN` may remain as labels.
4. Confirm `.cloudbase-mcp.env`, local logs, rollback archives, and local `cloudbaserc.local.json` are ignored or outside repos.
5. Write a concise closeout document listing:
   - production path;
   - verified write operations;
   - core scripts/config/docs;
   - local-only rollback artifacts;
   - known caveats and future operation levels.

## Post-migration operating model

- Level 0 read-only: wrapper call only.
- Level 1 code-only single-function deploy: preflight + diff + post-check.
- Level 2 config/env/trigger: per-operation runbook + rollback + explicit confirmation.
- Level 3 DB/storage/backfill/destructive: dry-run/read-only counts + runbook + rollback + explicit confirmation.

Do not delete CloudBase CLI immediately just because MCP migration worked. Prefer a short observation period where `tcb` is retained only as an emergency/read-only comparison tool and is forbidden as a production entry point. Later, disable or uninstall it if no fallback was needed.
