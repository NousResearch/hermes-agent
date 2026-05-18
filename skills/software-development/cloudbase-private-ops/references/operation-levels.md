# CloudBase Operation Levels

Use operation levels to avoid migration-era over-verification while preserving safety for high-risk writes.

## Level 0 — Read-only

Examples:

- `envQuery`
- `queryFunctions`
- `querySqlDatabase` read-only queries
- `queryStorage list/info/url`
- function detail/log inspection

Workflow:

```bash
scripts/cloudbase-mcp-call.mjs <project> <queryTool> ...
```

No rollback or diff required. Still use route-A wrapper to prevent project/environment mixups.

## Level 1 — Single-function code-only deploy

Examples:

- `manageFunctions action=updateFunctionCode` for one function.

Workflow:

```bash
scripts/cloudbase-preflight.mjs <project> deploy <functionName>
scripts/cloudbase-function-diff.mjs <project>
scripts/cloudbase-function-deploy-command.mjs <project> <functionName> --code
scripts/cloudbase-mcp-call.mjs <project> manageFunctions action=updateFunctionCode functionName=<functionName> functionRootPath=<functionRoot> --write-ok
scripts/cloudbase-function-diff.mjs <project>
scripts/cloudbase-mcp-call.mjs <project> queryFunctions action=getFunctionDetail functionName=<functionName>
```

Use a rollback source from git or a prepared snapshot. Do not combine with config/env/trigger updates.

## Level 2 — Config/env/trigger writes

Examples:

- `updateFunctionConfig`
- envVariables / secret changes
- timeout/memory/runtime-adjacent changes
- trigger create/delete

Workflow:

- create a per-operation runbook;
- prepare rollback;
- run preflight and diff;
- require explicit user confirmation naming project, EnvId, function, operation, and rollback;
- execute one operation only;
- post-check detail and health.

Do not perform these merely to test MCP. Code-only deploys already validate MCP migration.

## Level 3 — DB/storage/backfill/destructive writes

Examples:

- SQL writes, schema changes, backfills;
- NoSQL writes/deletes;
- Storage overwrite/delete;
- batch operations.

Workflow:

- run read-only counts / dry-run first;
- scope data by allowlist or prefix;
- prepare rollback or export where possible;
- require explicit confirmation;
- verify with post-counts and sample reads.

## Always forbidden in production

- `auth.set_env`
- `auth.logout`
- raw `tcb` environment switching
- using one global CloudBase credential/profile for multiple projects
