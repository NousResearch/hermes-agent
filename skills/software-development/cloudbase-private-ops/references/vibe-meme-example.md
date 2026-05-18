# Vibe Photoing + meme Example

This is the validated reference implementation for project-local CloudBase MCP isolation in this profile. Do not copy secrets from live project files into shared docs.

## Project map

| Project | EnvId | MCP server | Function root |
|---|---|---|---|
| `vibe-photoing` | `cloudbase-d6gflq1jq7012a851` | `cloudbase-vibe-photoing` | `miniapps/vibe-photoing/cloudfunctions` |
| `meme` | `meme-d9gg2ac1p467b701c` | `cloudbase-meme` | `meme/cloudbase/functions` |

## Files

Workspace-level:

```text
config/cloudbase-projects.json
scripts/cloudbase-preflight.mjs
scripts/cloudbase-mcp-local.mjs
scripts/cloudbase-mcp-call.mjs
scripts/cloudbase-function-diff.mjs
scripts/cloudbase-function-deploy-command.mjs
```

Vibe:

```text
miniapps/vibe-photoing/.mcp.json
miniapps/vibe-photoing/config/mcporter.json
miniapps/vibe-photoing/.cloudbase-mcp.env.example
miniapps/vibe-photoing/.cloudbase-mcp.env        # ignored
miniapps/vibe-photoing/cloudbaserc.json
```

meme:

```text
meme/.mcp.json
meme/config/mcporter.json
meme/.cloudbase-mcp.env.example
meme/.cloudbase-mcp.env        # ignored
meme/cloudbaserc.json
```

## Validated operations

Read-only:

```bash
scripts/cloudbase-mcp-call.mjs vibe-photoing envQuery action=info envId=cloudbase-d6gflq1jq7012a851
scripts/cloudbase-mcp-call.mjs vibe-photoing queryFunctions action=listFunctions
scripts/cloudbase-mcp-call.mjs meme envQuery action=info envId=meme-d9gg2ac1p467b701c
scripts/cloudbase-mcp-call.mjs meme queryFunctions action=listFunctions
```

Code-only writes validated:

```text
meme / api / updateFunctionCode -> success; health OK
vibe-photoing / listImageJobs / updateFunctionCode -> success; detail OK
```

## Pitfalls found during validation

- `getImageJob` had live envVariables missing from local `cloudbaserc.json`; merge live env shape before config deploys.
- Timer triggers return live as `TriggerName` / `Type` / `TriggerDesc={"cron":"..."}`. Normalize before diffing against local `{name,type,config}`.
- `functionRootPath` must point to the directory containing function subdirectories.
- Root workspace may not be a git repo; do not assume root docs/scripts/config are governed by git.
- Generated reports and deploy plans must redact real SecretId/SecretKey/API_TOKEN values.
- Do not test config/env/trigger/DB/storage writes just to prove MCP; code-only writes already proved migration viability.

## Current policy outcome

Production CloudBase operations should use MCP route-A. Raw `tcb`, direct `mcporter`, and Hermes profile-level CloudBase MCP are not production entry points for these projects.
