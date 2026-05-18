# Route-A Project-local MCP Policy

Route-A is the production CloudBase operation path for this profile.

## Production path

```text
scripts/cloudbase-mcp-call.mjs <project> <tool> [key=value ...]
  -> scripts/cloudbase-preflight.mjs
  -> <project>/config/mcporter.json
  -> scripts/cloudbase-mcp-local.mjs <project>
  -> npx @cloudbase/cloudbase-mcp@latest
```

## Required files

Repository/workspace-level:

```text
config/cloudbase-projects.json
scripts/cloudbase-preflight.mjs
scripts/cloudbase-mcp-local.mjs
scripts/cloudbase-mcp-call.mjs
scripts/cloudbase-function-diff.mjs
scripts/cloudbase-function-deploy-command.mjs
```

Project-level:

```text
<project>/cloudbaserc.json
<project>/config/mcporter.json
<project>/.mcp.json
<project>/.cloudbase-mcp.env.example
<project>/.cloudbase-mcp.env        # ignored, local only
<project>/.cloudbase-mcp-logs/      # ignored, local only
```

## Hard blocks

Production operations must not use:

- raw `tcb` CLI;
- direct `npx mcporter call ...`;
- Hermes profile-level CloudBase MCP for multiple projects;
- `auth action=set_env`;
- `auth action=logout`.

The wrapper should reject auth environment switching and write/manage tools without `--write-ok`.

## Preflight checks

Before writes, verify:

- project exists in `config/cloudbase-projects.json`;
- MCP server name matches the project;
- `cloudbaserc.json` EnvId matches expected EnvId;
- MCP env `CLOUDBASE_ENV_ID` matches expected EnvId;
- credentials are present and project-specific;
- resource is allowed when an allowlist exists.

## Safe invocation examples

Read-only:

```bash
scripts/cloudbase-mcp-call.mjs meme envQuery action=info envId=meme-d9gg2ac1p467b701c
scripts/cloudbase-mcp-call.mjs meme queryFunctions action=listFunctions
```

Code-only deploy after explicit confirmation:

```bash
scripts/cloudbase-mcp-call.mjs meme manageFunctions action=updateFunctionCode functionName=api functionRootPath=/abs/path/cloudbase/functions --write-ok
```

## Notes

`functionRootPath` must be the directory containing function subdirectories, not the function directory itself.
