# New Project Template

Use this to onboard a new CloudBase project into the private multi-account MCP isolation pattern.

Assume:

```text
project = new-app
envId = new-env-xxxx
projectRoot = /abs/path/new-app
server = cloudbase-new-app
```

## 1. Register in project map

Add to `config/cloudbase-projects.json`:

```json
{
  "new-app": {
    "envId": "new-env-xxxx",
    "mcpServer": "cloudbase-new-app",
    "projectRoot": "/abs/path/new-app",
    "cloudbaserc": "/abs/path/new-app/cloudbaserc.json",
    "mcpConfig": "/abs/path/new-app/config/mcporter.json",
    "envFile": "/abs/path/new-app/.cloudbase-mcp.env",
    "logDir": "/abs/path/new-app/.cloudbase-mcp-logs",
    "allowlist": {
      "functions": [],
      "collections": [],
      "storagePrefixes": []
    }
  }
}
```

## 2. Create project-local mcporter config

`new-app/config/mcporter.json`:

```json
{
  "mcpServers": {
    "cloudbase-new-app": {
      "command": "node",
      "args": [
        "/home/openclaw/hermes_workspace/hermes_sc/scripts/cloudbase-mcp-local.mjs",
        "new-app"
      ],
      "description": "Project-local CloudBase MCP for new-app only",
      "lifecycle": "keep-alive"
    }
  }
}
```

## 3. Create optional IDE MCP config

`new-app/.mcp.json`:

```json
{
  "mcpServers": {
    "cloudbase-new-app": {
      "command": "node",
      "args": [
        "/home/openclaw/hermes_workspace/hermes_sc/scripts/cloudbase-mcp-local.mjs",
        "new-app"
      ]
    }
  }
}
```

Production operations still use route-A wrapper, not IDE auto-loaded tools.

## 4. Create credentials template

`new-app/.cloudbase-mcp.env.example`:

```env
NEW_APP_TENCENTCLOUD_SECRETID=
NEW_APP_TENCENTCLOUD_SECRETKEY=
CLOUDBASE_ENV_ID=new-env-xxxx
CLOUDBASE_LOG_DIR=.cloudbase-mcp-logs
```

Real credentials go to `new-app/.cloudbase-mcp.env`, which must be ignored.

## 5. Patch .gitignore

```gitignore
.cloudbase-mcp.env
.cloudbase-mcp-logs/
cloudbaserc.local.json
```

## 6. Create cloudbaserc

`new-app/cloudbaserc.json`:

```json
{
  "envId": "new-env-xxxx",
  "functionRoot": "cloudfunctions",
  "functions": []
}
```

## 7. Smoke test

```bash
scripts/cloudbase-preflight.mjs new-app status
scripts/cloudbase-mcp-call.mjs new-app envQuery action=info envId=new-env-xxxx
scripts/cloudbase-mcp-call.mjs new-app queryFunctions action=listFunctions
```

The returned EnvId must be the new project EnvId, not any existing Vibe/meme environment.
