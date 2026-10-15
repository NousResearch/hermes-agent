# ACP Registry — Agent Communication Protocol

Hermes Agent supports the **Agent Communication Protocol (ACP)** for agent-to-agent and agent-to-editor integration. The ACP registry defines how Hermes advertises itself to ACP-compatible editors and tools.

## Registry Manifest

The `acp_registry/agent.json` file in the Hermes repo is the canonical registry entry:

```json
{
  "id": "hermes-agent",
  "name": "Hermes Agent",
  "version": "0.18.0",
  "description": "Self-improving open-source AI agent by Nous Research with ACP editor integration, persistent memory, skills, and rich tool support.",
  "repository": "https://github.com/NousResearch/hermes-agent",
  "website": "https://hermes-agent.nousresearch.com/docs/user-guide/features/acp",
  "authors": ["Nous Research"],
  "license": "MIT",
  "distribution": {
    "uvx": {
      "package": "hermes-agent[acp]==0.18.0",
      "args": ["hermes-acp"]
    }
  }
}
```

### Manifest Fields

| Field | Required | Description |
|-------|:---:|-------------|
| `id` | ✅ | Unique identifier for the agent |
| `name` | ✅ | Human-readable display name |
| `version` | ✅ | Semver version, matches `pyproject.toml` |
| `description` | ✅ | One-line summary |
| `repository` | — | Source code repository URL |
| `website` | — | Documentation / landing page |
| `authors` | ✅ | List of author/org names |
| `license` | ✅ | SPDX license identifier |
| `distribution` | ✅ | How to install and run the agent |

### Distribution Methods

#### `uvx` (recommended)

```json
"distribution": {
  "uvx": {
    "package": "hermes-agent[acp]==0.18.0",
    "args": ["hermes-acp"]
  }
}
```

Runs via `uv tool run` — no pre-installation needed. The editor downloads and runs the agent on first use.

#### `pipx`

```json
"distribution": {
  "pipx": {
    "package": "hermes-agent[acp]",
    "args": ["hermes-acp"]
  }
}
```

## Connecting Hermes ACP to Editors

### Zed

1. Open Zed → `zed: acp registry`
2. Search for **Hermes Agent**
3. Click **Install**

Or manually:
```json
// ~/.config/zed/settings.json
{
  "agent": {
    "enabled": true,
    "default_model": "hermes-agent"
  }
}
```

### VS Code (ACP extension)

Install the ACP extension for VS Code, then:

```json
{
  "acp.agents": [
    {
      "id": "hermes-agent",
      "command": "hermes-acp"
    }
  ]
}
```

### JetBrains

Install the ACP plugin from JetBrains Marketplace, then configure:

```
Settings → Tools → ACP → Add Agent
  Name: Hermes Agent
  Command: hermes-acp
```

### Cursor

```json
// ~/.cursor/acp.json
{
  "agents": {
    "hermes": {
      "command": "hermes-acp"
    }
  }
}
```

## ACP Check

Verify ACP is working:

```bash
hermes acp --check
```

Expected output:
```json
{
  "status": "ok",
  "version": "0.18.0",
  "protocol": "acp-2025-01-01"
}
```

## Registry Validation

The registry manifest is validated on every CI run:

```bash
# tests/acp/test_registry_manifest.py enforces:
# - agent.json is valid JSON
# - All required fields present
# - version matches pyproject.toml
# - icon.svg exists and is valid XML
pytest tests/acp/test_registry_manifest.py
```

## Submitting to the ACP Registry

To register your own agent in the community ACP registry:

1. Create an `acp_registry/` directory with `agent.json` and `icon.svg`
2. Ensure `version` matches your release
3. Open a PR to the ACP community registry

## ACP vs MCP

| Feature | ACP | MCP |
|---------|-----|-----|
| Purpose | Agent ↔ Editor integration | Tool ↔ Agent integration |
| Transport | Stdio (JSON-RPC) | Stdio/HTTP/SSE (JSON-RPC) |
| Hermes entry point | `hermes-acp` | `hermes mcp serve` |
| Use case | IDE code actions, inline edits | External tool access, browser/search |
| Install | Editor package manager (uvx) | `hermes mcp add` or config.yaml |

## See Also

- [MCP Reference](mcp-reference.md) — MCP server/client docs
- [IDE Integration Guide](ide-integration.md) — full IDE setup walkthrough
