---
name: context7-docs
description: Enable Context7 MCP Server for real-time, version-specific code documentation. Provides live documentation for any library or framework via the Context7 MCP protocol, accessible through Hermes Agent's built-in MCP client.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [MCP, Documentation, Context7, Tools, Integrations]
    related_skills: [native-mcp, mcporter]
prerequisites:
  commands: [npx]
---

# Context7 MCP Documentation Server

[Context7](https://github.com/upstash/context7) is an MCP server that provides real-time, version-specific code documentation for any library or framework. When connected to Hermes Agent, it gives the agent access to up-to-date API references, code examples, and usage patterns.

## When to Use

Use Context7 when you need:
- Real-time documentation for any npm, PyPI, or other package
- Version-specific documentation (e.g., React 18 vs React 19)
- Code examples and API references during coding sessions
- Documentation that's more current than the model's training data

## Prerequisites

- **Node.js** — required for `npx` to run the Context7 MCP server
- **mcp Python package** — required for Hermes Agent's MCP client support

Install the MCP SDK if not already installed:

```bash
pip install mcp
```

## Quick Start: One-Click Enable

### Option 1: Using the Python config generator

```python
from tools.documentation_fetcher import generate_context7_mcp_config, generate_context7_mcp_yaml

# Get as dict (for programmatic merging):
config = generate_context7_mcp_config()

# Or get as YAML (for copy-paste into config.yaml):
print(generate_context7_mcp_yaml())
```

### Option 2: Manual config edit

Add this to `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  context7:
    command: "npx"
    args: ["-y", "@upstash/context7-mcp"]
    timeout: 120
    connect_timeout: 60
```

Then restart Hermes Agent.

## How It Works

When Hermes Agent starts with the Context7 server configured:

1. Hermes launches `npx -y @upstash/context7-mcp` as a subprocess
2. The MCP client connects via stdio transport
3. Context7's tools are discovered and registered with the prefix `mcp_context7_*`
4. Tools are automatically injected into all platform toolsets

## Available Tools

After connection, Context7 provides tools like:

- **mcp_context7_resolve-library-id** — Find the correct library ID for a package name
- **mcp_context7_get-library-docs** — Fetch documentation for a specific library

These tools are called with the `mcp_context7_` prefix, e.g.:
```
mcp_context7_get-library-docs(libraryId="react", topic="hooks")
```

## Usage Examples

### Get React Hooks Documentation
Ask the agent: "How do I use React hooks? Use Context7 to get the latest docs."

### Search for a Specific API
Ask the agent: "Look up the Next.js App Router documentation using Context7."

### Version-Specific Docs
Ask the agent: "Show me the Vue 3 composition API docs from Context7."

## Combining with fetch_docs

Hermes Agent also has a built-in `fetch_docs` tool that works without MCP:
- `fetch_docs` uses HTTP API calls to Context7 and local llms.txt files
- `mcp_context7_*` tools use the MCP protocol directly

Both can be used simultaneously. The `fetch_docs` tool is always available; the MCP tools require the server to be running.

## Troubleshooting

### "missing executable 'npx'"
Node.js is not installed or not on PATH. Install Node.js:
```bash
# Using nvm:
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
nvm install --lts

# Or using your system package manager
```

### "MCP SDK not available"
The `mcp` Python package is not installed:
```bash
pip install mcp
```

### Connection timeout
The npx command takes time to download the package on first run. Increase `connect_timeout`:
```yaml
mcp_servers:
  context7:
    command: "npx"
    args: ["-y", "@upstash/context7-mcp"]
    connect_timeout: 120  # Increase to 2 minutes
```

### Tools not appearing
- Check that `mcp_servers` is at the top level of config.yaml (not nested)
- Verify YAML indentation is correct (2 spaces)
- Check Hermes startup logs for MCP connection messages
- Tool names use the pattern `mcp_context7_*`

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `command` | string | `"npx"` | Executable to run |
| `args` | list | `["-y", "@upstash/context7-mcp"]` | Command arguments |
| `timeout` | int | `120` | Per-tool-call timeout (seconds) |
| `connect_timeout` | int | `60` | Initial connection timeout (seconds) |

## Links

- [Context7 GitHub](https://github.com/upstash/context7)
- [Context7 Website](https://context7.com)
- [Hermes Native MCP Client](skills/mcp/native-mcp/SKILL.md)
- [mcporter CLI](skills/mcp/mcporter/SKILL.md)
