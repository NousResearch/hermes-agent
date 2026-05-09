---
name: gemini-cli-mcp-server
description: "Use when connecting to or troubleshooting the Gemini CLI MCP server — a FastMCP server that wraps Google's Gemini CLI as MCP tools (gemini_prompt, gemini_plan, gemini_version) for use with Hermes Agent or any MCP client."
version: 1.0.0
author: Jaspreet Singh (@jxsprt) + Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [mcp, gemini, gemini-cli, fastmcp, tools, server]
    related_skills: [fastmcp, native-mcp]
---

# Gemini CLI MCP Server

A lightweight [FastMCP](https://gofastmcp.com) server that wraps Google's [Gemini CLI](https://github.com/google-gemini/gemini-cli) and exposes it as MCP tools. Works with any MCP client that supports stdio transport.

**Source code:** [github.com/jxsprt/gemini-mcp-server](https://github.com/jxsprt/gemini-mcp-server)

## Overview

This MCP server shells out to `gemini -p "prompt" --approval-mode yolo` for tool calls. It uses the Gemini CLI binary (OAuth-authenticated), NOT the Google Gen AI Python SDK. This means:

- **No API keys to configure** — uses the CLI's existing OAuth session
- **Full Gemini model access** — the 1M token context window of Gemini models
- **Plan mode for safety** — read-only review/audit mode (guaranteed no mutations)

## When to Use

- You want to query Gemini models (research, analysis, code review) from within Hermes Agent
- You need read-only plan mode (`--approval-mode plan`) for safe code audits
- You want Gemini's 1M token context window for large codebase analysis
- You're troubleshooting why the MCP server isn't connecting

## Prerequisites

- **Gemini CLI** installed and authenticated:
  ```bash
  npm install -g @google/gemini-cli
  gemini --version          # Verify
  gemini                    # First run completes OAuth login
  ```
- **Python 3.11+** with `fastmcp` package

## Installation

### 1. Clone the server

```bash
git clone https://github.com/jxsprt/gemini-mcp-server.git
cd gemini-mcp-server
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 2. Add to Hermes config

In `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  gemini-cli:
    command: "/path/to/gemini-mcp-server/.venv/bin/python3"
    args: ["/path/to/gemini-mcp-server/server.py"]
    timeout: 240
```

Restart the Hermes gateway. Tools auto-discover as `mcp_gemini_cli_*`.

### 3. Verify

```bash
cd /path/to/gemini-mcp-server
.venv/bin/python -c "
import server
print('Version:', server.gemini_version())
print('Test:', server.gemini_prompt('Say hello in one word'))
"
```

## Tools (3)

### `gemini_prompt`
Send any prompt to Gemini CLI. Supports model selection and JSON output.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `prompt` | ✅ | Prompt text (max ~100K chars) |
| `model` | ❌ | Model name (e.g., `gemini-3-flash-preview`) |
| `output_format` | ❌ | `text`, `json`, or `stream-json` |

### `gemini_plan`
Read-only audit and review mode. Uses `--approval-mode plan` — **guarantees no file mutations or shell execution**.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `prompt` | ✅ | Analysis question or review request |
| `model` | ❌ | Gemini model |
| `include_directories` | ❌ | Comma-separated additional dirs in workspace |

### `gemini_version`
Returns the installed Gemini CLI version (e.g., `0.41.2`).

## Common Pitfalls

1. **Gemini CLI not on PATH.** The server discovers the binary via `shutil.which("gemini")`. If it can't find it, the server won't start. Ensure `npm install -g @google/gemini-cli` ran successfully.

2. **Plan mode timeouts.** Large repos with the full 1M context can take 2-4 minutes. The default timeout is 240s — increase it in the config if needed.

3. **Capacity errors.** Gemini's reasoning models hit rate limits under load. The CLI retries up to 7 times with exponential backoff. If it fails, try again later or use a different model.

4. **Filtered environment in MCP subprocesses.** Hermes passes a filtered environment to MCP servers. The `gemini` binary still works because it uses OAuth credentials stored at `~/.gemini/gemini-credentials.json`, not env vars.

## Verification Checklist

- [ ] Gemini CLI is installed: `gemini --version`
- [ ] Gemini CLI is authenticated: `gemini -p "hi"` works in terminal
- [ ] Server venv is set up: `.venv/bin/python -c "import fastmcp"`
- [ ] Config entry exists in `~/.hermes/config.yaml` under `mcp_servers.gemini-cli`
- [ ] Gateway is running and MCP tools appear in Hermes (check `/reload-mcp`)
- [ ] Direct test passes: `.venv/bin/python -c "import server; print(server.gemini_version())"`

## Related

- [github.com/jxsprt/gemini-mcp-server](https://github.com/jxsprt/gemini-mcp-server) — source code with full README
- `fastmcp` skill — for building similar MCP servers
- `native-mcp` skill — for configuring MCP servers in Hermes
