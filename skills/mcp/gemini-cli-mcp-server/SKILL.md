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

This MCP server shells out to `gemini -p "prompt"` for tool calls. It uses the Gemini CLI binary (OAuth-authenticated), NOT the Google Gen AI Python SDK. This means:

- **No API keys to configure** — uses the CLI's existing OAuth session
- **Full Gemini model access** — the 1M token context window of Gemini models
- **Plan mode for safety** — read-only review/audit mode (guaranteed no mutations)
- **Safe by default** — `gemini_prompt` uses `--approval-mode auto-edit`, requires explicit opt-in for yolo mode

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
- **Python 3.11+**

## Installation

### Option 1: pip install (when published to PyPI)

```bash
pip install gemini-mcp-server
gemini-mcp    # starts the MCP server
```

### Option 2: From source

```bash
git clone https://github.com/jxsprt/gemini-mcp-server.git
cd gemini-mcp-server
python3 -m venv .venv
.venv/bin/pip install -e .
```

### 2. Add to Hermes config

In `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  gemini-cli:
    command: "/path/to/gemini-mcp-server/.venv/bin/python3"
    args: ["-m", "gemini_mcp_server"]
    timeout: 240
```

Or if installed globally:

```yaml
mcp_servers:
  gemini-cli:
    command: "gemini-mcp"
    timeout: 240
```

Restart the Hermes gateway. Tools auto-discover as `mcp_gemini_cli_*`.

### 3. Verify

```bash
python3 -m gemini_mcp_server    # starts the server
# In another terminal:
pip install fastmcp
fastmcp list /path/to/server.py --json
```

## Tools (3)

### `gemini_prompt`
Send any prompt to Gemini CLI. **Safe by default** — uses `auto_edit` mode. Pass `dangerous=True` for full `yolo`.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `prompt` | ✅ | Prompt text (max ~100K chars, configurable via `GEMINI_MAX_PROMPT`) |
| `model` | ❌ | Model name (e.g., `gemini-3-flash-preview`) |
| `output_format` | ❌ | `text`, `json`, or `stream-json` |
| `dangerous` | ❌ | `true` = `--approval-mode yolo` (auto-approves shell commands) |

### `gemini_plan`
Read-only audit and review mode. Uses `--approval-mode plan` — **guarantees no file mutations or shell execution**.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `prompt` | ✅ | Analysis question or review request |
| `model` | ❌ | Gemini model |
| `include_directories` | ❌ | Comma-separated additional dirs (validated to CWD subdirs only) |

### `gemini_version`
Returns the installed Gemini CLI version (e.g., `0.41.2`).

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_CLI_PATH` | auto-discover | Explicit path to `gemini` binary |
| `GEMINI_TIMEOUT` | `120` | Timeout for `gemini_prompt` (seconds) |
| `GEMINI_PLAN_TIMEOUT` | `240` | Timeout for `gemini_plan` (seconds) |
| `GEMINI_MAX_PROMPT` | `100000` | Max prompt character length |
| `GEMINI_MCP_HTTP` | `0` | Set to `1`/`true` to enable HTTP transport on port 8000 |

## Common Pitfalls

1. **Gemini CLI not on PATH.** Set `GEMINI_CLI_PATH` to the full binary path if auto-discovery fails.
2. **Plan mode timeouts.** Large repos with 1M token context can take 2-4 minutes. Increase `GEMINI_PLAN_TIMEOUT`.
3. **Capacity errors.** Gemini's reasoning models hit rate limits. The CLI retries up to 7 times. If it fails, try again or use a different model.
4. **Not a pip package yet.** `gemini-mcp-server` is not on PyPI yet. Install from source or wait for it to be uploaded.
5. **Filtered environment.** Hermes passes a filtered env to MCP subprocesses. Gemini works because it uses `~/.gemini/gemini-credentials.json`, not env vars.

## Verification Checklist

- [ ] Gemini CLI is installed: `gemini --version`
- [ ] Gemini CLI is authenticated: `gemini -p "hi"` works in terminal
- [ ] Package installed: `python3 -c "from gemini_mcp_server import gemini_version; print(gemini_version())"`
- [ ] Config entry exists in `~/.hermes/config.yaml` under `mcp_servers.gemini-cli`
- [ ] Gateway is running and MCP tools appear (check `/reload-mcp`)

## Related

- [github.com/jxsprt/gemini-mcp-server](https://github.com/jxsprt/gemini-mcp-server) — source code
- `fastmcp` skill — for building similar MCP servers
- `native-mcp` skill — for configuring MCP servers in Hermes
