---
name: screenpipe
description: Connect Hermes to local Screenpipe memory. Diagnose the local API, install Screenpipe's MCP server into Hermes config, and run quick searches against the local Screenpipe index.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [Screenpipe, MCP, Desktop Context, OCR, Audio, Memory]
    related_skills: [fastmcp, qmd]
    category: mcp
---

# Screenpipe

Use this optional skill when you want Hermes to see what happened on your machine through Screenpipe's local screen and audio memory.

This skill handles the practical parts:

- check whether the Screenpipe REST API is reachable on localhost
- add the official `screenpipe-mcp` stdio server to Hermes config
- run quick search calls against the local API before you spend time debugging MCP

## Install

```bash
hermes skills install official/mcp/screenpipe
```

Locate the helper script:

```bash
SCRIPT="$(find ~/.hermes/skills -path '*/screenpipe/scripts/screenpipe_helper.py' -print -quit)"
```

## Upstream

- Desktop app / docs: `https://screenpi.pe`
- Local REST API: `http://127.0.0.1:3030`
- Official MCP launcher: `npx -y screenpipe-mcp`

## Quick Start

If Screenpipe is not already installed, either install the desktop app or start recording via CLI:

```bash
npx screenpipe@latest record
```

Run a doctor pass:

```bash
python3 "$SCRIPT" doctor
```

Install the MCP server into Hermes config:

```bash
python3 "$SCRIPT" install-mcp
```

Then reload MCP inside Hermes:

```text
/reload-mcp
```

Smoke-test the Screenpipe API directly:

```bash
python3 "$SCRIPT" search --query "meeting notes" --content-type ocr --limit 5
```

## Workflow

1. Make sure Screenpipe is recording locally.
2. Run `doctor` and confirm:
   - the API is reachable
   - `npx` exists
   - Hermes config already has or can accept a `screenpipe` MCP entry
3. Install the MCP entry if missing.
4. Reload MCP and ask Hermes natural questions like:
   - "What did I work on in the last 30 minutes?"
   - "Search my recent screen history for Linear tickets."

## Notes

- The helper uses the local Screenpipe REST API for quick verification.
- The MCP config it writes is stdio-based:

```yaml
mcp_servers:
  screenpipe:
    command: npx
    args: ["-y", "screenpipe-mcp"]
```

## Verification

Good output from `doctor` should show:

- `api_reachable: true`
- `npx: true`
- either `mcp_configured: true` or a clean config path where the helper can add it
