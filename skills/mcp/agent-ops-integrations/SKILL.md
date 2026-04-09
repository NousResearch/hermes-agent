---
name: agent-ops-integrations
description: Connect Hermes Agent to OpenTabs, Presenton, PPT Master, and CUA/cuabot with the native MCP client and terminal-side sandboxes.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [mcp, integrations, browser, presentations, sandbox]
    related_skills: [native-mcp, powerpoint]
---

# Agent Ops Integrations

Use this skill when Hermes should operate with the external stacks we vetted:
- **OpenTabs** for API-first browser actions
- **Presenton** for AI presentation generation over MCP
- **PPT Master** for editable PPTX generation workflows
- **CUA / cuabot** for sandboxed desktop actions

## 1. OpenTabs via native MCP

Prereqs:
```bash
npm install -g @opentabs-dev/cli
opentabs start
```

Add to `~/.hermes/config.yaml`:
```yaml
mcp_servers:
  opentabs:
    command: "opentabs"
    args: ["start", "--mcp"]
    timeout: 180
    connect_timeout: 60
```

Then restart Hermes. Tools appear as `mcp_opentabs_*`.

## 2. Presenton via native MCP

Presenton ships a built-in MCP server. Prefer that path over browser driving.

Typical setup:
```bash
# install/run Presenton per upstream docs, then expose its MCP endpoint
```

If running local HTTP MCP:
```yaml
mcp_servers:
  presenton:
    url: "http://127.0.0.1:9040/mcp"
    timeout: 300
    connect_timeout: 60
```

If upstream exposes stdio instead, use `command`/`args` instead of `url`.

## 3. PPT Master workflow

PPT Master is best used as a **skill + terminal workflow**, not as a browser bot.

```bash
git clone https://github.com/hugohe3/ppt-master.git ~/tools/ppt-master
cd ~/tools/ppt-master
pip install -r requirements.txt
```

Use Hermes to:
1. generate the deck brief/outline
2. write the source markdown/content files
3. run PPT Master scripts in terminal
4. QA the output with the built-in `powerpoint` skill

## 4. CUA / cuabot sandbox

Use CUA for higher-risk desktop automation that should not run directly on the host.

```bash
pip install cua
npx cuabot
```

Examples:
```bash
cuabot chromium
cuabot --screenshot
cuabot --type "hello"
```

## When to choose what

- **OpenTabs**: web app has stable internal APIs; avoid brittle UI clicking
- **Presenton**: generate a presentation from prompt/docs quickly via MCP
- **PPT Master**: need editable PPTX with layout control and manual QA loop
- **CUA/cuabot**: desktop/native-app automation, risky GUI actions, or sandbox-first workflows

## Hermes fit

- Prefer **native MCP** for OpenTabs and Presenton
- Prefer **skills + terminal** for PPT Master
- Prefer **sandbox sidecar** for CUA/cuabot
- Do **not** wire these into Hermes core decision logic; keep them as optional capability layers
