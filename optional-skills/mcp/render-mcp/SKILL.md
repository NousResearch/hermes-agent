---
name: render-mcp
description: Connect Hermes to Render and verify workspace access.
version: 1.0.0
author: Ojus Save (@ojusave), Render
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [render, cloud, deployment, devops]
    category: mcp
    upstream:
      repo: render-oss/skills
      path: skills/render-mcp
---

# Render MCP Skill

Connect Hermes to Render so it can inspect and manage the intended workspace. This entry installs Render's existing setup skill; it does not enable the Render Sandboxes terminal backend.

## When to Use

- Set up the Render MCP connection for the first time.
- Diagnose missing Render tools or authentication errors.
- Check which Render workspace a connection can access.

## Prerequisites

- A Render account with access to the intended workspace.
- The hosted Render MCP connection, or the Render CLI for operations it supports. For MCP, run `hermes mcp add render --url https://mcp.render.com/mcp` from the host terminal and follow its hidden credential prompt. Use [Render's access instructions](https://render.com/docs/mcp-server) for current requirements.
- For CLI operations, install the [Render CLI](https://render.com/docs/cli), authenticate on the host, and verify its selected workspace. Blueprint validation requires the CLI even when MCP is connected.
- Keep credentials in Hermes's supported secret storage or the CLI login. Do not paste them into prompts, source files or logs. A remote terminal backend does not automatically have the host's CLI, network access or credentials.

## How to Run

After this catalog entry is installed, ask Hermes:

```text
Check my Render connection and tell me which workspace is selected. Do not change anything.
```

Installation fetches `skills/render-mcp/` from [render-oss/skills](https://github.com/render-oss/skills/tree/main/skills/render-mcp), including its reference files. Hermes scans the downloaded content and records the source revision. This catalog entry describes the integration; the full instructions remain in Render's repository.

## Quick Reference

| Need | Surface |
| --- | --- |
| Read the installed instructions and references | `skill_view` |
| Inspect project files | `read_file`, `search_files` |
| Operate Render | Connected Render MCP tools, with their discovered schemas |
| Validate a Blueprint or use a CLI fallback | `terminal`, with the Render CLI available in that execution environment |

## Procedure

1. Read the upstream skill and its access reference with `skill_view`.
2. Check whether the Render MCP tools are already available before proposing another connection.
3. Verify authentication with a read-only operation. Check the selected workspace and ask which one to use if the intended workspace is unclear.
4. Report the connection result and available operations. Never print the key or collect it in chat.

## Pitfalls

- Connected credentials do not authorize every possible change. Preserve the user's requested scope and Hermes's approval rules.
- Check the active workspace before resource operations. Empty results can mean the wrong workspace or insufficient access.
- Tool names and supported operations can change. Use the connected tool schemas and current official documentation.
- Render CLI commands inside a sandbox need their own prerequisites. Keep Render management credentials outside generated-code sandboxes; use the host's MCP connection for management.
- Upstream skill updates are maintained by Render. Review changes through Hermes's normal skill update and scanning flow.

## Verification

- Authentication and the selected workspace are verified through actual tool responses.
- An unavailable connection produces a concrete setup step, without claiming successful access.
