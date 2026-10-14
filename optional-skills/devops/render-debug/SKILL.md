---
name: render-debug
description: Diagnose failed Render deploys and runtime errors.
version: 1.1.0
author: Ojus Save (@ojusave), Render
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [render, cloud, deployment, devops]
    category: devops
    upstream:
      repo: render-oss/skills
      path: skills/render-debug
---

# Render Debug Skill

Investigate a Render service failure using its deploy history, logs, metrics and relevant project files. This entry installs Render's existing debugging skill and does not restart or change services merely by being installed.

## When to Use

- A build or deployment fails.
- A service will not start or fails its health check.
- Logs show runtime errors or resource exhaustion.

## Prerequisites

- A Render account with access to the intended workspace.
- The hosted Render MCP connection, or the Render CLI for operations it supports. For MCP, run `hermes mcp add render --url https://mcp.render.com/mcp` from the host terminal and follow its hidden credential prompt. Use [Render's access instructions](https://render.com/docs/mcp-server) for current requirements.
- For CLI operations, install the [Render CLI](https://render.com/docs/cli), authenticate on the host, and verify its selected workspace. Blueprint validation requires the CLI even when MCP is connected.
- Keep credentials in Hermes's supported secret storage or the CLI login. Do not paste them into prompts, source files or logs. A remote terminal backend does not automatically have the host's CLI, network access or credentials.

## How to Run

After this catalog entry is installed, ask Hermes:

```text
Use render-debug to investigate the specified service and latest failed deploy. Read evidence first; do not restart or change it.
```

Installation fetches `skills/render-debug/` from [render-oss/skills](https://github.com/render-oss/skills/tree/main/skills/render-debug), including its reference files. Hermes scans the downloaded content and records the source revision. This catalog entry describes the integration; the full instructions remain in Render's repository.

## Quick Reference

| Need | Surface |
| --- | --- |
| Read the installed instructions and references | `skill_view` |
| Inspect project files | `read_file`, `search_files` |
| Operate Render | Connected Render MCP tools, with their discovered schemas |
| Validate a Blueprint or use a CLI fallback | `terminal`, with the Render CLI available in that execution environment |

## Procedure

1. Read the upstream skill and relevant references with `skill_view`.
2. Confirm the workspace and exact service before reading logs or metrics. Discover the Render MCP tools and their current input schemas.
3. Collect the relevant deploy state, bounded logs and metrics. Use `read_file` and `search_files` to compare configuration with the observed error.
4. Separate the observed failure from possible causes. Recommend the smallest supported fix and apply it only within the user's authorization.
5. If a change is made, inspect the new deploy and application behavior. Report unresolved errors instead of treating a restart as a fix.

## Pitfalls

- Connected credentials do not authorize every possible change. Preserve the user's requested scope and Hermes's approval rules.
- Check the active workspace before resource operations. Empty results can mean the wrong workspace or insufficient access.
- Tool names and supported operations can change. Use the connected tool schemas and current official documentation.
- Render CLI commands inside a sandbox need their own prerequisites. Keep Render management credentials outside generated-code sandboxes; use the host's MCP connection for management.
- Upstream skill updates are maintained by Render. Review changes through Hermes's normal skill update and scanning flow.

## Verification

- The diagnosis cites the failing deploy or relevant log evidence without exposing credentials.
- Any claimed repair is checked against a new deploy or the original failing behavior.
