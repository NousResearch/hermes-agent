---
name: render-deploy
description: Deploy applications and prepare Render Blueprints.
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
      path: skills/render-deploy
---

# Render Deploy Skill

Deploy an application to Render using the existing Render MCP tools, or prepare a render.yaml Blueprint for a multi-service application. This entry installs the deployment instructions maintained by Render.

## When to Use

- Deploy an application from its Git repository.
- Prepare or validate a render.yaml Blueprint.
- Choose between direct service creation and a Blueprint.

## Prerequisites

- A Render account with access to the intended workspace.
- The hosted Render MCP connection, or the Render CLI for operations it supports. For MCP, run `hermes mcp add render --url https://mcp.render.com/mcp` from the host terminal and follow its hidden credential prompt. Use [Render's access instructions](https://render.com/docs/mcp-server) for current requirements.
- For CLI operations, install the [Render CLI](https://render.com/docs/cli), authenticate on the host, and verify its selected workspace. Blueprint validation requires the CLI even when MCP is connected.
- Keep credentials in Hermes's supported secret storage or the CLI login. Do not paste them into prompts, source files or logs. A remote terminal backend does not automatically have the host's CLI, network access or credentials.

## How to Run

After this catalog entry is installed, ask Hermes:

```text
Use render-deploy to prepare a paid Render Blueprint for this repository. Validate it, but do not deploy or push changes.
```

Installation fetches `skills/render-deploy/` from [render-oss/skills](https://github.com/render-oss/skills/tree/main/skills/render-deploy), including its reference files. Hermes scans the downloaded content and records the source revision. This catalog entry describes the integration; the full instructions remain in Render's repository.

## Quick Reference

| Need | Surface |
| --- | --- |
| Read the installed instructions and references | `skill_view` |
| Inspect project files | `read_file`, `search_files` |
| Operate Render | Connected Render MCP tools, with their discovered schemas |
| Validate a Blueprint or use a CLI fallback | `terminal`, with the Render CLI available in that execution environment |

## Procedure

1. Read the upstream skill and its referenced deployment guidance with `skill_view`.
2. Inspect the project with `search_files` and `read_file`; establish its build and start commands, required secrets and related services.
3. Verify the workspace and Git source. Use the upstream decision between direct creation and a Blueprint; check current Render documentation before choosing fields or plans.
4. Use `write_file` or `patch` for configuration and `terminal` for the Render CLI validator. Keep secrets out of committed files.
5. Perform only changes the user authorized. For a live deployment, check its final status and application response. For a prepare-only request, return the validated file and remaining steps.

## Pitfalls

- Connected credentials do not authorize every possible change. Preserve the user's requested scope and Hermes's approval rules.
- Check the active workspace before resource operations. Empty results can mean the wrong workspace or insufficient access.
- Tool names and supported operations can change. Use the connected tool schemas and current official documentation.
- Render CLI commands inside a sandbox need their own prerequisites. Keep Render management credentials outside generated-code sandboxes; use the host's MCP connection for management.
- Upstream skill updates are maintained by Render. Review changes through Hermes's normal skill update and scanning flow.

## Verification

- A prepared Blueprint passes the installed Render CLI validator.
- A deployment claim requires a successful deploy and a reachable application; preparation alone is reported as preparation.
