---
name: mcp-google-orchestrator
description: "Install, check, or diagnose Gmail/Calendar/Drive/Sheets MCP servers in Hermes."
version: 1.0.0
author: Anton <tom475578-coder> (with Hermes Agent)
license: MIT
platforms: [windows, macos, linux]
---

# mcp-google-orchestrator

Install, configure, and diagnose the four Google MCP servers (Gmail, Calendar,
Drive, Sheets) that Hermes wraps through stdio.

## Prerequisites

- `hermes` CLI on `PATH`.
- For Google OAuth flows: `~/.config/gmail-mcp/gcp-oauth.keys.json` (and per
  server equivalents under `~/.calendar-mcp/`, `~/.google-workspace-mcp/`).
- Disk write access to `HERMES_HOME`. Detect with `hermes config path` — do
  not hardcode it (Windows: `%LOCALAPPDATA%/hermes` ; macOS:
  `~/Library/Application Support/hermes` ; Linux: `$XDG_DATA_HOME/hermes`).

## When to use

- Adding or repairing a Google MCP server entry (`gmail`, `calendar`,
  `google-workspace` for Drive/Sheets) in `HERMES_HOME/config.yaml`.
- Diagnosing Gmail/Calendar/Drive/Sheets auth errors, missing scopes, or stale
  subprocess failures (e.g. `ClosedResourceError`).
- Verifying a freshly installed server exposes the expected tools before
  wiring it into a skill.

## Architecture (in-repo)

| Goal                           | Hermes-native command                                                                  |
| ------------------------------ | -------------------------------------------------------------------------------------- |
| Add Gmail MCP                  | `hermes mcp add gmail --command node --args <wrapper>`                                 |
| Add Calendar MCP               | `hermes mcp add calendar --command node --args <wrapper>`                              |
| Add Drive + Sheets MCP         | `hermes mcp add google-workspace --command node --args <wrapper> --env GD_TOOLS`       |
| Diagnose stale subprocess      | `hermes mcp restart <name>` then re-call the tool                                      |
| Smoke-test after install       | Read-only tool call against the server (no writes)                                    |

> **Removed:** the Anton-specific `~/Downloads/MCP/` playbook (5 scripts,
> backup convention, calibration counts) is out of scope. It is untracked,
> not reproducible, and has been removed from this skill.

## Related skills

- `mcp-server-setup` — generic MCP installation patterns.
- `mcp-server-troubleshooting` — diagnose `ClosedResourceError`,
  `OAuth invalid_grant`, stale subprocesses.
- `hermes-agent` — `hermes config`, `hermes mcp` CLI reference.

## Common pitfalls

- **Don't hardcode `HERMES_HOME`.** Use `hermes config path` or read
  `HERMES_HOME` from the environment.
- **Don't retry on `403 scope`.** Re-run the OAuth flow with the missing
  scope instead.
- **A passing `node server.js --help` is not a working MCP.** Use the
  verification ladder: binary → CLI → MCP → read-only tool call.
