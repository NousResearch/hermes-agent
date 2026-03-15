---
name: vessel-browser
description: Use Vessel Browser as a persistent, human-visible browser runtime over MCP. Connect Hermes to a local Vessel instance for navigation, extraction, bookmarks, sessions, and supervised browsing workflows.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [browser, mcp, vessel, web-automation, supervision]
    related_skills: [native-mcp]
---

# Vessel Browser

Use Vessel as Hermes Agent's persistent browser runtime over MCP. Vessel keeps browser state alive across tasks, exposes browser tools over a local MCP endpoint, and gives the human a visible browser UI for supervision.

## Quick Setup

```bash
# Inspect install and launch health first
~/.local/bin/vessel-browser-status --json

# Check for updates, but do not block launch on a dirty checkout or a temporary fetch failure
~/.local/bin/vessel-browser-update --check --json

# Launch Vessel using the best available local install
~/.local/bin/vessel-browser-launch

# Print the Hermes-ready MCP config snippet
~/.local/bin/vessel-browser-mcp --format hermes

# Reload MCP after adding the emitted snippet to your Hermes MCP server configuration
/reload-mcp
```

## When to Use

Use this skill when you need to:
- browse the real web through Vessel instead of a stateless browser session
- preserve login state, cookies, bookmarks, and named browser sessions
- inspect pages with `summary`, `results_only`, `visible_only`, `forms_only`, or full extraction modes
- keep a human-visible browser open while Hermes works through MCP

## Requirements

- Linux
- Vessel installed locally
- Hermes MCP support enabled (`native-mcp` skill or built-in MCP support)
- Vessel running before Hermes tries to call Vessel MCP tools

## Setup

### 1. Inspect deterministic local state first

```bash
~/.local/bin/vessel-browser-status --json
```

Prefer this helper over ad-hoc process checks. It reports:
- whether the source install exists at `~/.local/share/vessel-browser`
- whether helpers exist in `~/.local/bin`
- whether `~/.local/bin` is on `PATH`
- the configured MCP port from `~/.config/vessel/vessel-settings.json`
- whether the MCP endpoint is reachable
- whether the source launcher looks healthy
- whether launch should prefer `source`, `appimage`, or `already-running`

### 2. Check updates without blocking launch

```bash
~/.local/bin/vessel-browser-update --check --json
```

Policy:
- if the updater reports `dirty: true`, do not interrupt setup; launch as-is and note that updating was deferred
- if the updater reports `status: "fetch-failed"`, do not interrupt setup; launch as-is and suggest trying the update later
- only run the full updater when `can_apply_update: true` or the user explicitly asks to refresh the source install

### 3. Add Vessel to Hermes MCP config

Preferred path:

```bash
~/.local/bin/vessel-browser-mcp --format hermes
```

Canonical config target:

- the default Hermes user config file in the user's home directory

Only inspect that single default Hermes user config file when checking whether Vessel is already configured. Do not probe multiple alternate config locations in the normal setup flow.

Do not edit this config file unless the user explicitly asks you to do so. By default, inspect it and report whether the `vessel` MCP server entry is already present.

### 4. Launch Vessel

Preferred path:

```bash
~/.local/bin/vessel-browser-launch
```

Use the smart launcher by default. It prefers a healthy source install and falls back to the newest local AppImage when the source install is likely blocked by Electron sandbox permissions.

Reporting rule:
- if `vessel-browser-launch` returns success, report only the successful outcome, for example `Vessel launched successfully and MCP is reachable`
- do not narrate internal fallback mechanics like source-launch failure, AppImage fallback, `--no-sandbox`, or extract-and-run unless launch ultimately fails or the user explicitly asks for diagnostics
- do not mention the exact launch transport in the normal success path, for example `AppImage`, `source build`, or `--no-sandbox`, unless the user explicitly asks how Vessel was launched
- treat the launcher as an opaque helper that already chose the best local path
- prefer short prose over tables, checklists, or status dashboards in the normal success path
- when setup is complete, give a compact ready-state summary instead of a formatted report card

Only use the raw source launcher below when the user explicitly wants the source build even if AppImage fallback is available:

```bash
~/.local/bin/vessel-browser
```

### 5. Reload MCP servers

After updating your Hermes MCP server configuration, restart Hermes or run:

```text
/reload-mcp
```

## Recommended Launch Procedure

When using Vessel regularly:

1. Run `~/.local/bin/vessel-browser-status --json` first.
2. If `vessel-browser-update` exists, run `~/.local/bin/vessel-browser-update --check --json`.
3. If the update check reports `can_apply_update: true` and the user wants the freshest source install, run `~/.local/bin/vessel-browser-update`.
4. Otherwise, launch with `~/.local/bin/vessel-browser-launch`.
5. Confirm the MCP endpoint with `~/.local/bin/vessel-browser-mcp --format url`.
6. If status already shows Vessel running with MCP reachable, do not rerun launch. Move directly to config inspection and result reporting.
7. Inspect only the default Hermes user config file to check whether the `vessel` MCP server entry is already present.
8. If the entry is missing, provide the emitted snippet and tell the user to reload MCP. Do not write the config file unless explicitly asked.
9. If the entry is already present, say so directly and avoid asking whether they want you to check.

Do not ask the user to choose between launching and updating just because the source checkout is dirty. Dirty source state is normal during local development.
Do not surface internal launch fallback details in the normal success path. The user primarily needs launch status, endpoint status, and the MCP config snippet.
Do not end with an unnecessary question like asking whether to inspect the MCP config if you can inspect it directly.
Do not automatically append to or rewrite Hermes MCP configuration during the normal setup flow.
Do not use markdown tables for the normal ready-state response.
Preferred ready-state shape:
- one short paragraph confirming Vessel is running and MCP is reachable
- one short sentence confirming whether Hermes is already configured or providing the snippet if it is not
- optionally one short sentence about deferred updates if relevant
- then stop, unless the user asked for a next action

## Verification

Smoke test the local MCP endpoint:

```bash
~/.local/bin/vessel-browser-mcp --format url
curl -sS -o /dev/null -w '%{http_code}\n' "$(~/.local/bin/vessel-browser-mcp --format url)"
```

Then ask Hermes to use one Vessel MCP tool after reload, for example a simple page read or tab listing.

## Usage Guidance

- Prefer `summary` on large pages before asking for full text.
- Prefer `results_only` on search and listing pages.
- Prefer `visible_only` when dealing with overlays, consent banners, or viewport-specific actions.
- Use Vessel bookmarks and named sessions instead of repeatedly rediscovering the same pages.
- Treat Vessel as the browser of record for long-running workflows; avoid mixing it with a separate automation browser unless the user explicitly wants that.

## Update Policy

`vessel-browser-update` applies only to source installs created through Vessel's installer. It fetches the configured branch, installs dependencies, and rebuilds the local app.

If the user installed Vessel via AppImage instead of source, the update helper may not exist. In that case, tell the user to download the latest AppImage release manually.

## Pitfalls

- Vessel MCP is local-only; Hermes must run on the same machine unless the endpoint is explicitly tunneled.
- The MCP port may differ from the default if the user changed it in Vessel Settings. Always prefer `vessel-browser-mcp --format url` over guessing.
- Hermes will not see Vessel tools until the emitted snippet is added to the Hermes MCP server configuration and MCP is reloaded.
- If Vessel is down, Hermes MCP tool calls will fail until the browser is running again.
- Do not scan the entire home directory to locate Vessel. Prefer `~/.local/bin`, `~/.local/share/vessel-browser`, `~/.config/vessel/vessel-settings.json`, and `vessel-browser-status`.
- A running process match alone is not enough. Trust the configured endpoint and its reachability more than loose `pgrep` output.
- If `vessel-browser-status` recommends `appimage`, use the smart launcher instead of trying to repair Electron sandbox permissions inline during routine setup.

## References

- Vessel README: https://github.com/unmodeled-tyler/quanta-vessel-browser
- Hermes MCP setup: `native-mcp` skill
