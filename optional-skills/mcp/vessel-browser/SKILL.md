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

Vessel is the user's persistent, human-visible browser that Hermes controls over MCP. Think of it like a tool that's usually already running — your job is to **use** it, not set it up.

## On skill load — readiness check (quiet, fast, no narration)

When this skill is invoked, run this single command silently:

```bash
~/.local/bin/vessel-browser-status --json
```

**Interpret the result and act — do not narrate the check to the user:**

- **`mcp_reachable` is `true`** → Vessel is ready. Say nothing about setup. If the user gave you a task, start doing it with Vessel MCP tools immediately. If no task was given, simply let the user know Vessel is available and ask what they'd like to do — e.g. "Vessel is ready — what would you like me to browse?"
- **Vessel is installed but not running** → run `~/.local/bin/vessel-browser-launch`. On success, briefly confirm "Vessel is up" and proceed to the task or ask what to do. On failure, see "Troubleshooting" below.
- **Vessel is not installed** → tell the user and stop.

**Critical rules for the readiness check:**
- Do NOT run `vessel-browser-update`, `vessel-browser-mcp`, `curl`, or any other command when `mcp_reachable` is `true`.
- Do NOT read, grep, or inspect Hermes config files as part of the readiness check.
- Do NOT describe the readiness check process to the user. No "Let me check the status" or "I'll verify the setup." Just run the command, read the result, and act.
- The entire readiness check should be invisible to the user in the success case. They should only see you either starting their task or asking what they want to do.

## Using Vessel

Once Vessel is ready, use its MCP tools directly. Here's how to get the most out of them:

- Prefer `summary` on large pages before asking for full text.
- Prefer `results_only` on search and listing pages.
- Prefer `visible_only` when dealing with overlays, consent banners, or viewport-specific actions.
- Use Vessel bookmarks and named sessions instead of repeatedly rediscovering the same pages.
- Treat Vessel as the browser of record for long-running workflows; avoid mixing it with a separate automation browser unless the user explicitly wants that.

## Troubleshooting

Only consult this section if launch fails, Vessel MCP tools are missing from your tool list, or the user asks for diagnostics.

### Vessel MCP tools not in tool list

Check the default Hermes user config file (and only that file) for a `vessel` MCP server entry:

```bash
~/.local/bin/vessel-browser-mcp --format hermes
```

If the entry is missing, show the snippet and tell the user to add it and run `/reload-mcp`. Do not edit the config unless explicitly asked.

### Launch failed

- The smart launcher prefers a healthy source install and falls back to the newest local AppImage.
- Do not narrate fallback mechanics (AppImage, `--no-sandbox`, etc.) unless asked.
- Only use the raw source launcher (`~/.local/bin/vessel-browser`) when the user explicitly wants the source build.

### Updates

Only relevant when the user asks to update or when status explicitly recommends one.

```bash
~/.local/bin/vessel-browser-update --check --json
```

- If `dirty: true` or `status: "fetch-failed"`, do not block — note it and move on.
- Only run the full updater when `can_apply_update: true` or the user explicitly asks.

## Pitfalls

- Vessel MCP is local-only; Hermes must run on the same machine unless the endpoint is explicitly tunneled.
- The MCP port may differ from the default. Always prefer `vessel-browser-mcp --format url` over guessing.
- Do not scan the entire home directory to locate Vessel. Use `~/.local/bin`, `~/.local/share/vessel-browser`, `~/.config/vessel/vessel-settings.json`, and `vessel-browser-status`.
- Trust the configured endpoint and its reachability more than loose `pgrep` output.
- If `vessel-browser-status` recommends `appimage`, use the smart launcher instead of trying to repair Electron sandbox permissions inline.

## References

- Vessel README: https://github.com/unmodeled-tyler/quanta-vessel-browser
- Hermes MCP setup: `native-mcp` skill
