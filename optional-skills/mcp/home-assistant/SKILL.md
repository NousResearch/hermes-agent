---
name: home-assistant
description: Control lights, climate, scenes and automations via MCP.
version: 1.0.0
author: Gunnar Beck Nelson (@GChief117)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [HomeAssistant, SmartHome, IoT, HomeAutomation, MCP, OAuth]
    homepage: https://selorahomes.com/docs/selora-ai/
    related_skills: [mcporter]
---

# Home Assistant Skill

Control your smart home from Hermes — lights, switches, climate, scenes, scripts, and automations. Describe new automations in plain English; the skill returns validated Home Assistant YAML with a risk assessment for review before deployment. Auth is OAuth 2.0 via Selora Connect; no long-lived tokens or copy-pasted secrets.

## When to Use

Use this skill when the user wants to:

- inspect or change device state in Home Assistant (lights, switches, climate, covers, media players)
- create or modify automations from a natural-language description
- validate hand-written automation YAML and check its risk profile before deploying
- list, accept, or dismiss proactive suggestions surfaced by the Home Assistant integration
- inspect existing automations, recent chat sessions, or detected behavior patterns

## Prerequisites

1. **Home Assistant** 2025.1+ with the [Selora AI integration](https://selorahomes.com/docs/selora-ai/installation/) installed and enabled.
2. A **Selora Connect** account with the HA installation linked.
3. The **MCP URL** for this installation. Find it in Selora Connect after enabling MCP remote access:
   - Local: `http://homeassistant.local:8123/api/selora_ai/mcp`
   - Remote: `https://mcp-<id>.selorabox.com/api/selora_ai/mcp`

## How to Run

Register the MCP server with Hermes:

```bash
hermes mcp add home-assistant \
  --url https://mcp-<id>.selorabox.com/api/selora_ai/mcp \
  --auth oauth
```

This writes the server into `~/.hermes/config.yaml` under `mcp_servers`:

```yaml
mcp_servers:
  home-assistant:
    url: https://mcp-<id>.selorabox.com/api/selora_ai/mcp
    auth: oauth
```

For LAN-only access, substitute `http://homeassistant.local:8123/api/selora_ai/mcp` (or the HA IP if mDNS is slow).

Confirm the connection with `hermes mcp test home-assistant`. On first tool invocation, Hermes performs an OAuth 2.0 authorization code + PKCE flow against Selora Connect:

1. Hermes reads `.well-known/oauth-authorization-server` from the MCP endpoint.
2. Hermes registers itself dynamically (`POST /oauth/register`).
3. Hermes surfaces an authorization URL — open it, approve access on the Selora Connect consent screen.
4. The browser callback completes the token exchange; tokens refresh silently from then on.

Verify with: *"Get a snapshot of my home"*. The `mcp__home_assistant__selora_get_home_snapshot` tool returns entities grouped by area.

## Quick Reference

Hermes registers MCP tools as `mcp__<server>__<tool>`, so the names below assume the server was
added as `home-assistant` per [How to Run](#how-to-run). Confirm the exact set with
`hermes mcp test home-assistant`.

**Read tools** (no admin role required):

| Tool | Description |
|------|-------------|
| `mcp__home_assistant__selora_get_home_snapshot` | Entity states grouped by area — call this first |
| `mcp__home_assistant__selora_list_automations` | Selora automations with status and risk (filterable) |
| `mcp__home_assistant__selora_get_automation` | Full detail: YAML, versions, risk |
| `mcp__home_assistant__selora_validate_automation` | Validate and risk-assess YAML without creating it |
| `mcp__home_assistant__selora_list_sessions` | Recent chat sessions |
| `mcp__home_assistant__selora_list_patterns` | Detected behavior patterns |
| `mcp__home_assistant__selora_get_pattern` | Pattern detail with linked suggestions |
| `mcp__home_assistant__selora_list_suggestions` | Proactive suggestions with YAML previews |

**Mutating tools** (🔒 require `owner` or `member` role on Selora Connect):

| Tool | Description |
|------|-------------|
| `mcp__home_assistant__selora_chat` 🔒 | Natural-language chat — proposes automations with YAML and risk |
| `mcp__home_assistant__selora_create_automation` 🔒 | Create automation from YAML (disabled by default) |
| `mcp__home_assistant__selora_accept_automation` 🔒 | Enable a pending automation |
| `mcp__home_assistant__selora_delete_automation` 🔒 | Delete permanently |
| `mcp__home_assistant__selora_accept_suggestion` 🔒 | Create automation from a suggestion |
| `mcp__home_assistant__selora_dismiss_suggestion` 🔒 | Dismiss a suggestion |
| `mcp__home_assistant__selora_trigger_scan` 🔒 | Trigger immediate suggestion scan (rate-limited 60s) |

## Procedure

### Inspect the home

1. `mcp__home_assistant__selora_get_home_snapshot` to learn entities and areas.
2. `mcp__home_assistant__selora_list_automations` / `mcp__home_assistant__selora_get_automation` for existing automations.

### Create an automation from YAML

1. `mcp__home_assistant__selora_validate_automation` — check YAML and surface risk.
2. Show normalized YAML + risk; ask the user to confirm.
3. `mcp__home_assistant__selora_create_automation` with `enabled=false`.
4. `mcp__home_assistant__selora_accept_automation` after explicit approval.

### Create an automation from natural language

1. `mcp__home_assistant__selora_chat` — describe the automation; Selora returns YAML + risk.
2. Summarize risk; ask the user to confirm.
3. `mcp__home_assistant__selora_create_automation`, then `mcp__home_assistant__selora_accept_automation`.

### Act on a proactive suggestion

1. `mcp__home_assistant__selora_list_suggestions` (optionally `mcp__home_assistant__selora_trigger_scan` first).
2. Show suggestion details; ask the user to confirm.
3. `mcp__home_assistant__selora_accept_suggestion` or `mcp__home_assistant__selora_dismiss_suggestion`.

## Pitfalls

- **Tool names follow the registered server name.** Registering the server as anything other than
  `home-assistant` changes the prefix (`hermes mcp add ha ...` yields `mcp__ha__selora_*`). Take the
  names from `hermes mcp test <server>` rather than assuming the ones documented here.
- **Never invent entity IDs**. Resolve them from tool output only.
- **Always surface `risk_assessment`** before any mutation. `high` or missing risk requires a second confirmation.
- **Create automations disabled by default**; enable only after explicit approval.
- **Do not skip validation** for externally provided YAML.
- **Cross-device OAuth callback**: the redirect targets `localhost` on the machine running Hermes. If the browser runs on a different machine, the callback can't reach Hermes' listener. As a fallback, ask the user to copy the full callback URL (including `code` and `state`) from the browser and paste it back for manual exchange.
- **`401 Unauthorized` loop** without an auth URL surfaced usually means the OAuth flow is not reaching Hermes' listener. Check gateway logs for `401`, auth URL emission, and MCP startup failures. `npx -y mcp-remote <url>` can isolate whether the endpoint or the client is at fault.
- **Admin tools rejected** when the Selora Connect role is `viewer`. Promote to `member` or `owner`.

## Verification

```bash
# Sanity-check the MCP endpoint exposes OAuth metadata.
curl -sS https://mcp-<id>.selorabox.com/api/selora_ai/mcp/.well-known/oauth-authorization-server | jq .
```

Then ask the agent: *"Get a snapshot of my home"*. Expected: a list of areas with entity states from `mcp__home_assistant__selora_get_home_snapshot`. Failure modes: `401 Unauthorized` with an authorization URL surfaced (expected on first use), connection refused (HA not running or URL wrong), or empty tool list (Selora AI integration not enabled in HA).
