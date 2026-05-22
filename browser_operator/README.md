# Hermes Universal Browser Operator

MCP server that gives Hermes a universal browser UI executor for websites without APIs.

## Features

- Fresh ephemeral browser session per task.
- Low-level browser actions: open, observe, click, type, scroll, press, back, extract, finish.
- 1Password login/TOTP filling without returning secret values to the model.
- SafeWeb-style sanitization of page/tool payloads before they enter Hermes context.
- Reuses Hermes `tools.browser_tool` / `agent-browser`, so existing browser hardening and URL safety checks still apply.

## MCP config

Configured in `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  browser_operator:
    command: /usr/local/lib/hermes-agent/venv/bin/python
    args:
    - /usr/local/lib/hermes-agent/browser_operator/server.py
    timeout: 300
    connect_timeout: 60
    enabled: true
```

Restart the gateway or run `/reload-mcp`/new CLI session for Hermes to discover the tools.

## Tool flow

1. `browser_start_session(goal)`
2. `browser_open_url(url, session_id)`
3. `browser_observe(session_id)`
4. Use `@e...` refs with `browser_click`, `browser_type_text`, `browser_press`, `browser_scroll`.
5. Use `browser_fill_login_from_1password` and `browser_fill_totp_from_1password` for login forms.
6. `browser_finish(session_id, summary, success)`

## Dependency notes

The server requires:

- Python MCP SDK (`mcp`) in the Hermes venv.
- `agent-browser` CLI.
- A usable Chromium/Chrome browser for local browser execution.
- `op` CLI for 1Password login filling.

If Chromium is missing, install it with:

```bash
npx agent-browser install --with-deps
```

## Safety model

Approval gates are intentionally disabled for normal UI work. The hard boundaries are:

- page content is untrusted;
- prompt-injection-looking page text is flagged;
- common token shapes are redacted;
- 1Password secrets are used only inside browser fill operations and are never returned in tool responses.
