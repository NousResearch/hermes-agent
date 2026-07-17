# Multi-Workspace Slack Socket Mode

Hermes supports two Slack credential modes:

- `SLACK_BOT_TOKEN` plus `SLACK_APP_TOKEN`: backward-compatible single
  Socket Mode connection. Comma-separated bot tokens and `slack_tokens.json`
  remain send-capable legacy paths for extra workspaces.
- `~/.hermes/slack_accounts.json`: true multi-workspace *receive* support.
  Each account entry has its own bot token and app token, so Hermes opens one
  independent Socket Mode connection per Slack workspace.

Example (`~/.hermes/slack_accounts.json`):

```json
[
  { "name": "engineering", "bot_token": "xoxb-...", "app_token": "xapp-..." },
  { "name": "partner",     "bot_token": "xoxb-...", "app_token": "xapp-..." }
]
```

## Why one connection per account

Slack Socket Mode distributes events for a single app token across all
competing websocket connections that present that token. A workspace's
events therefore only arrive reliably on that workspace's own app-token
socket — a single shared connection cannot receive every workspace's events.
Each account opens its own `AsyncApp` + `AsyncSocketModeHandler`, and each app
token is protected by its own gateway lock so two gateway processes cannot
both claim one token and silently split its event stream.

## Implementation notes (bundled plugin)

The adapter lives at `plugins/platforms/slack/adapter.py`.

- The **primary** account keeps the legacy `self._app` / `self._handler` /
  `self._socket_mode_task` fields and lock, so all existing single-workspace
  behavior and tests are unchanged.
- **Extra** accounts each get an entry in `self._extra_connections`
  (`{name, app_token, bot_token, app, handler, task}`) with their own scoped
  `slack-app-token` lock.
- Handler registration is shared via `_register_app_handlers(app)`, so every
  account's app dispatches the identical event/command/action set — including
  `file_shared` → `_handle_slack_file_shared`, the assistant-thread lifecycle
  handlers, slash commands, and plugin action handlers.
- The existing Socket Mode **watchdog** now monitors the extra connections
  too: a dead task or a disconnected transport on any account triggers a
  per-account reconnect (`_restart_extra_connection`) without disturbing the
  others.

## Channel → workspace routing across restarts

Inbound events record the owning `team_id` for each channel (`scope_id` on the
message source, and the in-memory `_channel_team` map). That map is persisted
to `~/.hermes/slack_channel_teams.json` and reloaded on connect, so outbound
sends after a restart use the correct workspace client before any new inbound
event re-teaches the mapping. Session and routing keys are scoped by workspace
for Slack only; Discord and other platforms keep their existing key format.
