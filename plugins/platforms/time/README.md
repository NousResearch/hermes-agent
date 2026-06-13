# Time (Т‑Банк) adapter

Time is the corporate messenger from Т‑Банк. It is **Slack‑API‑compatible**, so
this adapter is a thin subclass of the Slack adapter (`gateway/platforms/slack.py`)
that points the Slack Web API / Socket Mode transport at Time's endpoints and
reads Time‑specific environment variables.

## Setup (local prototype)

```bash
export TIME_BOT_TOKEN="..."        # bot token (Slack xoxb-equivalent)
export TIME_APP_TOKEN="..."        # Socket Mode app token (Slack xapp-equivalent)
export TIME_API_BASE_URL="https://<time-host>/api/"
export TIME_ALLOW_ALL_USERS=true   # dev only — otherwise set TIME_ALLOWED_USERS
hermes gateway
```

| Env var | Required | Purpose |
|---|---|---|
| `TIME_BOT_TOKEN` | yes | Bot token for Web API calls |
| `TIME_APP_TOKEN` | yes (Socket Mode) | Token for the Socket Mode websocket |
| `TIME_API_BASE_URL` | yes | Time Web API base URL, e.g. `https://time.tbank.ru/api/` |
| `TIME_ALLOWED_USERS` | no | Comma‑separated user IDs allowed to talk to the bot |
| `TIME_ALLOW_ALL_USERS` | no | Truthy = allow everyone (dev only) |
| `TIME_HOME_CHANNEL` | no | Channel for cron / notification delivery |

## Transport

- **Socket Mode (default):** an outbound websocket — **no public webhook URL
  required**. Use this when corporate policy forbids inbound webhooks.
- **Events / webhook:** supported via the inherited Slack Events path when a
  signing secret and a public endpoint are configured.

## Security — why this is corporate‑safe

Time sessions are locked down so the agent **physically cannot** perform a
dangerous real‑world action. Two enforcement layers:

1. **`corp_safe` toolset profile** (`platform_toolsets: { time: [corp_safe] }`).
   Time only gets: chat/messaging, skills, memory, session recall, web search,
   vision, image‑gen, planning (todo), clarify, and cron scheduling.

2. **Code‑level hard‑deny** (`CORP_RESTRICTED_PLATFORMS` in
   `hermes_cli/tools_config.py`). Even if an admin widens Time's config, the
   platform unconditionally strips every toolset in `CORP_DANGEROUS_TOOLSETS`
   — `terminal`, `file` (write/patch), `code_execution`, `computer_use`,
   `browser`, `homeassistant`, `delegation`, `kanban`, `discord_admin` — and
   any composite (e.g. `coding`) that bundles them. The agent never receives
   their tool *schemas*, so it cannot call them even indirectly via the
   `tool_search` bridge.

3. **Cron is covered too.** A Time agent can schedule automations (daily
   reports etc.), but cron‑spawned agents have the same dangerous toolsets
   denied (`cron/scheduler.py::_resolve_cron_disabled_toolsets`), so a
   scheduled job cannot be used to escalate into shell/file/code execution.

What stays intact (core functionality): conversation, the self‑improving
skills loop, persistent memory, cross‑session search, web/vision, and
scheduled automations — just without any physically dangerous tools.

### Known residual considerations (prototype)

These are out of scope for the prototype but should be reviewed before a
production rollout:

- **Custom MCP servers pass through the hard-deny by design.** The hard-deny
  filters built-in toolsets; an MCP server name listed in `platform_toolsets.time`
  is not inspected. Only add MCP servers to Time whose tools are known-safe.
- **`HERMES_KANBAN_TASK` worker context.** If Time messages are ever routed
  through the kanban dispatcher (which sets `HERMES_KANBAN_TASK`), the kanban
  toolset can be re-added downstream of the gateway hard-deny. Don't run Time
  sessions as kanban workers. (`kanban` is denied in the normal gateway and
  cron paths.)
- **System-prompt platform note.** Time relies on its registry `platform_hint`
  rather than the hardcoded Slack note in `gateway/session.py`.

See `docs/superpowers/specs/2026-06-12-time-messenger-safe-integration-design.md`
for the full design and threat model.
