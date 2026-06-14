# Time (Т‑Банк) adapter

Time is the corporate messenger from Т‑Банк. It is **Mattermost-based** — its
REST API is the Mattermost v4 API (`/api/v4/`). This adapter therefore
subclasses the bundled **Mattermost** adapter (`plugins/platforms/mattermost`)
and maps Time-specific env vars onto it, while carrying its own
`Platform("time")` identity so the corporate security profile applies.

> Time blocks the Mattermost **WebSocket** (the proxy returns HTTP 403 on
> `/api/v4/websocket` for bot tokens). Inbound messages therefore arrive via
> **REST long-polling** (`TimeAdapter._poll_loop`), the policy-friendly
> fallback. Outbound sending uses the normal Mattermost REST API.

## Setup (local prototype)

```bash
export TIME_BOT_TOKEN="..."                  # Mattermost bot/PAT token
export TIME_API_BASE_URL="https://time.tbank.ru"
export TIME_ALLOWED_USERS="<user-id>"        # restrict who can talk to the bot
hermes plugins enable time-platform
hermes gateway
```

| Env var | Required | Purpose |
|---|---|---|
| `TIME_BOT_TOKEN` | yes | Mattermost bot/personal-access token |
| `TIME_API_BASE_URL` | yes | Time server base URL, e.g. `https://time.tbank.ru` |
| `TIME_ALLOWED_USERS` | no | Comma-separated user IDs allowed to talk to the bot |
| `TIME_ALLOW_ALL_USERS` | no | Truthy = allow everyone (dev only) |
| `TIME_HOME_CHANNEL` | no | Channel ID for cron / notification delivery |
| `TIME_POLL_INTERVAL` | no | REST poll interval, seconds (default 3) |

## Security — why this is corporate‑safe

Time sessions are locked down so the agent **physically cannot** perform a
dangerous real‑world action. Two enforcement layers, both independent of the
wire transport:

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

3. **Cron is covered too.** A Time agent can schedule automations, but
   cron‑spawned agents have the same dangerous toolsets denied
   (`cron/scheduler.py::_resolve_cron_disabled_toolsets`).

Core functionality stays intact: conversation, the self‑improving skills loop,
persistent memory, cross‑session search, web/vision, and scheduled
automations — just without any physically dangerous tools.

### Known residual considerations (prototype)

- **Custom MCP servers pass through the hard-deny by design** — only add
  MCP servers to Time whose tools are known-safe.
- **`HERMES_KANBAN_TASK` worker context** — don't run Time sessions as kanban
  workers (`kanban` is denied in the normal gateway and cron paths).
- **Token storage** — keep `TIME_BOT_TOKEN` in `~/.hermes/.env` (chmod 600),
  never in code or `config.yaml`.

See `docs/superpowers/specs/2026-06-12-time-messenger-safe-integration-design.md`
for the full design and threat model.
