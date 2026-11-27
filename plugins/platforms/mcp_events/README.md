# MCP Events receiver (DRAFT)

Subscribe your Hermes agent to event streams from MCP event emitters and wake it
on signed deliveries — no cron, no human message.

## Quick start

1. `hermes gateway setup` → MCP Events → generate a webhook secret (or set
   `MCP_EVENTS_WEBHOOK_SECRET` yourself: `whsec_` + base64 of 24–64 random bytes).
2. For remote emitters, add to `~/.hermes/config.yaml`:
   ```yaml
   mcp_events:
     enabled: true
     port: 9901
     public_base_url: https://your-host.example.com
   ```
   Localhost-only mode (no secret) works for local emitters with no config.
3. (Recommended) Name your emitters so the agent works with names, not URLs —
   and so emitter credentials never reach the model or the logs:
   ```yaml
   mcp_events:
     emitters:
       mcp-events-bridge:            # the name the tools take
         url: http://127.0.0.1:8080/mcp
   ```
   A URL that carries a secret (a capability URL — common for MCP servers
   without OAuth) belongs in `.env` instead, e.g.
   `MCP_EVENTS_EMITTER_MCP_EVENTS_BRIDGE_URL=...`; a bearer-token emitter uses
   `MCP_EVENTS_EMITTER_<NAME>_HEADERS='{"Authorization": "Bearer ..."}'`
   (JSON) or a plain `headers:` map for non-secret values. Env wins over
   config.yaml.
4. In any session:
   - `mcp_events_list <emitter>` (a configured name, or an emitter URL) — see
     what events the emitter offers.
   - `mcp_events_subscribe <emitter> <event>` — subscribe; deliveries wake the
     agent in a per-subscription conversation.
   - `mcp_events_subscriptions` / `mcp_events_unsubscribe <id>` — manage them.

## How it works

The plugin registers two things: a platform adapter (the webhook receiver,
`POST /mcp/events/webhook/<id>`) and four client tools. When a signed delivery
arrives it is verified (Standard Webhooks HMAC-SHA256, timestamp freshness),
deduped, rate-limited, defanged, framed as untrusted external input, and routed
into the agent's live session — the same agent serving the user wakes up, with
full memory and context.

Security defaults: no secret ⇒ localhost-only bind; emitter allow-list
(`mcp_events.trusted_emitters`) with fail-closed remote exposure; SSRF-guarded
URLs; per-emitter rate limits and per-subscription storm guards; append-only
audit log at `~/.hermes/mcp_events_audit.jsonl`.

See `DESIGN.md` for the protocol notes, the security model, and what's
deliberately out of scope.
