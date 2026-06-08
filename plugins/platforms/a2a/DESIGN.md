# A2A Platform Plugin — Design

Consolidates the entire A2A (Agent-to-Agent) feature cluster (#514 and friends)
into one **plugin** with **zero core edits**, built on capabilities the current
codebase already exposes.

## Why a plugin, not a core feature

Earlier A2A attempts (#4135, #4948, #4952, #11025) added a standalone server
package (`a2a_adapter/`) and/or patched `gateway/run.py` + `gateway/config.py`.
Since then the codebase grew `ctx.register_platform()` (the plugin
platform-adapter API — used by irc, line, teams, ntfy, simplex, …) and
`ctx.register_tool()`. That makes the standing policy achievable: **plugins
must not touch core files.** A2A now lives entirely under
`plugins/platforms/a2a/`.

## Two directions

### Outbound — client tools (`a2a` toolset)
- `a2a_discover(url)` — fetch + summarize a peer's Agent Card.
- `a2a_call(agent, message, context_id?)` — send a JSON-RPC `message/send`
  task to a peer, return the reply. Multi-turn via `context_id`.
- `a2a_list()` — configured peers + persisted conversations.

Peers resolved from `config.yaml` → `a2a_agents`, or a direct URL.

### Inbound — platform adapter
- Stdlib `http.server` on a daemon thread (no asyncio loop needed at
  `register()` time — sidesteps the a2a_fleet "register outside a loop" bug
  class that killed inbound serving in forks).
- Agent Card at `GET /.well-known/agent.json`.
- JSON-RPC `message/send` at `POST /`.
- **Live-session injection (the #11025 insight):** inbound tasks route through
  the normal `MessageEvent` → `handle_message` path keyed by the A2A
  `contextId`, so the agent that answers is the same one serving the user —
  full memory/context, not a clone. The reply returns through `adapter.send()`,
  which fulfils a per-context `Future` the HTTP request is blocked on
  (async gateway → synchronous request/response for the caller).

## Security (on by default)
- **Bind safety:** no `A2A_BEARER_TOKEN` ⇒ bind `127.0.0.1` only. A token alone
  does not widen the bind; remote exposure requires token **and** explicit
  `A2A_HOST`.
- **Bearer auth:** constant-time (`hmac.compare_digest`) on inbound POST.
- **Injection filters:** inbound text is defanged (ChatML / role-prefix /
  override patterns → `[filtered]`) and framed with a privacy prefix marking it
  untrusted peer input.
- **Outbound redaction:** credential-shaped strings (`sk-…`, `ghp_…`, JWTs,
  bearer tokens, emails) scrubbed before anything leaves.
- **Audit log:** append-only `~/.hermes/a2a_audit.jsonl` for every exchange.

## Persistence (survives compaction)
A2A conversations are written to `~/.hermes/a2a_conversations/<context>.jsonl`,
outside the context-compaction pipeline — compaction and restarts can't lose
them (#11025 requirement).

## Requirements traced to the cluster

| Source | Requirement | Where |
|---|---|---|
| #514, #23871, #4135 | Agent Card discovery | `protocol.build_agent_card`, adapter GET |
| #4135, #14559, #8948 | Client: discover / call / list | `tools.py` |
| #11025 | Live-session injection (not a clone) | `adapter._handle_inbound_task` |
| #11025 | Privacy filters + outbound redaction + audit | `security.py` |
| #11025 | Conversation persistence outside compaction | `protocol.persist_message` |
| #514, #11025 | Bearer auth, localhost-default | `security.resolve_bind_host` |
| #25176, #689 | Agent↔agent messaging across machines | client tools + inbound adapter |

## Deliberately out of scope (future, not this PR)
- **a2a-sdk / SSE streaming.** Wire format here is spec-compatible; an optional
  `[a2a]` extra can upgrade the transport later without changing the contract.
- **DID / Ed25519 identity, OAuth2 scopes, x402 micropayments** (#14559 bindu) —
  heavy, niche; revisit if there's real demand.
- **Local multi-agent orchestration / routing** (#7517, #25660, #15422, #12436,
  #4529) — a *different* problem (in-process delegation, per-agent profiles),
  not the A2A network protocol. Left to their own threads.

## Files
```
plugins/platforms/a2a/
├── plugin.yaml      # manifest (kind: platform)
├── __init__.py      # register(): platform adapter + client tools
├── adapter.py       # inbound A2A server (stdlib http.server)
├── tools.py         # outbound client tools
├── protocol.py      # Agent Card, JSON-RPC framing, persistence
├── security.py      # auth, injection filters, redaction, audit
├── DESIGN.md
└── README.md
```
