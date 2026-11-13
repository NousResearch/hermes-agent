# A2A — Agent-to-Agent protocol for Hermes

Talk to other agents, and let other agents talk to you, over the open
[A2A protocol](https://a2a-protocol.org) **v1.0**. Works with any A2A-compliant
peer (another Hermes, LangChain, CrewAI, Google ADK, OpenClaw, …). Stdlib only —
no `a2a-sdk` dependency.

## Enable

```bash
hermes gateway setup      # pick A2A, or:
```

```yaml
# ~/.hermes/config.yaml
gateway:
  platforms:
    a2a:
      enabled: true
      extra:
        port: 9900
        input_wait_timeout_seconds: 86400
        task_retention_seconds: 86400

# peers you want to call (outbound):
a2a_agents:
  researcher:
    url: "http://localhost:9999"
    auth: { type: bearer, token: "sk-..." }
    timeout: 120
    capabilities: [web_search, research]
```

## Outbound — call other agents

The agent gets five tools:

- `a2a_discover(url)` — what can this agent do?
- `a2a_call(agent, message, context_id?)` — send it a task, get the reply.
- `a2a_list()` — configured peers, saved conversations, metrics.
- `a2a_history(context_id)` — recall a saved A2A conversation.
- `a2a_orchestrate(capability, message, mode?)` — fan-out a task to every
  peer advertising a capability (`all` / `first` / `best`).

## Inbound — be callable

When the `a2a` platform is enabled, Hermes serves a v1.0 Agent Card at
`http://<host>:<port>/.well-known/agent-card.json` (the legacy
`/.well-known/agent.json` path is also answered for pre-1.0 clients) and
accepts JSON-RPC
`message/send`, `message/stream` (SSE), `tasks/get|list|cancel|subscribe`,
and push notification configs (inline or via
`tasks/pushNotificationConfig/create`). Incoming tasks are injected into your
**live** agent session — the same agent that's talking to you, with full
memory — and the reply is returned over A2A. Completed tasks stay queryable
via `tasks/get`.

### Send, retrieve, and continue tasks

Send a task and receive its ID immediately:

```bash
curl http://localhost:9900/ -H 'Content-Type: application/json' -d '{
  "jsonrpc": "2.0", "id": 1, "method": "SendMessage",
  "params": {
    "message": {"role": "ROLE_USER", "messageId": "request-1", "parts": [{"text": "Research this topic"}]},
    "configuration": {"returnImmediately": true}
  }
}'
```

Save `result.task.id` and `result.task.contextId`. Use the returned task ID to
check progress or retrieve the result:

```bash
curl http://localhost:9900/ -H 'Content-Type: application/json' -d '{
  "jsonrpc": "2.0", "id": 2, "method": "GetTask",
  "params": {"id": "<task-id>"}
}'
```

If the task asks for input (`TASK_STATE_INPUT_REQUIRED` or
`TASK_STATE_AUTH_REQUIRED`), answer with its `taskId` and a fresh `messageId`:

```bash
curl http://localhost:9900/ -H 'Content-Type: application/json' -d '{
  "jsonrpc": "2.0", "id": 3, "method": "SendMessage",
  "params": {"message": {
    "role": "ROLE_USER", "messageId": "request-2", "taskId": "<task-id>",
    "parts": [{"text": "Focus on the Python implementation"}]
  }}
}'
```

- Without `returnImmediately`, `SendMessage` waits for a result or request for input.
- Use `SendStreamingMessage` for live updates; save the IDs from its first Task event.
  A disconnected client does not cancel accepted work. Reconnect with
  `SubscribeToTask` or retrieve the result with `GetTask`.
- A `taskId` continues that interrupted task; `contextId` alone starts a new task
  in the conversation. If both are sent, they must match. Finished tasks cannot restart.
- Use `CancelTask` with `params.id` to cancel a task; this does not guarantee that
  already-running tools stop. Retry requests refused for temporary capacity limits.
- `extra.input_wait_timeout_seconds` cancels unanswered input/auth waits (default
  24 hours). Polling does not extend this deadline.
- `extra.task_retention_seconds` removes terminal results after 24 hours by default;
  the 500-terminal-task cap may remove them sooner. Task IDs do not survive a Hermes restart.
  Neither setting limits healthy running work.

For authenticated servers, add `-H 'Authorization: Bearer <token>'` to each request.
Forwarding to another Hermes profile has additional recovery limits; see
[DESIGN.md](DESIGN.md#forwarded-owner-link-failure-limitation).

## Security

- **No token ⇒ localhost only.** The server binds `127.0.0.1` and refuses to
  widen unless you configure a token *and* set `A2A_HOST`.
  Without a token it also refuses browser-originated POSTs (`Origin`,
  cross-site `Sec-Fetch-Site`, a missing or non-JSON `Content-Type`).
- **No token ⇒ loopback `Host` only.** Every route refuses a `Host` that is
  not a loopback name (`localhost`, `127.x.x.x`, `[::1]`) or the
  `A2A_PUBLIC_URL` host, so a DNS-rebound page cannot reach it. Only set
  `A2A_PUBLIC_URL` to a name you control.
- **Per-peer tokens**: `A2A_PEER_TOKENS="alice:tok1,bob:tok2"` gives each
  remote agent its own credential; that authenticated name (never anything
  in the request body) drives rate limiting, trust, and audit.
- Inbound text — including `/`-prefixed text — is run through
  prompt-injection filters and framed as untrusted peer input; remote peers
  cannot invoke operator slash commands.
- Outbound text is scrubbed of credential-shaped strings.
- Push callbacks are SSRF-guarded and HMAC-SHA256 signed (`X-A2A-Signature`).
- Every exchange is logged to `~/.hermes/a2a_audit.jsonl`.
- Conversations persist to `~/.hermes/a2a_conversations/` — they survive context
  compaction and restarts (`a2a_history` recalls them).

## Env vars

| Var | Default | Meaning |
|---|---|---|
| `A2A_PEER_TOKENS` | _(unset)_ | Per-peer credentials `name:token,…` (preferred). |
| `A2A_BEARER_TOKEN` | _(unset)_ | Shared token; identity falls back to caller IP. |
| `A2A_HOST` | `127.0.0.1` | Bind host. Only widens with a token set. |
| `A2A_PORT` | `9900` | Inbound port. |
| `A2A_AGENT_NAME` | hostname-derived | Name on the Agent Card. |
| `A2A_PUBLIC_URL` | _(unset)_ | Routable URL advertised on the card (reverse proxies). |
| `A2A_TRUSTED_PEERS` | _(unset)_ | Allow-list of authenticated identities. Required for a non-loopback bind (with a token set) unless `A2A_ALLOW_ALL_USERS=true`; otherwise dispatch is refused. |
| `A2A_ALLOW_ALL_USERS` | `false` | Allow any authed peer (dev only). |
| `A2A_RATE_LIMIT` | `60` | Requests/minute per identity. |
| `A2A_MAX_PINGPONG_TURNS` | `5` | Anti-loop turn cap per context (max 20). |
| `A2A_REPLY_TIMEOUT` | `300` | Grace period before orphaned work is marked failed (seconds, clamped to 300–86400); also the default forwarded-profile reply timeout. Does not limit healthy local execution or HTTP waits. |
| `A2A_PUSH_SECRET` | bearer token | HMAC secret for push signing. |
| `A2A_ADVERTISED_TOOLSETS` | all registered | Restrict skills on the Agent Card. |

See [DESIGN.md](DESIGN.md) for architecture, lifecycle internals, and protocol compatibility notes.
