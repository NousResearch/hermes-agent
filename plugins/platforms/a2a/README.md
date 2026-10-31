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

### Connections and completion

`SendMessage` waits for a terminal or interrupted outcome by default. Set
`configuration.returnImmediately: true` to receive the live Task promptly and
retrieve the result later with `GetTask`. `SendStreamingMessage` starts with a
Task containing its `id` and `contextId`, then sends updates and keepalives until
completion or an interrupted outcome such as `TASK_STATE_INPUT_REQUIRED`.
Disconnecting an SSE client releases only that observer: accepted work continues.
`SubscribeToTask` starts with the current Task snapshot and observes the same result.
The existing terminal-task subscription replay is retained for compatibility;
this differs from the v1.0 rule rejecting terminal subscriptions.

Completion callbacks store the redacted result before waking request waiters.
Results remain queryable in memory while the process lives; conversation logs
do not provide task restart/resumption. Cancellation
wins at most once against completion and does not abort the underlying turn.
Clarification replies end the current observation. To continue that logical task,
send the answer with `message.taskId` set to its ID and a fresh `messageId`.
`contextId` may be omitted (inherited from the task); if supplied it must match
literally. This works for `INPUT_REQUIRED` and `AUTH_REQUIRED`. A context-only
message creates a new task; it never implicitly resumes an interrupted task.
Unknown or purged task IDs return TaskNotFound; terminal tasks reject messages
with UnsupportedOperation. Task IDs cannot be chosen by callers.

Each continuation gets an independent internal execution identity. Old sends,
cleanup callbacks and HTTP observers cannot complete or observe the next attempt.
Repeating a continuation's message ID while the task is nonterminal observes that
accepted attempt without executing again. After the task becomes terminal, even
that resend is rejected; retrieve its result with `GetTask`. Continuations count
toward the context's anti-loop cap; retry observations do not. Push configuration
remains attached across clarification and is captured separately for each outcome.

Local tasks in one context execute in sequence so gateway input coalescing cannot
merge accepted tasks. Cancellation of a queued task skips its execution.

Audit and conversation writes use a separate single worker, drained on ordinary
shutdown. Admission reserves recording capacity (512 executions or outstanding
recordings); a recorded clarification releases its slot. A resumed attempt
reserves a distinct slot even if its preceding write is still queued. Exhaustion
refuses new execution explicitly. Cancellation of an idle interrupted task needs
a recording slot: overload returns a retryable capacity error without canceling
or losing its record. Expiry retries on the next watchdog sweep or task read.
Webhooks
use two workers with at most 128 outstanding jobs. Saturation or shutdown may
drop optional pushes, never local records or stored results. Shutdown releases
request waiters and joins running I/O (webhook requests have a 10s timeout).
`extra.input_wait_timeout_seconds` expires input/auth waits into `CANCELED` with an
explicit reason, measured from entering the interrupted state. Polling does not
renew this deadline. `extra.task_retention_seconds` purges terminal results by
terminal-transition age. Both default to 86400 seconds (24 hours), require positive
finite values, and are Hermes policies rather than A2A mandates. The 500-terminal
count cap can evict results sooner. Owned execution has no age deadline; genuine
unowned work still uses the separate orphan grace. Tasks are in-memory and do not
resume after process restart. Full conversations remain in the persistence log;
the Task response does not advertise a complete wire `history`.

**Forwarded-owner link limitation:** inbound HTTP disconnect safety does not
guarantee recovery when the adapter's separate link to a forwarded profile owner
times out or fails. Accepted owner work may continue while the adapter's public
task becomes `FAILED`; same-task continuation cannot recover that terminal facade.
The existing context-only resend with the same stable `messageId` can reread the
owner admission through a new public task. This is distinct from same-task
continuation and is not complete end-to-end recovery. No full A2A conformance is
claimed.

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
| `A2A_REPLY_TIMEOUT` | `300` | Legacy orphan-record grace in seconds, clamped to 300–86400s and swept every 60s. Only records without an execution owner can expire; live accepted/queued local work remains protected until completion callbacks or shutdown, regardless of connected observers. No HTTP observation or local execution deadline. Also the default forwarded-profile transport timeout; an unverified owner reply does not cancel accepted owner work. |
| `A2A_PUSH_SECRET` | bearer token | HMAC secret for push signing. |
| `A2A_ADVERTISED_TOOLSETS` | all registered | Restrict skills on the Agent Card. |

See `DESIGN.md` for architecture and the requirement-tracing table.
