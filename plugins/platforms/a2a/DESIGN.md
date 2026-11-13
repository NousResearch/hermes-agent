# A2A Platform Plugin — Design

Consolidates the entire A2A (Agent-to-Agent) feature cluster (#514 and friends)
into one **plugin** with **zero core edits**, built on capabilities the current
codebase already exposes. Implements **A2A Protocol v1.0** (JSON-RPC binding).

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
- `a2a_discover(url)` — fetch + summarize a peer's Agent Card (v1.0
  `supportedInterfaces` aware, tolerates 0.3 cards).
- `a2a_call(agent, message, context_id?)` — send a JSON-RPC `message/send`
  task to a peer, return the reply. Multi-turn via `context_id` (carried
  inside the Message per v1.0). Surfaces `TASK_STATE_INPUT_REQUIRED` so the
  model knows to answer and continue the context.
- `a2a_list()` — configured peers + persisted conversations + metrics.
- `a2a_history(context_id, limit?)` — recall a persisted conversation
  (this is the production consumer of the persistence layer).
- `a2a_orchestrate(capability, message, mode?)` — fan-out one task to every
  configured peer advertising a capability. Modes: `all` (every reply),
  `first` (first success), `best` (longest successful reply — a deliberately
  coarse heuristic; errors never win, and an all-error fan-out reports the
  failures instead of picking one).

Peers resolved from `config.yaml` → `a2a_agents`, or a direct URL.

### Inbound — platform adapter
- Stdlib `http.server` on a daemon thread (no asyncio loop needed at
  `register()` time — sidesteps the a2a_fleet "register outside a loop" bug
  class that killed inbound serving in forks). The request handler is a
  module-level class (`A2ARequestHandler`) reached through
  `server.adapter`, so RPC handlers are unit-testable without HTTP.
- Agent Card at `GET /.well-known/agent-card.json` (canonical v1.0 path; legacy `agent.json` also answers) (v1.0: `supportedInterfaces[]`,
  `provider`, `capabilities.extendedAgentCard`). **Dynamic**: skills are
  built from the live tool registry at serve time
  (`A2A_ADVERTISED_TOOLSETS` / `extra.advertised_toolsets` restricts them).
- JSON-RPC methods: `message/send`, `message/stream` (SSE), `tasks/get`,
  `tasks/list`, `tasks/cancel`, `tasks/subscribe`,
  `tasks/pushNotificationConfig/create` (legacy `set` names accepted).
- **Live-session injection (the #11025 insight):** inbound tasks route through
  the normal `MessageEvent` → `handle_message` path keyed by the A2A
  `contextId`, so the agent that answers is the same one serving the user —
  full memory/context, not a clone. The reply returns through `adapter.send()`,
  which completes the stored task before fulfilling the per-**task** `Future`.
  The HTTP request is only an observer. Originating message IDs and the dispatch
  context bind final sends to their task; the processing-start hook rebinds that
  identity for queued turns rather than inheriting the previous turn's ID.
   Local per-context asyncio lanes wait for the real base processing task before
   dispatching the next input, preventing debounce/merge from dropping task IDs.
   Context FIFO remains an ambiguous compatibility fallback for unanchored sends;
   it does not skip canceled entries and misroute their late replies to a successor.
   `on_processing_complete` also handles failures,
  cancellations, empty replies, and the gateway's streamed-final-response stash.
- **Task store:** terminal tasks are retained for `extra.task_retention_seconds`
   (default 86400 seconds) after their terminal transition, bounded additionally
   to the last 500 by completion recency. Count eviction may shorten age retention.
   Tasks stay queryable via `tasks/get` / `tasks/list`, and `tasks/subscribe`
  reattaches to a running task's stream via store watchers. A watchdog fails
  stale records without an execution owner after the `A2A_REPLY_TIMEOUT` grace,
  clamped to 300–86400 seconds and checked every 60 seconds. This marks orphans
  failed; terminal-result retention removes their records later. The same setting
  supplies the default forwarded-profile transport timeout. Accepted local work,
  including queued turns, stays owned until completion
  callbacks or shutdown; HTTP disconnects and the return of the dispatch coroutine
  do not release ownership. There is no local execution deadline.
  Winning transitions alone count metrics and
  schedule completion I/O; no detached-task waiter threads are created.
- **input-required:** the platform hint tells the agent to start a reply with
  `[INPUT_REQUIRED]` when it needs clarification; the adapter maps that to
   `TASK_STATE_INPUT_REQUIRED` with the question in `status.message`.
- **Continuation:** only an explicit `message.taskId` resumes an input/auth wait;
   `contextId` alone still creates a new task. A task-only message inherits the
   exact context. Unknown/purged and out-of-scope references return TaskNotFound,
   mismatched contexts return invalid params, and terminal or active tasks reject
   new messages with UnsupportedOperation. Scoped resource APIs compare the selected
   slug and tenant exactly, including the empty root scope. Each resumed execution
   gets a fresh opaque attempt ID carried through event delivery, forwarding and
   cleanup. The original creation identity/time remains fixed. Old observer futures
   retain their own outcome. Unanchored legacy context-FIFO delivery is restricted
   to initial attempts because it cannot prove continuation ownership.
   Continuations count toward the existing context turn cap; retries do not.
   Repeated continuation message IDs observe the accepted attempt while the task
   remains nonterminal. Terminal tasks reject all sends, including repeats;
   clients retrieve those results with GetTask. Forwarded continuations retain the
   context but use independent owner input admissions per message.
- **Wait expiration:** `extra.input_wait_timeout_seconds` (default 86400 seconds)
   starts at the input/auth interruption, never creation or polling. Expiry cancels
   with an explicit input/auth wait-expired reason. Resume, cancellation and expiry
   arbitrate under the store lock, checking attempt identity and the deadline.
   Both lifecycle durations must be positive finite values. These 24-hour defaults
   are Hermes policy, not protocol mandates. Purged tasks return TaskNotFound.
   Task state is not durable across process restart. Full context conversation
   logs remain intact; partial internal clarification history is not serialized
   as a purported complete Task history.
- **Push notifications:** config accepted inline in `message/send`
  (`configuration.taskPushNotificationConfig`) or via the create method
  (returns `configId` + `createdAt`). On terminal transition the callback
  receives a v1.0 `StreamResponse` (`statusUpdate`) payload, HMAC-SHA256
  signed (`X-A2A-Signature`, secret `A2A_PUSH_SECRET` falling back to the
  bearer token), with SSRF-guarded callback URLs.
- **Completion I/O:** admission reserves one of 512 recording slots before execution;
   a separate single worker writes audit/conversation records and drains on shutdown.
    A reservation belongs to an execution attempt, not its public task ID. Queueing
    transfers the slot to that record; finishing any outcome, including clarification,
    releases it without touching a successor's reservation. Cancellation/expiry of
    an interrupted task reserves a separate outcome slot. Saturation leaves the
    state unchanged: cancellation returns a retryable capacity error and expiry
    retries on subsequent watchdog sweeps or GetTask/ListTasks/continuation reads.
    Capacity exhaustion refuses new execution. Recording is queued atomically with
    the winning transition. Its worker waits on a commit gate released outside the
    task lock, before observers wake; no disk I/O or Future callbacks run under that
    lock. This uses one recording worker, not a waiting thread per task.
    Two independent push workers allow at most
   128 jobs; overload or shutdown may drop only optional pushes. The owning profile
   context is copied per job. Shutdown joins running workers. Push URLs are
   captured at transition time so delayed workers cannot consume later configs;
   interruptions preserve the config for subsequent outcomes.
- **Observation:** blocking sends and healthy SSE streams wait for terminal or
  interrupted outcomes without the old HTTP reply timer. Explicit
  `returnImmediately` returns a live task. Subscribers receive a current Task
  snapshot and unregister on disconnect. Terminal subscription replay remains a
  legacy compatibility behavior, rather than a claim of full v1.0 conformance.

## Forwarded-owner link failure limitation

Incoming HTTP observation and the adapter-to-profile-owner transport are separate
links. Disconnect-safe inbound execution does not promise recovery if the latter
times out or fails. The existing forwarding boundary reports an unverified reply
as a terminal `FAILED` public task, even though accepted work may continue at the
owner. That facade cannot accept explicit same-task continuation. A context-only
resend with the same stable message ID uses the existing owner-admission retry
mechanism via a new public task; it is not same-task continuation or complete
end-to-end recovery. Recovery of this terminal facade remains a separate known
issue. This implementation does not claim full A2A conformance.

## v1.0 wire format notes

- Task states / roles are SCREAMING_SNAKE_CASE (TASK_STATE_*, ROLE_*).
- Parts are member-presence discriminated — no kind field. All three
  Part types are supported: text (text + mediaType), file
  (url|raw + filename + mediaType), and data (data + mediaType).
  extract_text renders file/data Parts into the text stream (URL +
  filename for files, JSON for data) so the agent sees them; it also
  accepts v0.3 (kind) and pre-0.3 (type) shapes from older peers.
  Outbound replies are still text-only — the agent produces text, and
  file/data Parts are for inbound richness.
- Push notification config: full CRUD — create (inline in message/send
  via configuration.taskPushNotificationConfig, or via the create
  method), get, list, delete. Each config has a configId and createdAt.
  One config per task (v1.0 allows multiple; we keep one).
- SSE events are StreamResponse objects (statusUpdate / artifactUpdate
  members); stream closure signals the terminal state — no final field.
- contextId lives inside the Message (legacy top-level accepted inbound).
- Timestamps are ISO 8601 with millisecond precision; Tasks carry
  createdAt / lastModified.
- Error codes: A2A-reserved codes are used only with their spec semantics
  (`-32001` TaskNotFound, `-32002` TaskNotCancelable); custom errors sit at
  `-32050..-32052` (unauthorized / rate-limited / untrusted).

## Security (on by default)
- **Bind safety:** no token configured (`A2A_BEARER_TOKEN` or
  `A2A_PEER_TOKENS`) ⇒ bind `127.0.0.1` only. A token alone does not widen
  the bind; remote exposure requires token **and** explicit `A2A_HOST`.
- **Peer identity:** `A2A_PEER_TOKENS="alice:tok1,bob:tok2"` gives each peer
  its own credential; the matched name is the authenticated identity used
  for rate limiting, the trust gate, message framing, and audit. A shared
  `A2A_BEARER_TOKEN` authenticates as `ip:<addr>`. Nothing in the request
  body can assert identity. Comparisons are constant-time.
- **Trust gate:** `A2A_TRUSTED_PEERS` (or config `a2a.trusted_peers`)
  optionally restricts which authenticated identities may run tasks.
- **Injection filters:** ALL inbound text (including `/`-prefixed — remote
  peers can never reach operator slash commands) is defanged (ChatML /
  role-prefix / override patterns → `[filtered]`) and framed with a privacy
  prefix marking it untrusted peer input.
- **Outbound redaction:** credential-shaped strings (`sk-…`, `ghp_…`, JWTs,
  bearer tokens, emails) scrubbed before anything leaves.
- **Rate limiting:** sliding window per authenticated identity
  (`A2A_RATE_LIMIT`/min).
- **Anti-loop:** per-context turn cap (`A2A_MAX_PINGPONG_TURNS`, default 5,
  hard max 20) rejects (v1.0 `TASK_STATE_REJECTED`) runaway agent↔agent
  ping-pong; `tasks/cancel` resets the counter for the task's context.
- **Audit log:** append-only `~/.hermes/a2a_audit.jsonl` for every exchange.

## State placement
Task store, turn tracker, and rate limiter are **adapter-instance** objects
(classes in `protocol.py`). The metrics counter bag stays a module singleton
because it is intentionally shared between the inbound adapter and the
outbound client tools (`/metrics` and `a2a_list` report both directions).

## Persistence (survives compaction)
A2A conversations are written to `~/.hermes/a2a_conversations/<context>.jsonl`,
outside the context-compaction pipeline — compaction and restarts can't lose
them (#11025 requirement). The `a2a_history` tool recalls them by context id.

## Requirements traced to the cluster

| Source | Requirement | Where |
|---|---|---|
| #514, #23871, #4135 | Agent Card discovery | `protocol.build_agent_card`, adapter GET |
| #4135, #14559, #8948 | Client: discover / call / list | `tools.py` |
| #11025 | Live-session injection (not a clone) | `adapter._prepare_task` |
| #11025 | Privacy filters + outbound redaction + audit | `security.py` |
| #11025 | Conversation persistence outside compaction | `protocol.persist_message`, `a2a_history` |
| #514, #11025 | Auth, localhost-default | `security.authenticate`, `resolve_bind_host` |
| #56434 | Trusted peer approval | `security.is_trusted_peer` |
| #56435 | Task completion notifications | push notifications (`_send_push_notification`) |
| #25176, #689 | Agent↔agent messaging across machines | client tools + inbound adapter |
| #7517 et al. | Multi-peer orchestration | `a2a_orchestrate` |

## Deliberately out of scope (future, not this pass)
- **a2a-sdk / gRPC + HTTP+JSON bindings.** Only the JSONRPC binding is
  served; the card advertises exactly that.
- **`tenant` field, extended Agent Card, `stateTransitionHistory`.**
- **True task abort:** `tasks/cancel` marks the task canceled and drops the
  reply, but cannot abort the live session's in-flight turn.
- **DID / Ed25519 identity, OAuth2 scopes, x402 micropayments** (#14559
  bindu) — heavy, niche; revisit if there's real demand.

## Files
```
plugins/platforms/a2a/
├── plugin.yaml      # manifest (kind: platform)
├── __init__.py      # register(): platform adapter + client tools
├── adapter.py       # inbound A2A v1.0 server (stdlib http.server)
├── completion.py    # bounded completion I/O offload
├── continuation.py  # explicit task-reference validation and attempt admission
├── tools.py         # outbound client tools
├── protocol.py      # Agent Card, JSON-RPC framing, task store, persistence
├── security.py      # auth/identity, injection filters, redaction, audit
├── DESIGN.md
└── README.md
```
