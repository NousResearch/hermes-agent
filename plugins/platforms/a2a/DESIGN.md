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
  which fulfils the pending per-**task** `Future` the HTTP request is blocked
  on (per-context FIFO, so concurrent same-context requests can't cross-talk);
  `on_processing_complete` resolves failures/cancellations promptly.
- **Task store:** every task (including terminal ones, bounded to the last
  500) stays queryable via `tasks/get` / `tasks/list`, and `tasks/subscribe`
  reattaches to a running task's stream via store watchers. A watchdog fails
  orphaned tasks after 5 minutes (idempotent transitions — no double
  counting in metrics).
- **input-required:** the platform hint tells the agent to start a reply with
  `[INPUT_REQUIRED]` when it needs clarification; the adapter maps that to
  `TASK_STATE_INPUT_REQUIRED` with the question in `status.message`.
- **Push notifications:** config accepted inline in `message/send`
  (`configuration.taskPushNotificationConfig`) or via the create method
  (returns `configId` + `createdAt`). On terminal transition the callback
  receives a v1.0 `StreamResponse` (`statusUpdate`) payload, HMAC-SHA256
  signed (`X-A2A-Signature`, secret `A2A_PUSH_SECRET` falling back to the
  bearer token), with SSRF-guarded callback URLs. Every intended
  newly-published terminal state — `COMPLETED` (normal, forwarded,
  late, loopback), `FAILED` (watchdog orphan, disconnect/shutdown,
  processing failure), and `CANCELED` (`tasks/cancel` and deferred
  cancellation) — makes exactly one best-effort `statusUpdate` callback
  attempt when `newly_published=true` and a valid persisted callback URL
  exists. Missing/invalid URL remains callback-free, duplicate terminal
  replay makes no second callback, callback failure is best-effort and
  never rolls back durable state, and no retry/replay/ACK is performed.
  `tasks/cancel` does not abort the in-flight agent turn.

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
Task store, turn tracker, rate limiter, and **windowed duplicate suppression**
(`_inbound_seen` map, 60 s window, 1,024 entries, process-local) are
**adapter-instance** objects (classes / maps in `protocol.py` / `adapter.py`).
The metrics counter bag stays a module singleton because it is intentionally
shared between the inbound adapter and the outbound client tools
(`/metrics` and `a2a_list` report both directions).

**Windowed duplicate suppression** is bounded admission control, not durable
idempotency or replay protection: the key is process-local
`(contextId, messageId)`, entries expire after 60 s, the map is capped at
1,024, it is not persisted, restart forgets it, and a duplicate gets a new
`REJECTED` Task rather than the first request's Task/result. It does not
cause request replay, result replay, or exactly-once execution.

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

## Edison re-baseline (2026-09-03) — supersession notes

This re-baseline supersedes the following prior assumptions; the
terminology below is canonical:

1. `protocol.is_valid_a2a_result` no longer defines validity by
   meaningful-key presence. The strict parser/schema in this doc's
   §4 (Task/Message/Part/Artifact rules, exact-one wrapper, explicit
   `V1_WRAPPED` vs `LEGACY_BARE`) is authoritative.

2. `unwrap_send_message_response` may not select `task` from a
   both-member wrapper. The oneof contract requires `v1_payload_count`
   failure; production callers use `parse_send_message_result`.

3. The `TaskStore.complete() then persist()` pattern is replaced by the
   disk-first durable publication primitive
   `TaskStore.publish_durable(ledger_path, task_id, candidate_record)`:
   stage clone → atomically replace ledger → update memory → wake
   observers → post-commit audit/metrics/callback/push → return success.
   No `memory terminal → persist → return` path is permitted.

4. `adapter.send()` per-context FIFO does not prevent cross-talk.
   Exact `task_id` (via `HERMES_SESSION_THREAD_ID` or `reply_to`) plus
   `contextId`/`peer`/`agent_slug`/`tenant` verification prevents it;
   context-only selection is valid only when exactly one active task
   exists in the context. Two concurrent same-context tasks via FIFO is
   replaced by exact-ID and ambiguity tests.

5. `DESIGN.md` statements that task transitions are merely “idempotent”
   mean **terminal-state immutability inside one TaskStore**. They do not
   mean durable request idempotency, exactly-once, or at-least-once
   delivery.

6. The `DESIGN.md` Part compatibility statement remains valid for
   tolerant inbound `extract_text` only. It does not loosen successful
   `v1` result validation.

7. The windowed inbound dedupe map is renamed to **windowed duplicate
   suppression** and retained under §8 of the decision: 60 s, 1,024
   entries, process-local `(contextId, messageId)`, bounded admission
   control, not durable idempotency or replay protection.

8. Prior successful transport probes remain authoritative preservation
   evidence, but aggregate green counts do not override the hostile
   predicates in the durability matrix.

## Amendment ac32ee — durability correction (2026-09-03)

This amendment locks the five residual boundaries that the prior
Edison artifact left ambiguous. It does not reopen parser, peer
identity, shutdown, transport trust, or dedupe decisions.

### A. Push result, conversation, and audit ownership

A conversation `persist_message(context_id, "agent", ...)` entry is
evidence of a validated successful push — not transport bookkeeping.

| Outcome | `PushOutcome` | Conversation `persist_message(..., "agent", ...)` | Audit | Success log/metric |
|---|---|---|---|---|
| Valid v1 result | `success=True` | Exactly once | Exactly one success `push` audit | Permitted once |
| JSON-RPC top-level `error` | `success=False, category="jsonrpc"` | Prohibited | Exactly one `push_failed` with redacted peer code/message | Prohibited |
| Malformed/foreign result | `success=False, category="invalid_response"` | Prohibited | Exactly one `push_failed` | Prohibited |
| Transport/timeout/no response | `success=False, category="transport"` | Prohibited | Exactly one `push_failed`; detail says indeterminate | Prohibited |
| Routing failure | `success=False, category="routing"` | Prohibited | Exactly one `push_dropped` or `push_failed` | Prohibited |
| Local durable failure | `success=False, category="durability"` | Prohibited | Exactly one `push_failed` durability audit | Prohibited |

Transport uncertainty permits one failure audit only; it does not
permit a conversation entry, success `push` audit, or success log.
JSON-RPC error is a stronger operation failure and also forbids a
conversation entry.

### B. Typed loopback propagation

Production return contracts are exact:

- `_push_loopback_in_process(...) -> PushOutcome`
- `_push_out_of_band(...) -> PushOutcome`
- `_try_push_reply(...) -> PushOutcome`
- `_push_reply_after_client_gone(...) -> PushOutcome`
- `adapter.send(...) -> SendResult`

No production branch returns `True`/`False`/`None` in place of
`PushOutcome`; no bool-compatibility branch hides failure.

For fire-and-forget loopback: durable WORKING creation precedes
local dispatch; durable COMPLETED publication precedes terminal
conversation/audit/log/watcher/success.  Failed WORKING leaves
ABSENT; failed COMPLETED leaves memory/disk WORKING, unresolved
Future/watcher, no terminal side effect, and `category="durability"`
through every caller. `adapter.send` maps it to
`SendResult(success=False, error=<category plus detail>)`.  A
durable terminal task is never rolled back because later network
delivery fails.

### C. fsync and atomic publication

Under the established lock order, the ledger is written via a
temporary file that is fully flushed and file-fsynced before
`os.replace`.  Serialization, flush, temp-file fsync, and replace
exceptions are publication failures: the temp file is cleaned where
possible, memory/observers are not updated, and the store returns
`DurablePublishOutcome(published=False, newly_published=False, ...)`.

Directory fsync is attempted after replace.  An unsupported
capability (`AttributeError`, `NotImplementedError`, `EINVAL`,
`ENOTSUP`, `EOPNOTSUPP`) falls back once per process to the weaker
guarantee (file-fsync + atomic replace) with a single warning;
it does not claim full directory-entry persistence.  Unexpected I/O
(`EIO`, `ENOSPC`, permission loss, unclassified `OSError`) fails
closed: after a post-replace unexpected error the store returns a
structured durability failure with `safeToRetry=false`, resolves no
watcher/Future, emits no success side effect, and marks the ledger
unavailable until a fresh locked reload or restart re-establishes
authority.  No A2A TaskState `INDETERMINATE` is invented.

### D. Missing authoritative Task record

A pending map/Future is not Task authority.  When
`_durable_complete_pending(task_id, ...)` cannot read an
authoritative `TaskStore` record for `task_id` it returns failure,
`adapter.send` returns `SendResult(success=False)` with
task-authority/durability detail, the Future remains unresolved, the
pending and pending-order entries are retained for reconciliation or
shutdown, no Task is created, no replacement ID is selected, no
context/FIFO fallback follows an explicit task ID, and no terminal
conversation/audit/metric/callback/success log is emitted.  Tests
must create a durable WORKING record before using a pending Future;
production contains no memory-only success fallback.

### E. Same-task terminal authority across TaskStores

The per-ledger interprocess file lock owns same-task serialization.
For every `publish_durable`:

1. Acquire the in-process lock, then the per-ledger file lock in the
   established order, and load the authoritative ledger under that
   lock; an unreadable/unparseable ledger fails closed and is never
   replaced with an empty snapshot.
2. Compare ownership and terminal state against the disk record, not
   only `self._tasks`.
3. Existing terminal + identical state/reply: return the disk record
   with `published=True, newly_published=False`; do not rewrite or
   repeat side effects.
4. Existing terminal + conflicting state/reply: return
   `published=False, newly_published=False` with terminal-conflict
   error; do not rewrite or resolve observers.
5. Reconcile the caller cache to the disk record before both
   terminal returns.
6. Only a nonterminal authoritative record may take a legal candidate
   transition; unrelated IDs may merge without stale same-task
   overwrite.

### F. Failure audit ownership and commit-latched side effects with safe semantic reply versus bounded audit copy (Wave 16 — 82c1eb05 successor, amended by Edison convergence)

The layer that first creates a failed `PushOutcome` owns exactly one failure-audit attempt via the central `_failure_outcome` creator and `_audit_safe` writer; propagators (`_try_push_reply`, rescue, `adapter.send`, OOB wrappers including `_drop_unresolvable_reply`) return the existing outcome unchanged and never re-audit. `routing` maps to `push_dropped`, `transport`/`jsonrpc`/`invalid_response`/`durability` map to `push_failed`. The owner invokes `_audit_safe` once with bounded/redacted observability copies (peer/task/context capped at 128 code points, detail at 300, truncation marker inside cap, fallback `[redacted]`); a writer failure is an observability-only failure, emits at most one bounded diagnostic warning, performs no retry or audit-of-audit, and never changes the returned `category`/`error`/`payload`. Every failed push has `agent` conversation appends `==0`, success `push` audits `==0`, success logs/metrics `==0`, exactly one failure audit attempt (`persisted ==0` when writer fails), and leaves durable state at `ABSENT` (WORKING publish failed) or last durable `WORKING` (terminal publish failed) with no terminal Future/watcher resolution.

Safe semantic reply versus bounded audit copy: `adapter.py` exposes `def _redacted_reply_text(value: object) -> str` — string as-is through `security.redact_outbound`, non-string, redaction failure, or non-string redaction result becomes `[redacted]`, no arbitrary conversion. The full safe redacted semantic reply is derived before every success surface; persistence, durable/display state, and loopback use the full safe reply and are not truncated to the 300-code-point observability limit. A separate `audit_reply = _bounded_redacted_detail(safe_reply, 300)` is passed through `adapter._audit_safe` at the actual audit writer; audit is `<=300` code points, best effort, no retry/reclassification, and cannot downgrade committed success. Successful `PushOutcome` is `success=True, category="transport", error="", payload=None`; raw response, parsed payload, and raw reply never reach persistence, audit, loopback, logs, or returned results.

- Remote `_push_out_of_band` commits only after a valid strict `V1_WRAPPED` parse. Before commit, `None` is transport/no-response, non-mapping is invalid_response, top-level error is jsonrpc via `_redacted_jsonrpc_detail`, and strict parser rejection is invalid_response — each returns a typed failure from `_failure_outcome` with one owner audit and no `agent`/`push`. After commit the implementation extracts `raw_reply = _parsed.text`, derives `safe_reply = _redacted_reply_text(raw_reply)` and `audit_reply = _bounded_redacted_detail(safe_reply, 300)`, constructs `PushOutcome(payload=None)`, and for a nonempty `safe_reply` persists `safe_reply` via `protocol.persist_message`, audits only `audit_reply` via `_audit_safe`, and loops back `safe_reply` via `_push_loopback_in_process`. No raw reply, response, or parsed payload reaches those surfaces; logs use no reply text. A valid result with no textual reply remains a committed transport success with no reply persistence or loopback.

- Loopback entry boundary: `_push_loopback_in_process` derives `safe_text = _redacted_reply_text(text)` and `audit_text = _bounded_redacted_detail(safe_text, 300)` before constructing params or invoking `_prepare_task`; every branch uses `safe_text`, never the caller object. `TaskRPCHandler._finalize_task` exposes both `_redacted_reply_text` and `_audit_safe` to the mixin; `display_reply = _redacted_reply_text(reply or "")` is the authoritative persisted/display value (full, not 300-truncated) with input-required marker detection/removal, durable publication happens before any post-commit effect, and `audit_reply` (bounded copy) is sent once via `_audit_safe` only after `newly_published`. Persistence and notification receive full `display_reply`; audit receives only `audit_reply`.

- `_drop_unresolvable_reply(context_id, peer) -> PushOutcome` calls `_failure_outcome("routing", "peer identity not resolvable", ...)` and returns that outcome. Both loopback-reply and own-endpoint-reply refusals return it directly without constructing a second `PushOutcome` or calling a throwing `security.audit`.

- `_try_push_reply` invalid/non-pushable or empty reply is owned routing with one `push_dropped`; `pending.pushed` dedupe stays successful with no effects; a delegated `_push_out_of_band` outcome is returned unchanged with no re-audit; only an exception before a delegated outcome exists becomes a locally owned transport failure with `_bounded_redacted_detail` for log/audit/error and one `push_failed`.

- `_push_reply_after_client_gone` strict parse rejection is locally owned invalid_response with one `push_failed`; Message/not-pushable/empty is owned routing with one `push_dropped`; a delegated `_push_out_of_band` failure propagates unchanged; unexpected pre-outcome exception is locally owned transport with one `push_failed`. All logs use bounded values; no raw `ve.detail` or `str(exc)` escapes.

- Local loopback `want_reply=True` commits after durable `WORKING` publication and accepted local scheduling. Before commit, WORKING failure returns durability (`ABSENT`) and scheduling/runtime failure returns transport (unless `DurablePublishError`), each with one `push_failed`. Immediate terminal rejection is routing. After commit the task remains `WORKING` for `adapter.send` to complete; the `agent` append and success `push` audit are best-effort via `safe_text`/`audit_text` and cannot downgrade.

- Local loopback fire-and-forget commits after durable `COMPLETED` publication. Before commit, WORKING failure leaves `ABSENT` and COMPLETED failure leaves `WORKING`, each with one `push_failed` and no terminal effects. After `COMPLETED` commits, the terminal `agent` append, success `push` audit, metrics, and push notification are best-effort. `TaskRPCHandler._finalize_task` converts publication exceptions/failures to `DurablePublishError` before commit and encloses the entire post-commit tail (persist/audit/metric/push) so no conversation/audit/metric/callback/notification or logging exception escapes a published `True`; post-commit exceptions are logged via `self._bounded_redacted_detail`.

Side effects are commit-latched. Before the authoritative commit point, failures return typed outcomes, make no `agent` append or success `push` audit, emit one owner audit, and leave the required durable state. After commit, `agent` append, success `push` audit, metrics, logs, and callbacks are best-effort and cannot downgrade a committed success; an audit-writer fault after commit is not a durability failure. No post-commit diagnostic may use raw `repr`, `str(exc)`, task/peer objects, or exception text; the loopback and `task_routing` post-commit paths use the adapter-provided helpers and are enclosed so no exception escapes a published success.

Direct `_push_out_of_band` routing exits (`no peer`, `registered peer not resolvable`, loopback `want_reply` refusal, own-endpoint reply refusal) each emit one `push_dropped` via `_failure_outcome`. `_try_push_reply` invalid/empty, rescue parse/message/state/empty, and `adapter.send` unmarked loopback/missing-peer are owned `push_dropped` or `push_failed` exactly once. Delegated failures propagate unchanged without outer re-audit, and `adapter.send` maps every push-derived failure through `_send_result_from_outcome` to `SendResult(success=False, error="<category>: <sanitized bounded detail>")` with the category prefix exactly once and final error `<=300` code points. Pre-outcome thread-pool exceptions are locally owned transport failures with one `push_failed` then mapped once. No `SendResult` leaks an unbounded detail or raw sentinel.

### G. Recursive hard-final JSON-RPC sanitizer and globally bounded detail with hard bounded dict traversal (Wave 16 — 82c1eb05 successor, amended by Edison convergence)

Every dynamic OOB failure detail — invalid-response `resp`, JSON-RPC `raw_error`, parser `ve.detail`, loopback `str(exc)`, user reply extraction, and mapped `SendResult` error — passes through the single guarded `_bounded_redacted_detail(value, cap=300)` boundary before reaching `PushOutcome.error`, `payload`, audit, logs, or `SendResult`. No raw `repr(resp)`, `resp!r`, `ve.detail`, `str(exc)`, task/peer objects, or exception text reaches those surfaces; credential-shaped sentinels are absent. `PushOutcome.category` is preserved as `routing`/`transport`/`jsonrpc`/`invalid_response`/`durability` and never downgraded.

`_bounded_redacted_detail` order is guarded `str(value)` conversion, guarded `security.redact_outbound`, final strict `_truncate_codepoints` with fallback `[redacted]`; the adapter exposes it as `self._bounded_redacted_detail` for `TaskRPCHandler` to avoid a duplicate sanitizer and circular import. Dynamic log/audit copies use caps 128 for context/task/peer fields and 300 for error/detail; routing/TaskStore keys keep original values and only their rendered copies are bounded.

`PushOutcome.payload` for a JSON-RPC `error` exposes only a new recursively sanitized bounded object with allowlisted top-level `code`, `message`, and `data`; all other peer fields are dropped. A non-object error becomes `{"message": <bounded redacted string>}` via `_bounded_redacted_detail`. The sanitizer never mutates or returns the peer object.

Hard-final JSON-RPC bounds (all caps are final, markers inside caps, no transformation after cap):
- `_DETAIL_MAX_CODEPOINTS = 300`; `_JSONRPC_KEY_MAX_CODEPOINTS = 64`; `_JSONRPC_STRING_MAX_CODEPOINTS = 300`; `_JSONRPC_MAX_DEPTH = 4`; `_JSONRPC_MAX_WIDTH = 16`; `_JSONRPC_MAX_BYTES = 2048`; signed 32-bit `code` range `[-2147483648, 2147483647]`; markers `"...[truncated]"`, `"[redacted]"`, `"[truncated]"`.
- `_truncate_codepoints` reserves marker space before slicing; output length `<=cap`; marker is part of cap.
- `_sanitize_string_for_jsonrpc` is a thin wrapper: `redact_outbound` then `_truncate_codepoints`.
- `_sanitize_jsonrpc_value(value, depth)` with `data` at depth `0`; depths `0..4` representable, depth `5` becomes `"[redacted]"`; `None`/bool/signed-32-bit-int/finite-float retained, out-of-range/non-finite/non-JSON become `"[redacted]"`; for dict/dict subclass the sanitizer obtains `iterator = iter(dict.items(value))` (built-in descriptor bypassing overridden `value.items()` without copying the source), runs `for _ in range(_JSONRPC_MAX_WIDTH)` with at most one `next(iterator)` per iteration, stops on `StopIteration`, any other iterator error replaces the whole mapping with `[redacted]`, counts source visits before key acceptance (width budget is source entries visited, not output keys retained), sanitized-key collisions consume a visit and first key wins without touching the later duplicate value (do not sanitize, recurse into, or otherwise touch the duplicate value), keys string-only (non-string maps to `[redacted]` without conversion) redacted and capped at `64`, first collision wins; list retains at most first `16` items via slicing; tuples/arbitrary iterables and non-dict `collections.abc.Mapping` become `[redacted]` without traversal/conversion/indexing/rendering — a generic mapping can perform unbounded work inside one user-defined `items`, `__iter__`, `__len__`, or `__getitem__` call, so rejecting it is the only in-process hard bound; the helper never calls `list`, `tuple`, slicing, `len`, `repr`, `str`, `keys`, `values`, or the instance `items` method on the mapping source, and never calls overridden `__iter__`/`__len__`/`__getitem__` of a dict subclass; nested work is bounded independently by depth 4 and width 16, compact final serialization adds no marker; the helper is total and never exposes the raw object.
- `_redacted_jsonrpc_detail(raw_error) -> (error, payload)` accepts only built-in `dict`/dict subclass for JSON-object sanitization; a non-dict `collections.abc.Mapping` becomes constant `[redacted]` without traversal. For dict/dict subclass it allocates a new payload and probes only `code`/`message`/`data` via built-in dict operations (`dict.__contains__`, `dict.get`) — no top-level enumeration — so a dict subclass whose overridden `__contains__`/`get`/`__getitem__` trap is not invoked. `code` integer excluding bool within signed 32-bit is retained, `message` is sanitized string via `_sanitize_string_for_jsonrpc` or `_bounded_redacted_detail` for non-string, `data` via `_sanitize_jsonrpc_value` with the exact bounded dict iteration described above; otherwise `{"message": _bounded_redacted_detail(raw_error)}`; ensures `message` fallback to `"[redacted]"` when nothing survives; compact-serializes with `separators (",", ":")`, `ensure_ascii=False`, `allow_nan=False` and enforces UTF-8 `<=2048`; on oversize retains validated code/message and sets `data="[truncated]"` for a re-check; on remaining failure returns constant `{"message":"[redacted]","data":"[truncated]"}`; builds `error` only from retained sanitized code/message with guarded `redact_outbound` and final `_truncate_codepoints(...,300)`; never reuses an unbounded hostile integer; `category` remains `jsonrpc` with `PushOutcome.category="jsonrpc"`.

`PushOutcome.error` is built from sanitized `code` and `message` only, capped at `300`, already redacted. Every warning log, failure audit, and `SendResult.error` for JSON-RPC uses only that sanitized error and bounded payload. The adapter exposes `_redacted_reply_text`, `_bounded_redacted_detail`, and `_audit_safe` to `task_routing` so no duplicate sanitizer lives in `task_routing.py`.

### H. All-exit managed-loop lifecycle (Wave 16 — Edison convergence)

The reusable helper in `tests/plugins/test_a2a_result_durability_contract.py` owns the loop, thread, scheduled coroutine objects, returned concurrent futures, residual asyncio tasks, and adapter registration from setup through teardown across ten states: `NEW` → `RUNNING` (loop created, thread started, readiness confirmed, adapter bound, real scheduler saved, capture wrapper installed) → `BODY_EXITED` (body returned, raised assertion/error, or cancellation — preserve original `BaseException` and traceback) → `SETTLING` (inspect every captured future; retrieve completed results; cancel unfinished futures and reconcile in-loop; unexpected completed exceptions are cleanup failures) → `DRAINING` (submit one cleanup coroutine with the saved real scheduler, never the monkeypatched wrapper; exclude itself, cancel every remaining task, await `gather(..., return_exceptions=True)`, yield once, return survivors; any survivor/timeout is cleanup failure) → `STOPPING` (request `loop.stop` thread-safely) → `JOINING` (join with bounded timeout; live thread after timeout is cleanup failure) → `CLOSING` (close loop only after thread stopped; assert `loop.is_closed()`) → `UNREGISTERING` (unregister adapter in all cases) → `CLOSED` (all resources released, no owned pending work remains). Every teardown phase is attempted even if an earlier phase fails; cleanup errors are accumulated, not swallowed.

Scheduler ownership: the capture wrapper calls the saved real `asyncio.run_coroutine_threadsafe` exactly once. If that call returns, ownership transferred to the loop and the returned future is recorded. If it raises before transfer, the wrapper explicitly closes the received coroutine and re-raises. The same close-on-rejection rule applies when submission of the cleanup coroutine itself fails. A test that intentionally installs a rejecting scheduler must assert the coroutine reached `CORO_CLOSED`; warning suppression and garbage collection are not substitutes.

Exception precedence: no body exception and no cleanup failure returns normally; no body exception and cleanup failure raises `AssertionError` containing every failed phase; body exception and no cleanup failure re-raises the original exception with its traceback; body exception and cleanup failure raises `BaseExceptionGroup` containing the original body exception first and one cleanup `AssertionError` second. Neither failure may mask the other. This rule applies to normal exceptions, pytest assertions, and cancellation. `KeyboardInterrupt` and `SystemExit` are preserved as body failures while teardown still runs.

Acceptance: normal, assertion/error, and cancellation exits reach `CLOSED`; captured futures are retrieved or canceled and reconciled; residual tasks are canceled and gathered before stop; rejecting scheduler coroutines are explicitly closed; thread is not alive; loop is closed; adapter is unregistered; a survivor, drain timeout, thread timeout, future exception, or close failure is visible as test failure. Relevant nodes pass under `-W error::RuntimeWarning` with no `coroutine was never awaited`, `Task was destroyed but it is pending`, or unclosed-loop diagnostic.

## Files
```
plugins/platforms/a2a/
├── plugin.yaml      # manifest (kind: platform)
├── __init__.py      # register(): platform adapter + client tools
├── adapter.py       # inbound A2A lifecycle, dispatch, persistence, origin/send/OOB
├── http_transport.py # HTTP/wire boundary: JSON-RPC bounds, redaction, _A2AServer, A2ARequestHandler
├── task_routing.py  # TaskRPC mixin: message/send, stream, task, push-config RPC handlers
├── a2a_persistence.py # file-lock, context→peer/session, fanout, task-ledger paths
├── tools.py         # outbound client tools
├── protocol.py      # Agent Card, JSON-RPC framing, task store, persistence + strict parser + durable primitive
├── security.py      # auth/identity, injection filters, redaction, audit
├── DESIGN.md
└── README.md
tools/
├── send_message_tool.py        # schema, target parsing, cron dedup, live-adapter dispatch
└── send_message_transports.py  # standalone platform transports (telegram, signal, matrix, …)
```

Inline push: `TaskRPCHandler._inline_push_fields` is a pure validator (inspects only
`configuration.taskPushNotificationConfig`, accepts direct `.url` and nested
`.pushNotificationConfig.url`, requires dict containers, nonblank string, agreement,
and `is_safe_callback_url`); it generates one `cfg-` + 12 hex ID before the
accepted WORKING candidate, and `_prepare_task` publishes URL/ID atomically in
the first `publish_durable` call (no pre-task `set_push_config`). Missing or
malformed/unsafe config keeps empty fields with at most one bounded warning.
