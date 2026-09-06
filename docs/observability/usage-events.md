# Local Codex usage events (PR1, persistence slice)

This is an exact aggregation of **recorded provider-reported usage**, not exact
provider consumption, subscription quota, billing, or an account-wide ledger.
The persistence slice adds no materialized bucket or backfill. The backend-RPC
slice below adds sanitized quota retrieval and Active Work; no UI is included.

## Storage and contract

`hermes_state_usage_events.py` adds a `SessionDB` mixin. The additive
`usage_events` table and two indexes live in `hermes_state_common.SCHEMA_SQL`.
`SessionSchemaMixin._init_schema` creates them for both existing and new stores.
There is no row migration or schema-version bump: that version gates DATA
migrations; this store uses idempotent declarative schema reconciliation.
Existing sessions, their counters, and their indexes are not rewritten.

The recorder uses the executing agent's existing `_session_db`, not a new default
home connection. This preserves non-launch profile routing and the dedicated
child handle returned by `tools.delegate_tool._open_child_session_db`. The event's
`profile` is an opaque SHA-256 fingerprint of `hermes_home_key(db.db_path.parent)`;
no absolute path is exposed in a result. It identifies a local profile directory,
not an account; copying/moving a database does not relabel its historical events.

An immutable observation snapshots provider, outgoing model, profile, DB handle,
and a random attempt UUID at execution. It stamps UTC Unix microseconds immediately
when the execution callback returns (before execution middleware returns, response
checks, plugin callbacks, or database writes). This is backend-observed completion,
not a provider-server clock or individual-token timing. A long request belongs wholly
to its completion bin. Wall-clock adjustments are not smoothed or backfilled.

The UUID is independent of `api_request_id`: the latter is a logical-loop ID and
can repeat across retries. Successful outer retry attempts and continuation calls
receive separate UUIDs. Re-observing the same attempt at the post seam uses the same
UUID; `ON CONFLICT(attempt_id) DO NOTHING` keeps the first observation. Concurrent
processes seeing the same event cannot insert it twice. Rows are not tied to a
session foreign key, so deleting a transcript does not silently change the timeline.

Only these response fields are retained:

- input/output token counts (input INCLUDES cached input),
- input details `cached_tokens`, `cache_write_tokens` (older alias
  `cache_creation_tokens` only when the primary field is absent),
- output details `reasoning_tokens`,
- an allowlisted response status and the outer loop's retry count.

No message, prompt, tool arguments, response object, reasoning text, response ID,
URL, header, credential, exception string, or parent summary is stored.
`CanonicalUsage.input_tokens` in pricing is **uncached** input, so it cannot be
copied directly into this event field. The event normalizer follows the Codex
Responses field contract but preserves absence rather than pricing's zero defaults.
Nonnegative integer fields are accepted; bools, negatives, non-integers, and values
outside SQLite's integer range become NULL and mark the usage `invalid`. All absent
counts mean `missing`; absent input or output means `partial`; measured zero remains
zero. Optional cache/reasoning absence does not invalidate known input/output.

## Proven call-path coverage and limits

The integration lives at `turn_api_call.perform_api_call`'s execution callback
(snapshot only) and `turn_response_intake._fire_post_api_request_hook` (persistence).
Recording does not depend on an installed plugin or `has_hook` returning true.

Source trace:

- `conversation_loop` / retry phases call `perform_api_call`, then
  `check_api_response`, then `normalize_model_response` / the post hook.
- Streaming Codex calls take `interruptible_streaming_api_call` →
  `_stream_codex_passthrough` → `_interruptible_api_call` → `run_codex_stream`.
  The non-display/non-stream-dispatch route also reaches `run_codex_stream`.
  `codex_runtime._CodexResponseAssembler` copies terminal usage; if no terminal
  usage arrives it remains absent. There is no delta-by-delta recording.
- `turn_response_intake` fires the post hook before Codex-incomplete continuation
  and scratchpad retry guards, so these reported responses are separate events.
- Provider fallback mutates the agent route and rebuilds requests before the next
  execution. Attribution is snapshotted from THAT execution, not initial settings
  or a later mutable agent route. Only `openai-codex` + `codex_responses` is recorded.
- Delegated children use their own AIAgent execution and parent's profile DB file.
  Parent-facing delegated totals are never an ingestion source. MoA/agent-as-provider
  parent projections are not counted as Codex requests.

Behavioral evidence: the new tests exercise streaming/non-streaming dispatch into
real SQLite, immutable attribution, duplicate/retry identities, a real AIAgent
Codex incomplete→complete loop, and the real child DB routing helper. Provider I/O
is synthetic; no live subscription was used. Existing Codex loop, first-chunk,
fallback-state, and usage-attribution suites are also exercised. This is not an
end-to-end live credential-rotation or live provider-fallback certification.

Explicit exclusions / partial coverage:

- Auxiliary direct clients (`auxiliary_client` compression, titles, vision, etc.)
  use relay APIs rather than this normalized post hook. Iteration-limit summaries
  call `_run_codex_stream` directly. Background review deliberately clears its DB.
- Codex app-server runtime has turn-level accounting and bypasses this seam. It is
  excluded rather than pretending a provider-owned multi-request turn is one HTTP
  attempt. External Codex CLI, other devices, other profiles and other providers
  are excluded. Scheduled/CLI/gateway work is covered only when using this ordinary
  AIAgent loop with a usable profile SessionDB.
- Transport/SDK-internal failures/retries do not each return usage to the outer
  callback. Only a returned response that reaches the post seam is recorded;
  no usage or request-count claim is made for unobserved wire attempts.
- Exceptions, cancellation/redirect crossings, malformed responses rejected by
  `check_api_response`, refusals or truncation paths returning before normalization,
  and any crash before the post seam/commit are not reconstructed. Some of these
  can have consumed tokens. Normalizing a returned error status is not proof of
  zero consumption. No historical session totals are converted into events.
- Execution middleware short circuits are not actual requests and create no event.
  Middleware making multiple callback calls exposes only its final observation;
  replacing the response identity is detected as incomplete, not attributed by
  guesswork. Bypassing the canonical seam remains outside this slice.
- Agents without a DB expose an in-process incomplete flag and sanitized warning;
  they do not silently open the process-default profile. No telemetry is persisted
  for them. This avoids both profile leakage and unexpected DB ownership.

Consequently, timeline coverage is always `partial` even when there are no known
write failures. Never present its zero subtotal as zero account consumption.

## Failure isolation and retention

Ingestion returns `False` on failure, never fails the conversation, and emits the
static warning `usage_event_recording_failed`. The same-process profile failure
counter is returned in timeline coverage. A successful later ingestion persists a
sticky `state_meta.usage_events_recording_incomplete` marker, visible to other
processes and read-only readers. The marker is deliberately not a per-window lost
request count. A locked/read-only/full disk cannot reliably persist its own failure;
if the process exits before recovery, only its log may survive. Coverage therefore
never asserts completeness on the absence of a marker.

Each write uses the existing SessionDB lock, `BEGIN IMMEDIATE`, jittered busy
handling, rollback/commit, and PASSIVE WAL checkpoint conventions with 100ms
application retry patience. Same-process lock admission follows SessionDB's existing
behavior and is not independently timeout-bounded in this slice. The existing SQLite busy handler
can itself wait about one second; this is NOT a 100ms hard wall-clock guarantee.
Existing periodic FTS/checkpoint work and filesystem stalls can add latency. No new
connection, writer queue, daemon, WAL reset, or busy-timeout mutation is introduced.

Each ingestion transaction removes at most 256 events strictly older than seven
days, using an indexed ordered subquery, and inserts at most one new event. An old
redelivery outside retention is not resurrected. `prune_usage_events()` offers one
explicit bounded batch for maintenance; `None` means failure, not zero deleted.
Reads NEVER prune, vacuum or write. Idle stores retain expired rows until ingestion
or explicit maintenance resumes, and a backlog drains over bounded batches. This
is logical retention, not secure erasure from SQLite free pages/backups.

## Backend timeline

`SessionDB.codex_usage_timeline()` takes no caller clock or provider parameter.
It chooses backend UTC `as_of_us` once and queries
`provider='openai-codex' AND completed_at_us >= as_of_us-6h AND completed_at_us < as_of_us`.
Integer-microsecond arithmetic produces exactly 24 contiguous 15-minute bins,
anchored to that same as_of (not rounded wall-clock quarters). SQL returns at most
24 aggregate rows, with an indexed provider/time range scan. The bin totals and
interval total are built from that same read snapshot, so concurrent ingestion
cannot make their sums disagree. Tests compare them to an independent interval SQL
sum at exact microsecond boundaries, including future timestamps.

`processed_tokens = input_tokens + output_tokens`. Cache/reasoning counts are detail
fields and are NEVER added again. All numeric values are known-token subtotals;
every token field has an `unknown_<field>` event counter. Empty/missing/invalid
observations are not fabricated as measured zero. Consumers must retain these
counters and the coverage block. A read failure or old read-only schema returns
`coverage.status='unavailable'`, `total=None`, and NULL bin usage, not a zero chart.

## Desktop/TUI read RPCs

`usage.codex_quota`, `usage.codex_timeline`, and `usage.active_work` accept no
parameters. The inherited TUI stdio pipe is a capability; WebSockets require the
existing upgrade authentication plus backend-stamped local telemetry admission.
Gated/cloud and non-loopback peers are unsupported. Loopback can be an SSH tunnel:
these results describe the backend device, never assert renderer-device locality.
All reads use the backend launch profile, not a renderer-selected session/profile.
Responses have version, opaque profile scope, backend timestamp, freshness and
coverage metadata; successful result JSON is capped at 32 KiB. Malformed IDs are
rejected before parameter validation, without reflecting their contents.

Timeline RPCs open only the launch profile state.db with SQLite mode=ro, no
migrations or maintenance, a 100ms busy timeout and a 250ms SQL progress budget.
Filesystem stalls are not a hard wall-clock guarantee. They filter the event
profile fingerprint as well as provider/time; moved foreign history is excluded.
Unavailable stores retain exactly 24 null bins, not fabricated zero usage.

Quota reads only the launch-profile singleton credential and the fixed usage
endpoint, without runtime recovery, pool rotation, refresh, or external CLI import.
A credential change during retrieval discards the result. Account identity remains
unverified: pool-only accounts, account-bound cache/single-flight/backoff and
credential refresh are not completed by this slice. Each call is uncached, with
a 15-second HTTP timeout; consumers must not poll aggressively. No retrieval
method invokes an LLM or enumerates arbitrary OS processes.

Active Work counts running TUI gateway turns and queued prompts, exact live
owner-record delegations, and profile-scoped in-process cron fire owners. Registry
absence, legacy unscoped cron claims, contention or saturation return null/unknown,
not zero. At most 1024 records per registry are inspected; no task content or IDs
are emitted. Coverage is always partial: other processes, other delegation parents
and non-TUI queues are not observed. Calm/busy/heavy means the sum of known
category counts (busy >=1, heavy >=4), not proof of whole-device idleness.

## Reproducible measurement

Run `python scripts/benchmark_usage_events.py --events 10000 --queries 100` using
the project Python. It creates a disposable profile and no network requests.
Workload: 10,000 one-event transactions uniformly distributed across six hours
(1,000 input / 100 output / 500 cache-read / 50 reasoning tokens); 100 timeline
queries; then 1,000 full observation+record helper calls. Every query verifies
11,000,000 processed tokens and bin-sum equality. Storage growth is allocated
SQLite page growth, not peak WAL size. It measures a fresh synthetic DB, not a
multi-GB production transcript store or a saturated machine.

Initial native Windows measurement (Python 3.11.16, SQLite 3.53.1, WAL,
`synchronous=FULL`, while a regression run was active):

| Operation | Mean ms | Median ms | P95 ms | Max ms |
| --- | ---: | ---: | ---: | ---: |
| Single-event ingestion | 2.225 | 2.002 | 4.148 | 26.636 |
| 10,000-event timeline | 18.526 | 17.096 | 23.668 | 56.743 |
| Observation + ingestion | 2.534 | 2.115 | 4.905 | 19.283 |

Allocated DB growth: 1,642,496 bytes. These are measurements, not latency SLAs.
