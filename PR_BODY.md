feat(telemetry): opt-in per-turn waterfall tracing (HERMES_TURN_TRACE)

## Problem

When a turn feels slow there is no way to tell where the time went. A single user message crosses the gateway asyncio loop, the run_sync executor, the per-LLM-call worker, and concurrent tool executor threads; existing logs record isolated durations at best, so hermes-added overhead (persistence, compression preflight, context assembly, inter-tool delays, delivery) cannot be separated from model inference time, and regressions in the non-LLM portion of the turn go unnoticed until they are large.

## Change

Adds an opt-in, zero-dependency tracing instrument:

- `agent/turn_trace.py` — span collector gated by `HERMES_TURN_TRACE=1` (default off). One JSON line per turn is appended to `~/.hermes/logs/turn_traces.jsonl` (override: `HERMES_TURN_TRACE_FILE`; 64 MB single-rotation cap; single `O_APPEND` write per record so a gateway daemon and a CLI run can share the sink). Spans store absolute wall-clock start/end; parent/child nesting is derived at render time from interval containment, so cross-thread and overlapping (concurrent tool batch) spans need no stack bookkeeping.
- `agent/turn_trace_render.py` — `python -m agent.turn_trace_render` renders traces as a terminal waterfall (plus `--summary` cross-turn aggregation and `--html` export; `--demo` renders a built-in sample).
- Span sites across the turn lifecycle: gateway ingest / session resolve / transcript load / hygiene / agent setup; turn prologue (system-prompt restore, early persist, compression preflight, pre-LLM hook, memory prefetch); per-iteration context assembly, request setup, LLM call (with TTFT and error attempts), accounting; tool batches including per-tool calls and the inter-tool delay sleep; verify gates; finalize children (trajectory, persist, post hooks, memory dispatch, resource cleanup); gateway persist; transport delivery.

Trace ownership: for gateway turns the runner `begin()`s the trace at message ingress and the platform adapter `finish()`es it after delivery, so the record covers the full ingress-to-delivery interval; CLI turns begin/finish inside `run_conversation`. Because a turn crosses threads, the trace is carried explicitly — bound to objects the instrumentation sites already share (the event, its `SessionSource`, the agent instance) with a thread-local current as a same-thread convenience — never via ambient globals that could bleed across concurrent sessions.

## Correctness notes

- **Default off is a strict no-op**: every site is guarded by `trace is not None` (or the no-op-safe `turn_trace.span()` which returns a shared null context manager), so the disabled cost is one env/attribute check per site. No behavior change when disabled.
- **Tracing can never break a turn**: sink emission catches all exceptions (unwritable sink, non-serializable tags fall back to `str`), `bind()` swallows `setattr` failures on slotted objects, and span context managers record-and-reraise without altering control flow.
- **Pre-dispatch event replacement**: the `pre_gateway_dispatch` hook loop may swap the runner's event for a `dataclasses.replace` copy (prepend/rewrite directives), while the adapter's finish site holds the original event. The trace is therefore also bound to the shared `SessionSource`; finish sites resolve event-then-source and clear the source binding afterwards so a stale trace cannot leak into a later turn on a reused source.
- **Lifecycle safety**: `finish()` is idempotent and first-status-wins; adapter cancel/error paths and a belt-and-braces `finally` all funnel through it. Cached agents outlive the turn, so the gateway clears the agent binding in a `finally`; pooled executor threads adopt the trace on entry (a previous turn's trace is already finished and inert). Span appends are lock-protected for concurrent tool executors, and `finish()` snapshots spans/tags before serializing.
- Inline dispatches that bypass the adapter's background processing (bypass commands, clarify text-intercept, kanban watcher synthetic events) own the finish themselves so those records are not dropped.

## Tests

`tests/agent/test_turn_trace.py` (22 tests): disabled-path no-ops, full-cycle JSONL emission, idempotent finish, post-finish spans ignored, span sorting, bind/resolve precedence (explicit > bound > thread-local), worker-thread adoption, unwritable-sink survival, error tagging and re-raise, non-serializable tag fallback, concurrent span appends, and renderer demo mode.

Suites run on this branch: `tests/agent` (5490 passed), `tests/gateway` + `tests/plugins` + `tests/run_agent` + `tests/scripts` (12876 passed) — failure sets are identical to a clean checkout of `main` run in the same environment (environment-dependent and order-dependent failures only; every difference between the two runs passes repeatedly when run standalone on this branch). `tests/run_agent` alone on this branch: 1974 passed, 0 failed.

## Measured impact

Used in production to attribute a "turns feel ~30% slower" report to concrete, fixable causes — first-call prompt-cache loss costing ~20 s per turn at 150k context, fixed 1.0 s inter-tool delay sleeps, ~19 ms-per-call HTTP client rebuilds, and synchronous persistence/accounting work sitting on the turn thread — findings that seeded the follow-up perf PRs #64169–#64172. When disabled (the default), overhead is a single env check per instrumentation site; when enabled, per-turn cost is one JSON serialization and one appending write.
