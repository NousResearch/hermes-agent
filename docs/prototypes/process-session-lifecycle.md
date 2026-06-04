# tmux-derived process-session lifecycle prototype

**Kanban task:** `t_d9c66e7b`
**Workspace:** `/home/filip/.worktrees/t_d9c66e7b`
**Branch:** `wt/tmux-process-session-lifecycle-202605`
**Status:** PROTOTYPE — disabled-default, purely additive, NOT wired into runtime

## Provenance

This prototype draws concepts from the read-only tmux source spike at
`/home/filip/spearhead-execution/20260528-source-spikes/tmux-tmux/source-spike.md`.
The spike commit pin: `f0669334189995dba860f59c3cf9cb12ae15865c`.

**No tmux source code is copied.** The C/libevent/pty machinery in tmux is
irrelevant to Hermes (Python/threaded/asyncio). What is borrowed are *patterns*
expressed in Hermes idioms:

| tmux source                                       | tmux concept                                     | Hermes adoption                                                                            |
| ------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| `job.c:41-65, 337-386`                            | JOB_RUNNING / JOB_DEAD / JOB_CLOSED with delayed cleanup; complete callback fires only when both fd and process are settled | `LifecycleState.RUNNING → EXITED → STREAM_DRAINED → CLOSED` with idempotent forward-only transitions |
| `server-fn.c:336-374`, `options-table.c:1442-62`  | `remain-on-exit` + `remain-on-exit-format`       | `DeadSessionSummary` + `RetentionPolicy.RETAIN_ON_FAILURE` default                         |
| `cmd-pipe-pane.c:93-100, 193-235`                 | `-o` "open if absent" pipe registration; delayed pane destruction until pipe drained | `IdempotentSubscriberRegistry.register()` returns existing handle on duplicate id; `STREAM_DRAINED → CLOSED` invariant |
| `tmux.1:7926-7928, 6443-6445`                     | control-mode "too far behind" exit reason; discarded-bytes status | `SubscriberBackpressure` with `too_far_behind_threshold`                                   |
| `proc.c:36-117`                                   | one dispatch path per peer; broken sinks marked `PEER_BAD`, not crashing siblings | `IdempotentSubscriberRegistry.broadcast()` isolates failing sinks via backpressure counter |
| `spawn.c:89-119, 231-271, 313-321, 342-350`       | respawn with kill-or-refuse semantics; persisted launch metadata | **Deliberately not implemented.** See "Respawn note" at the bottom of `process_session_lifecycle.py`. Future card with idempotency + approval gates required. |

## Why this lives outside `tools/process_registry.py`

`tools/process_registry.py` (~1600 LoC, ~1170 LoC of tests) is the production
process-tracking surface used by `terminal(background=true)` and the gateway
session-reset machinery. Direct edits would either:

1. Make the disabled-default constraint hard to honor — the existing
   `ProcessSession` dataclass is referenced by checkpoint serialization,
   gateway watchers, and the LLM-facing `process` tool schema.
2. Conflate the lifecycle FSM (testable without subprocesses) with the I/O
   plumbing (subprocess.Popen, ptyprocess, sandbox env.execute) — exactly
   the conflation tmux avoids by separating `job_t.state` from the
   bufferevent.

The prototype is therefore additive: a standalone module with its own tests.
Adoption — if approved — would be incremental:

- Phase 1 (no runtime risk): use `ProcessSessionLifecycle` in a feature-flagged
  branch of `_move_to_finished` to *also* produce a `DeadSessionSummary`,
  without removing the existing exit/output handling.
- Phase 2 (still safe): route the existing `watch_patterns` + `notify_on_complete`
  delivery through `IdempotentSubscriberRegistry` so duplicate watch_patterns
  registrations from race-y reconnects collapse cleanly.
- Phase 3 (requires approval): replace ad-hoc `session.exited` flag with the
  FSM state, deleting the orphaned-pipe reconciler workaround (which would
  become a regular `abandon_stream("orphaned descendant")` call).

## Lifecycle invariant

```
RUNNING  ─┐
          ▼   mark_exited(exit_code, signal=?, reason="exited"|"killed"|"lost"|"abandoned")
EXITED   ─┤
          ▼   mark_stream_drained() | abandon_stream(reason)
STREAM_DRAINED
          ▼   close(output_tail, log_pointer)
CLOSED
```

Transitions are monotonic. Backwards or skip transitions raise
`LifecycleTransitionError`. Idempotent re-entry (same target state, same
payload) returns `False` instead of raising — this is what makes the FSM
race-safe under the existing pattern where reader threads and the
gateway-side reconciler can both detect "process exited".

The split between `EXITED` and `STREAM_DRAINED` is the central guard:
the existing `tools/process_registry.py::_reader_loop` already sets
`session.exit_code` BEFORE moving to finished, but the bug class is real
(see issue `#17327` referenced in `_reconcile_local_exit`) — exit happens
on one path, stream cleanup on another, and conflating them lost either
the exit code or the final output. The FSM makes that impossible by
construction.

## Retention policy

Default: `RETAIN_ON_FAILURE`. A summary is generated and held only when:

- `exit_code` is non-zero, OR
- `exit_reason` is `killed`, `lost`, or `abandoned`, OR
- `exit_code` is `None` (unknown — treat as failure).

This matches the tmux pattern of treating `remain-on-exit` in "failed" mode
as the useful default, and aligns with the spike's observation that
durable Kanban metadata should not carry full log dumps — only pointers
and a bounded tail.

Configurable to `ALWAYS` (debug) or `NEVER` (caller does its own logging).

## Safety guardrails encoded by the prototype

| Guardrail                                       | Where                                              | Why                                                       |
| ----------------------------------------------- | -------------------------------------------------- | --------------------------------------------------------- |
| No respawn API                                  | comment block at bottom of `process_session_lifecycle.py` | Agent side-effect duplication is dangerous; requires its own approval gate. |
| `output_tail` bounded by configurable byte limit | `ProcessSessionLifecycle.__init__(output_tail_limit_bytes=4096)` | Avoid unbounded summary growth feeding Kanban metadata. |
| Subscriber registration refused after CLOSED    | `IdempotentSubscriberRegistry._closed`             | Sinks cannot accidentally attach to finalized sessions and miss everything. |
| Failing subscriber isolated from siblings       | `broadcast()` try/except → `record_discard()`      | One flaky watcher cannot break others.                    |
| `too_far_behind_threshold`                      | `SubscriberBackpressure`                           | Slow consumers eventually surface, mirroring tmux control mode. |
| Summary is frozen (`@dataclass(frozen=True)`)   | `DeadSessionSummary`                               | Post-mortem cannot be silently mutated after the fact.    |
| Conflicting `mark_exited` payload raises        | `mark_exited()`                                    | Exit-code drift (e.g. reconciler reading proc.poll() while reader sees pty exit) is a bug, not a race — surface it. |

## Out of scope / non-goals (per task body)

- No direct tmux code import.
- No terminal UI / pane / layout clone.
- No broad process architecture rewrite.
- No production/live enablement, deploy, credentials, public/client effects.
- No automatic respawn of agents with side effects.

## Test surface

`test_process_session_lifecycle.py` (~330 LoC, 36 tests):

- State enum / ordering sanity
- All forward transitions, including failure-path summary retention
- All invalid transitions (premature drain, premature close, conflicting exit code, abandon-before-exit)
- Idempotency under reader/reconciler races (concurrent `mark_exited`)
- Abandon path (orphaned-pipe analogue) records a note in the summary
- Retention policy: `RETAIN_ON_FAILURE`, `ALWAYS`, `NEVER`
- `DeadSessionSummary`: timestamps, byte-bounded tail, frozen, log_pointer, backpressure snapshot
- `IdempotentSubscriberRegistry`: register/replace/unregister, broadcast, failing-sink isolation, post-close refusal
- `SubscriberBackpressure`: discard counting, `too_far_behind` threshold
- Notes: appear in summary, refuse post-close mutation
- Snapshot consistency across all states
- Thread safety: 8 concurrent `mark_exited`s → exactly one winner; 8 concurrent registrations of same id → identical handle

No subprocesses are spawned in the tests. The FSM is intentionally
testable without I/O so all edge cases are deterministic.

## Files in this worktree

- `process_session_lifecycle.py` — the prototype module (~470 LoC, well-commented)
- `test_process_session_lifecycle.py` — pytest suite (~330 LoC)
- `DESIGN.md` — this document
- `README.md` — orientation + how to run

## Open questions / follow-up gates

1. **Wiring approval:** Phase 1 adoption (additive summary alongside existing
   `_move_to_finished`) requires Filip approval before being added to
   `tools/process_registry.py`. Tracked as: NEEDS FILIP APPROVAL — phase 1 wire-in.
2. **Respawn gate:** A separate Kanban card with idempotency-key + checkpoint
   contract is required before any respawn semantics are implemented.
3. **Log pointer schema:** `DeadSessionSummary.log_pointer` is currently a
   free-form string. If Kanban metadata is to consume this, the existing
   evidence-store schema should be checked for fit.
4. **Subscriber id space:** `IdempotentSubscriberRegistry` uses opaque string
   ids. The existing `watch_patterns` rate-limiter keys off session + per-session
   state; if adopted, the id convention should be defined
   (e.g. `"watch:<session_id>:<pattern_hash>"`).
