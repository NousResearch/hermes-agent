# Startup-restore fail-open — progress ledger

Task: `t_0861a67d`
Branch: `fix/startup-restore-fail-open-t-0861a67d`
Base: latest `fork/main`

## Diagnosis

- Default-gateway incident window grounded from `~/.hermes/logs/agent.log`: first queued inbound at 11:55:43; additional inbound remained gated through 12:11:06; the longest boot-resume turn finished at 12:15:39; queue drain completed at 12:15:42.
- PR #256's timeout warning was absent. Default timeout was 30s and had no config/env override.
- Aegis telemetry proved H3 (task-set race / detached wrapper): at 12:48:53.670 `_finish_startup_restore` entered with one tracked task; at 12:48:54.784 the wait returned `done=1, pending=0`; the actual internal resume event entered the runner only at 12:48:54.797 and its turn completed at 12:49:07.955. The #256 timeout watched the dispatch wrapper, not the actual turn.
- The same Aegis cycle proved the remaining gate-critical await: a startup-window poke queued at 12:48:53.915; replay entered `gateway/run.py::_drain_startup_restore_queue -> await adapter.handle_message(event)` at 12:48:54.784 and exited 1.007s later; only then did the gate release at 12:48:55.791. That await was unbounded in the 20-minute incident.
- Apollo added log-only `PHASE=startup_restore_*` instrumentation to the shared runtime tree. It records gate flips, `_finish_startup_restore` entry/task count/timeout, wait exit, and each queued replay enter/exit.
- Re-wedge risk checked before the rig cycle: default session `agent:main:discord:thread:1524057034696425765:1524057034696425765` still has `resume_pending=true`; Apollo's default gateway is not being restarted.

## RED/GREEN worktree tests

- RED on unmodified `fork/main`: 3/3 targeted regressions failed — blocked replay retained the gate, no absolute watchdog existed, and a wait exception escaped without releasing the gate.
- GREEN candidate fix in the worktree only: release the gate before queued replay, arm an absolute timeout watchdog when the gate is first raised, and transfer queued events to one background replay owner that retains/retries unaccepted events.
- Updated RED on unmodified `fork/main`: all 14 targeted fail-open, ownership, ordering, scheduler-yield, retry-noise, adapter-contract, wiring, and fallback-observability regressions failed.
- Momus pass 1 blocked on pop-before-await ownership and pre-registration adapter loss. The worker now peeks instead of popping, retries failures/unavailable adapters with backoff, and serializes watchdog/finisher replay through one task.
- Momus pass 2 relocated the remaining ownership blocker to cancellation and required same-session FIFO, an explicit adapter acceptance contract, bounded retry noise, and corrected timeout docs. The live-gateway cancellation path now transfers the retained queue to a successor owner; unrelated sessions may pass a blocked one without overtaking within that session; `BasePlatformAdapter.handle_message` documents exception-before-acceptance; retry warnings include queue age/depth and are rate-limited to one per event per minute.
- Ace's follow-up observability directive is included: default prompt-mode `PHASE=boot_resume_scheduled` bytes remain frozen, while an auto-mode fallback logs `mode=prompt fallback_reason=...`; auto scheduling already logs `mode=auto`.
- Momus pass 3 caught a cross-session no-yield retry loop and WARNING-level replay-enter spam. Every failed attempt now yields through exponential backoff even when another session is promoted, and replay-enter warnings are rate-limited per event; regressions prove two permanently blocked sessions cannot starve an unrelated coroutine and repeated dispatch failures emit bounded warnings.
- Momus pass 4 had no blockers and requested the missing keeper arm event; pass 5 APPROVED with no required changes after the WARNING-level `gate_flip state=armed` event and production ordering contract were added.
- Final full restart-family regression gate: 185 passed across `test_restart_resume_pending.py`, `test_restart_cascade.py`, and `test_auto_continue_interrupted_turns.py`; `py_compile` and `git diff --check` also passed.

## Live test venue

- Aegis gateway only; Apollo/default gateway remains untouched.
- Instrumented Aegis rig transcript captured at 12:48:47-12:49:07 PT. It proves H3 and the replay await above without restarting Apollo/default.
- Behavioral fix is committed in fork PR #301 (`1916afc`) but is not deployed. Shared runtime received log-only instrumentation for the rig and was restored clean immediately after capture.

- `SessionStore.mark_resume_pending` holds the store lock, rejects suspended sessions, mutates the entry, and calls `_save()` synchronously before returning (`gateway/session.py:2191-2218`). No extra flush API is required for ordinary marks.
- `suspend_recently_active` also performs one synchronous `_save()` after its in-memory marking loop (`gateway/session.py:2297-2331`).
- Divergence/constraint: crash injection after in-memory mark but before `_save()` (T9o) cannot be represented by calling the public method as-is. Boot reconciliation needs an explicit internal two-step test seam (mark mutation, crash hook, synchronous save) while production still preserves the public single-call contract.

### 4. F1/F2 breaker and replay-mark sites

- F1/programmatic restart records the initiating session before setting global restart state (`gateway/run.py:8060-8073`).
- C1 recognizes an executed `safe-restart.py` terminal call (not mere mention) and sets the same per-session flag (`gateway/run.py:20006-20032`).
- Clean-turn F2 consumes the breadcrumb unconditionally, ORs it with the in-memory flag, records a replay mark for restart-initiating turns, otherwise clears marks after genuine work (`gateway/run.py:7576-7661`).
- Breadcrumb validation/consumption is per-session, same-boot, TTL-bounded, and single-use (`gateway/run.py:7663-7758`). It currently unlinks while validating, so deferred arm requires a dedicated consume-and-return validation result (or equivalent) rather than a second consume.
- The reusable replay API is `_record_restart_replay_mark(session_key, now=...)` (`gateway/run.py:7523-7564`). It is not request-id-idempotent today. Cross-boot deferred reconciliation therefore needs request-id dedup persisted alongside breaker state before it can satisfy T9e/T9j.
- Drain-time marking records a replay mark before `mark_resume_pending` (`gateway/run.py:6707-6724`), and the hard timeout marks only genuinely still-running agents before interrupting them (`gateway/run.py:10237-10309`).

### 5. #269 resume-request dropbox read/write sites

- External writes are atomic temp+fsync+replace at `gateway/resume_requests.py:51-83`.
- Current reads enumerate only `*.json`, parse, then unlink *before* returning the request (`gateway/resume_requests.py:86-142`). Malformed files become `*.rejected`; stale files are deleted.
- Gateway folding is `_sweep_resume_requests` (`gateway/run.py:8389-8419`). It immediately calls synchronous `mark_resume_pending`.
- Reads occur before tail classification (`gateway/run.py:8211-8218`), again at scheduler entry before candidate enumeration (`gateway/run.py:8446-8457`), and from housekeeping when a `.json` file exists (`gateway/run.py:23016-23030`).
- Divergence from new lifecycle: the existing sweep's same-pass unlink cannot support submitted→armed→claimed→terminal boot-owned cleanup. Existing plain resume requests must retain current behavior; `deferred_restart` needs a typed lifecycle API that does not flow through the destructive tuple sweep.

- RED before implementation:
  `scripts/run_tests.sh tests/cron/test_scheduler.py -q` -> `226 passed, 1 failed`;
  the new worker test failed because the authoritative target was absent.
- GREEN after implementation:
  the same command -> `227 passed, 0 failed`.
- Mutation check: removed the single prompt-binding call and reran the same file
  -> `226 passed, 1 failed`; restored the call and reran -> `227 passed, 0 failed`.
- Related gateway/send coverage:
  `scripts/run_tests.sh tests/gateway/test_session_context_inheritance.py tests/tools/test_send_message_origin.py tests/tools/test_send_message_tool.py -q`
  -> `172 passed, 0 failed`.
- Lint: shared-venv `ruff check cron/scheduler.py tests/cron/test_scheduler.py`
  -> `All checks passed!`; `git diff --check` passed.
- Deterministic pre-scan ran Bandit, Ruff, and Semgrep. Whole-file mode reported
  existing repository/test-file findings; none are on the added production lines.
- Momus review transport could not start because its configured
  `opus-review-direct.py` path is absent under the Daedalus profile. The task is
  therefore handed off `review-required` rather than self-approved.
---

# delegate_task restart-survival build progress

Task: `t_50d839af`
Spec: `~/.hermes/plans/2026-07-10_delegate-restart-survival-spec.md`
Momus review: `~/.hermes/plans/2026-07-10_delegate-restart-survival-momus-p1.md`

## Status

- Rebased this worktree onto `fork/main` before implementation and again after
  upstream advanced; `fork/main` is an ancestor of the feature commit.
- Added a profile-scoped, owner-only durable registry at `$HERMES_HOME/state/async-delegations.json`.
- Added persist-before-submit, submission telemetry, attempt fencing, dead-boot claiming, a two-relaunch breaker, restart/terminal outbox replay, durable acknowledgement, retention/cap enforcement, and record integrity checks.
- Added a shared live/resume adapter in `tools/delegate_tool.py`; recovery reconstructs the original single/batch unit with continuation instructions and re-resolves current credentials under the originating profile.
- Wired one recovery pass per gateway boot, including degraded boots with no
  connected adapters. Restart and terminal events continue through
  `process_registry.completion_queue` and the existing
  `_async_delegation_watcher`; no second drain or conversation-history mutation
  was added.
- Wired bounded best-effort `recoverable` marking on gateway shutdown. Dead-owner recovery of a still-`running` record is the guaranteed fallback.
- Wired `/stop`, `/new`, parent, and session cancellation to terminally cancel
  durable work before signalling live children, including detached work with no
  resident foreground agent.
- Added `delegation.resume_on_restart: true` to both config default surfaces and documented it.
- Added a loud registry-cap log and `sync_fallback_registry_cap` observability counter; unpersistable work degrades to synchronous execution.

## Momus required changes folded

- **RC-1 folded:** a dead boot's claimed-but-never-submitted generation is reconciled without incrementing `redispatch_count`; only durable `submitted_at` telemetry consumes one of the two replacement launches. Named test: `test_rc1_claimed_but_never_submitted_does_not_count_attempt`.
- **RC-2 folded:** restart and terminal outbox entries both replay after the queuing boot dies. A restart event from an older generation becomes terminal `dropped(superseded)` rather than remaining pending. Named test: `test_rc2_pending_restart_event_replays_and_superseded_event_drops`.
- **RC-3 folded:** clean-shutdown `recoverable` marking uses a bounded lock timeout and is explicitly best-effort. Recovery treats a dead owner's `running` record equivalently.
- **RC-4 folded:** all anchors were re-grounded against current `fork/main`; event delivery extends the existing `process_registry.completion_queue`, gateway watch drain checkpoint, and `_async_delegation_watcher`.

## Re-grounded anchor map

| Surface | Current anchor |
|---|---|
| Child construction / execution | `tools/delegate_tool.py:1146` `_build_child_agent`; `:2010` `_run_single_child` |
| Live/resume adapter | `tools/delegate_tool.py:2663` `_build_durable_background_spec`; `:2798` `build_recovered_delegation_runner`; `:2847` `delegate_task` |
| In-memory dispatch / recovery | `tools/async_delegation.py:153` single dispatch; `:443` batch dispatch; `:706` recovery; `:851` outbox replay; `:930` cancellation/shutdown; `:975` session cancellation |
| Durable registry | `tools/async_delegation_store.py:35` path; `:126` lock; `:303` persist; `:368` submit telemetry; `:435` terminal; `:540` durable cancellation; `:652` claim; `:811` replay; `:847` acknowledgement |
| Gateway boot/profile scope | `gateway/run.py:1805` `_profile_runtime_scope`; `:7560` `_current_boot_id`; `:8424` startup |
| Gateway fresh-turn delivery | `gateway/run.py:17772` injection; `:17852` once-per-boot recovery; `:18030` existing async-delegation watcher; `:18722` explicit session cancellation |
| Shared completion rail | `tools/process_registry.py:173` `completion_queue`; `:2080` `format_process_notification` |
| Boot identity | `gateway/status.py:142` process start; `:182` current boot ID; `:187` liveness; `:443` boot ID construction |
| Config defaults | `hermes_cli/config.py:976` `DEFAULT_CONFIG`, delegation at `:2278`; CLI fallback at `cli.py:499` |

## Verification log

Canonical runner: `scripts/run_tests.sh`.

- Initial RED: `tests/tools/test_async_delegation_persistence.py` — 13 failed because durable APIs did not exist.
- Core GREEN: persistence + existing async delegation — 34 passed.
- Broad focused pass: 5 files / 348 tests passed.
- Gateway recovery/routing pass: 2 files / 34 tests passed.
- Expanded persistence pass: 21 tests passed.
- Shared delegate resume adapter pass: 3 tests passed.
- Final post-rebase broad focused pass: 12 files / 506 tests passed, 0 failed.
- Exact Momus gate:
  `test_rc1_claimed_but_never_submitted_does_not_count_attempt` and
  `test_rc2_pending_restart_event_replays_and_superseded_event_drops` — 2 passed.
- Shared-venv `compileall` passed for all changed runtime modules.
- Ruff passed for all changed Python files.
- `git diff --check` passes after removing two Markdown hard-break spaces from
  this progress file.
- Final diff-aware Bandit/Ruff/Semgrep pre-scan found one production finding:
  Semgrep flagged owner-only `0o700` directory permissions as more permissive
  than `0o644`; this is a scanner false positive because `0o700` grants no
  group/other access. Remaining added-line findings are test-only assertions and
  fixed `/tmp` fixture values.
- Independent review was not obtained: the configured Momus transport path is
  absent and the standing-profile fallback timed out after 500 seconds. Final
  disposition remains `review-required`.
### 6. Boot startup order

- Real startup order after adapters settle is restart notification, `_prepare_auto_resume_decisions`, `_schedule_resume_pending_sessions`, then `_finish_startup_restore` (`gateway/run.py:9315-9345`).
- `_prepare_auto_resume_decisions` currently sweeps the dropbox before snapshot/classification (`gateway/run.py:8199-8218`). `_schedule_resume_pending_sessions` sweeps again before locked candidate enumeration (`gateway/run.py:8421-8462`).
- Therefore boot deferred-restart reconciliation belongs at the start of `_prepare_auto_resume_decisions`, before the existing plain-request sweep and before scheduling. T9k must execute this real order, not call reconciliation in isolation.

### Grounded design decisions / divergences

1. Preserve plain #269 request behavior; add typed deferred-request lifecycle APIs beside it.
2. Reuse the common `SendResult`/post-delivery path as the barrier; do not invent per-platform durable receipts.
3. Add request-id-bearing SELF metadata to `SessionEntry`; current schema has only `resume_pending`, `resume_reason`, and timestamp (`gateway/session.py:715-720`, serialized at `775-783` and restored at `858-863`).
4. Extend the replay store with request-id idempotence; the current API is timestamp-only.
5. The safe-restart script and skill are outside this repository; spec §4 explicitly assigns the skill rewrite to Apollo post-merge. This branch will implement and test the gateway/dropbox contract and list the required external script change in handoff, without mutating the live skill tree.

## Phase 2 — RED/GREEN implementation

- Added explicit taxonomy constants and the authority mapper in `gateway/auto_resume.py`; persisted `resume_kind`, `resume_handoff`, and `resume_request_id` through `SessionEntry` and `SessionStore`.
- Added the typed deferred lifecycle in `gateway/deferred_restart.py:68-434`: atomic request publication, submitted→armed→claimed→consumed/rejected transitions, same-boot breadcrumb validation, epoch-owned `mkdir` CAS, atomic `meta.json` commit publication, committed-only loser coalescing, and boot-only terminal reconciliation.
- Added a real delivery-ack seam at `gateway/platforms/base.py:4098-4139`. It is generation-owned and fires only from the successful final-send path; confirmed streamed delivery directly acknowledges the armed callback before normal-send suppression. Handler-finally callbacks are intentionally not accepted as delivery proof, and ordinary streamed turns retain no confirmation cache.
- Wired boot reconciliation before the prompt/auto branch and candidate snapshot (`gateway/run.py:8254-8311`), release-time arm after running-state persistence (`gateway/run.py:18736-18855`), request-id replay dedupe (`gateway/run.py:7547-7612`), explicit SELF scheduling without SIBLING attempt credit, and `kind=sibling|self` schedule logs.
- Preserved plain #269 requests by making their sweep ignore `kind=deferred_restart` files.
- Added G1 tests in `tests/gateway/test_deferred_restart_taxonomy.py:51-900`, including T9a/b/c/d/j/k/m/n1..n6/o/p/q. T9n1..T9n6 correspond to every fallible post-`mkdir`/pre-signal boundary introduced: initial leader publication, claim persistence, replay mark, SELF mark, all-peer delivery completion, and committed-meta publication. Every boundary is exercised both with a concurrent loser and as the normal single-request case.

RED observations:

- Initial collection failed on missing `RESUME_KIND_SELF`.
- Added delivery/note contracts failed until release-time strong acknowledgement and SELF note wiring existed.
- Combined regression caught the intentional new taxonomy log and an unintended legacy SIBLING-credit bypass; the bypass was narrowed to explicit persisted `resume_kind=self`, preserving existing behavior for legacy reason-only rows.
- Tree review caught that existing post-delivery callbacks are finally callbacks, not send acknowledgements; the strong callback registry above replaced that unsafe assumption.

GREEN evidence via the repository virtualenv runner:

- `scripts/run_tests.sh tests/gateway/test_deferred_restart_taxonomy.py tests/gateway/test_auto_continue_interrupted_turns.py tests/gateway/test_restart_cascade.py -q` → 122 passed.
- `scripts/run_tests.sh tests/gateway/test_resume_requests.py tests/gateway/test_session.py tests/gateway/test_post_delivery_callback_chaining.py tests/gateway/test_restart_cascade.py tests/gateway/test_auto_continue_interrupted_turns.py tests/gateway/test_deferred_restart_taxonomy.py -q` → 249 passed.
- Final post-rebase command across resume requests, sessions, callback chaining, restart cascade, interrupted-turn behavior, and deferred taxonomy → 277 passed; latest deferred lifecycle file → 61 passed after staggered multi-initiator delivery, T9n6 coverage, injected arm-to-owner failures, host-reboot clock reset, real startup scheduling, stale-latch deletion failure, and loser backoff.
- `ruff check` on all changed Python files → all checks passed.
- Static pre-scan: bandit/ruff/semgrep, 0 HIGH/CRITICAL findings; low/medium output is baseline noise from scanning complete large files and pytest assertions.

Mutation evidence (each mutation was restored from the staged implementation immediately after the expected RED):

- Replace strong delivery-ack registration with finally callback → T2/T8 fails because no strong callback is armed.
- Remove SELF mark → T2/T8 fails (`resume_kind` remains `None`).
- Bypass `mkdir` CAS → T9c fails with two restart signals.
- Remove request-id dedupe → T9j trips the breaker one request early.
- Remove boot reconciliation from preparation → T9k fails in prompt mode.

## Phase 3 — adversarial review corrections

- Momus pass 1 identified generation fail-open and boot-reconcile retry ownership. Both were fixed and regression-tested; disputed signal ordering, legacy log mapping, and cross-boot submitted handling were dismissed by the authoritative spec/source pack.
- Pass 2 identified orphaning of a sole failed winner and malformed committed metadata. The coordinator now retries non-owning/injected precommit cancellation and fallible precommit exceptions with bounded backoff while propagating real external task cancellation to shutdown/boot ownership. Committed metadata requires a finite positive numeric `commit_ts` before any loser transition.
- Pass 3 identified an unbounded alternate-order stream-confirmation cache. Tree ordering proves release precedes streaming suppression, so the cache was removed; confirmed stream delivery directly fires an existing strong callback and ordinary delivery retains no state.
- Pass 4 identified cross-request delivery-barrier bypass. The elected leader now waits until every same-boot armed/claimed request has independently acknowledged delivery or timed out before atomically committing; a staggered two-initiator regression proves no early signal.
- Pass 5 identified an arm-to-task ownership gap when the durable `armed` transition succeeded but a later scan or callback-registration step failed. The coordinator now retains the exact armed/claimed request in memory, schedules its owner immediately after the durable arm, and treats callback-registration failure as UNKNOWN delivery while preserving eventual one-signal progress. Injected post-arm scan and callback failures both prove the request remains owned.
- Pass 6 found the same fingerprint at the narrower rename→payload-refresh boundary. Lifecycle state is authoritative in the successfully renamed filename, so ownership is now established immediately after rename and a failed redundant payload refresh is logged without abandoning the recoverable request. The exact post-rename failure is injected for both a sole request and a concurrent peer; both cases produce exactly one signal.
- Pass 7 certified the ownership fingerprint resolved, then found two new classes: persisted monotonic timestamps fail across host reboot, and T9k tested reconciliation/order proxies instead of the startup effect. Deferred intent/commit and boot-start ordering now use wall-clock timestamps, with a simulated monotonic-epoch reset proving the handoff survives. The startup prepare→snapshot→schedule→finish sequence is one binding helper used by `start()` and T9k; the real surviving request is scheduled as SELF in the same boot.
- Pass 8 certified both pass-7 classes resolved, then identified a non-yielding stale-latch deletion retry and fixed 10 ms loser polling. Failed stale deletion now verifies the directory remains and yields with bounded exponential backoff; all uncommitted/torn loser polling uses the same 10 ms→1 s bounded backoff. Injected regressions prove the event loop yields before deletion retry and the loser never falsely coalesces or signals.

NEXT: Commit, push/open the fork PR, wait for green CI, then arm auto-merge and record the CI/merge handoff. External safe-restart writer/skill wiring and the physical T6 rig remain Apollo/Aegis post-merge work per the authoritative sequence.
NEXT: Wait for PR #301 CI to turn fully green, then enable auto-merge; after merge, verify the merged runtime on Aegis first. Apollo deployment remains deferred to an Ace-approved combined bounce.
