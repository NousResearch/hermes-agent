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

NEXT: Wait for PR #301 CI to turn fully green, then enable auto-merge; after merge, verify the merged runtime on Aegis first. Apollo deployment remains deferred to an Ace-approved combined bounce.
