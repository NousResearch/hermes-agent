# FTD V2 Claude COI Review

**Date:** 2026-05-14
**Reviewer:** Claude Sonnet 4.6 (adversarial/COI mode)
**Inputs:** Final spec, Codex implementation report, all runtime scripts and tests

---

## Verdict

**PASS_WITH_FIXES**

The Phase 1/2 control-plane floor is directionally correct. The previously broken stop semantics, wrapper-exit-while-ACTIVE → ERROR, set_state PID preservation, and STARTING reservation are all implemented soundly. However, one acceptance criterion (#17: "checkpoint caps are machine-enforced") is completely unimplemented, the watchdog's SPAWNING transition has a timestamp bug that can produce premature false-ERROR cycles, and four of the required watchdog tests from the spec are absent.

---

## Blocking Issues

- [ ] **Missing checkpoint cap check in `ftd_watchdog.py` READY_NEXT_SPRINT handler.** Acceptance criterion #17: "Checkpoint caps are machine-enforced." The spec (§6.7, §5.1, §5.3) requires:
  ```
  case READY_NEXT_SPRINT:
    if checkpoint cap reached: set FEATURE_CHECKPOINT_READY_FOR_BENJAMIN and emit
    else increment sprints_since_benjamin_checkpoint → SPAWNING_NEXT_SPRINT
  ```
  The implementation (`ftd_watchdog.py:100-110`) blindly increments `sprints_since_benjamin_checkpoint` and spawns without ever reading `max_sprints_without_benjamin_review` from config or comparing the counter against it. `FEATURE_CHECKPOINT_READY_FOR_BENJAMIN` is **never set by the watchdog**; projects run indefinitely without a Benjamin checkpoint from the counter path. Required fix: read `max_sprints_without_benjamin_review` from `.fulltime-dev/config.yaml` (default 6) before spawning; if `sprints_since_benjamin_checkpoint >= cap`, set `FEATURE_CHECKPOINT_READY_FOR_BENJAMIN` and emit instead of spawning.

- [ ] **`spawning_started_at` assigned from stale `updated_at` in `ftd_watchdog.py:107`.** Code:
  ```python
  state["state"] = "SPAWNING_NEXT_SPRINT"
  state["spawning_started_at"] = state.get("updated_at")   # BUG
  save_state(state)
  ```
  `state.get("updated_at")` at this point is the old value from before `save_state` is called; `save_state` writes a new `updated_at = now_iso()` internally. If the project was in `READY_NEXT_SPRINT` for, say, 10 minutes, `spawning_started_at` is set 10 minutes in the past. On the **next** watchdog tick the staleness check `age_seconds(state.get("spawning_started_at")) >= SPAWNING_STALE_SECONDS (300s)` fires immediately — creating a spurious recovery/ERROR cycle against a spawn that may have just succeeded. Required fix: `state["spawning_started_at"] = now_iso()`.

---

## Non-blocking Issues / Risks

- **Lock held across `create_next_sprint` + `spawn_pm_runner` in `ftd_watchdog.py:100-110`.** Both `create_next_sprint` (network call: `hermes kanban create`) and `spawn_pm_runner` (`Popen` + `save_state`) are called while holding `project_lock`. Spec §6.7 says: "Do not hold `project_lock` while doing slow external CLI calls if avoidable. Use short critical sections: read/mark intent, release, call external command, reacquire and commit result." If `hermes kanban create` hangs, the watchdog process holds the project lock for up to `WATCHDOG_HARD_TIMEOUT_SECONDS` (90s), blocking `ftd_start.py`, `ftd_stop.py`, and `ftd_set_state.py` for any concurrent operator command. Recommendation: set `SPAWNING_NEXT_SPRINT` intent inside the lock, release, do `create_next_sprint` + `spawn_pm_runner`, reacquire lock to commit the resulting PID/ACTIVE state.

- **TOCTOU in watchdog ACTIVE dead-runner check (`ftd_watchdog.py:87-97`).** PID liveness is checked from the pre-lock state load:
  ```python
  child_live = process_alive(state.get("active_pm_child_pid"))   # pre-lock
  wrapper_live = process_alive(...)                               # pre-lock
  if not child_live and not wrapper_live:
      with project_lock(project_id):
          state = load_state(project_id)                          # reload
          state["state"] = "ERROR"                               # no re-check
  ```
  After acquiring the lock and reloading state, PID liveness is **not re-verified** from the fresh state. Race: user runs `ftd_start.py` concurrently, spawns a new runner with new PIDs and writes ACTIVE between the pre-lock check and the lock acquisition. Watchdog reloads the new ACTIVE state but unconditionally sets ERROR on it. The new PM runner's sprint would begin in ERROR. Mitigation: re-check `process_alive` on the newly loaded state's PIDs inside the lock before setting ERROR.

- **`ensure_project_watchdog` calls `save_state` outside any lock (`ftd_lib.py:276`).** `ensure_project_watchdog` is called in `ftd_start.py` outside the project lock (between inner-lock-1 and inner-lock-2). It receives the post-lock-1 state dict and calls `save_state(state)` without holding the lock. Narrow race: watchdog fires during this window (unlikely for STARTING, but possible), and the watchdog's write is overwritten by `ensure_project_watchdog`'s save, or vice versa. Low probability in practice but violates the intended lock discipline.

- **`--watchdog` flag in `ftd_status.py:99` has no effect — always `True`.**
  ```python
  status_payload(s, include_watchdog=args.watchdog or True)
  ```
  `args.watchdog or True` evaluates to `True` regardless of `--watchdog` presence, because `True` is always truthy. Watchdog status is always probed via `cron_job_exists` even in non-watchdog JSON output. This adds unnecessary `hermes cron list` calls on every `ftd_status.py --json` invocation and cannot be disabled. Fix: `include_watchdog=args.watchdog`.

- **`dispatch_board` called with pre-lock state in watchdog ACTIVE path (`ftd_watchdog.py:97`).** If state changes between the initial load and `dispatch_board(state)` (e.g., PM runner just finished and set READY_NEXT_SPRINT), the dispatch uses stale state that still says ACTIVE. The guard inside `dispatch_board` re-reads `state.get("state")` from the passed dict — that dict is from the pre-lock load. This can cause an extra dispatch tick after the PM runner has set READY_NEXT_SPRINT. Low severity: dispatch is idempotent and the next watchdog tick would behave correctly.

- **Cron `delete` subcommand correctness is unverified.** `ftd_stop.py:79` and `_delete_watchdog_if_known` use `hermes cron delete <job_id>`. If the Hermes CLI uses `remove` instead of `delete`, this silently does nothing (`check=False`). No test exercises the actual cron CLI path.

- **`preexec_fn` + `start_new_session=True` combination is untested end-to-end (`ftd_pm_runner_wrapper.py:73`).** The wrapper spawns the child with both flags set. The spec (§6.6) explicitly requires a tested signal/session strategy. Current tests use `FakePopen` and never exercise the actual `Popen` call. A real hermes child receiving SIGINT from its operator terminal session would not be caught by the mocked tests.

- **`spawn_pm_runner` does not write `active_pm_child_pid` before returning (`ftd_lib.py:708-719`).** The child PID is written by the wrapper process (after it spawns the hermes child). Between `spawn_pm_runner` returning and the wrapper writing `active_pm_child_pid`, the liveness check sees ACTIVE with no child PID. The watchdog guard `not process_alive(child_pid) and not process_alive(wrapper_pid)` uses `child_pid = None → False` but `wrapper_pid` is alive, so no false error. However, if the watchdog fires in this window, `dispatch_board` dispatches with a state that has no child PID — the dispatch guard passes only because wrapper is alive. Acceptable for Phase 1 but a documentation gap.

---

## Test Gaps

- **No test for checkpoint cap enforcement.** Spec acceptance criterion #17, spec test requirement #6 (`test_watchdog_pauses_checkpoint_state_and_notifies_once`). Since the checkpoint cap logic is also missing from the implementation, the gap is doubly confirmed. Proposed test: write state with `sprints_since_benjamin_checkpoint >= max_sprints_without_benjamin_review`, mock `kanban_create` and `spawn_pm_runner`, assert watchdog sets `FEATURE_CHECKPOINT_READY_FOR_BENJAMIN` and does NOT spawn, and that `_emit_once` returns a non-empty message.

- **No test for `test_watchdog_errors_stale_spawning_next_sprint` (spec test #8).** Required by spec §11.2. Proposed test: write `SPAWNING_NEXT_SPRINT` state with `spawning_started_at` set to 600 seconds in the past and no `spawning_sprint_task_id`; assert watchdog sets ERROR and emits once.

- **No test for `test_watchdog_recreates_missing_cron_job` (spec test #7).** Required by spec §11.2. The stale cron detection path in `ensure_project_watchdog` is not tested. Proposed test: write state with an existing `watchdog.cron_id`, mock `cron resume` to fail and `cron_job_exists` to return False, mock `cron create` to return a new job ID; assert state and global config contain the new job ID.

- **No test for `test_watchdog_does_not_dispatch_in_ready_next_sprint` (spec test #5).** Required by spec §11.2. The dispatch guard is tested via `test_dispatch_guard` which covers READY_NEXT_SPRINT→ACTIVE transition correctly, but there is no explicit test asserting dispatch is NOT called during the READY_NEXT_SPRINT watchdog handler path. Proposed test: write READY_NEXT_SPRINT state, mock `kanban_create` and `spawn_pm_runner`, intercept any call to `hermes kanban … dispatch`, assert dispatch is not called.

- **`test_wrapper_exit_zero_while_active_sets_error` (`test_ftd_control_plane.py:265`) uses `FakePopen` which bypasses actual subprocess and signal handling.** The test verifies state transitions correctly, but the SIGINT/SIGHUP preexec behavior and session isolation are untested. A canary-level test (spec §12, Phase 9) would catch regressions here.

- **`test_watchdog_ready_next_sprint_spawns_next_pm` does not assert `sprints_since_benjamin_checkpoint` increment.** The counter is incremented at `ftd_watchdog.py:105`, but the test only asserts that `spawn_pm_runner` was called and state is ACTIVE. The counter check should be added given the cap logic will be added.

---

## Spec Deviations

- **Checkpoint cap not implemented (§5.1, §6.7, §11.2 test #6, acceptance criterion #17)** — **not acceptable**. This is listed as a hard acceptance criterion and a machine-enforced invariant. The counter infrastructure is present; the cap check is entirely absent.

- **`spawning_started_at` assigned from `state.get("updated_at")` instead of `now_iso()` (§6.7)** — **not acceptable**. The spec writes: "set SPAWNING_NEXT_SPRINT with timestamp." Using an old timestamp creates an incorrect staleness baseline.

- **Lock held during slow external calls in watchdog READY_NEXT_SPRINT handler (§6.7: "short critical sections")** — **marginal, acceptable for Phase 1** given the 90s hard alarm, but should be fixed before Phase 9 canary.

- **`ftd_liveness_check.py` is missing many checks from §6.8:** READY_NEXT_SPRINT older than threshold with missing/paused watchdog; checkpoint/error state still dispatching; project watchdog cron missing/paused when project should run; runner log contradiction (traceback/nonzero vs. success state); superseded main-worktree FTD still active — **acceptable for Phase 1/2 floor** as the spec phases these in incrementally.

- **`ftd_validate.py`, `ftd_git.py`, `ftd_resources.py`, `ftd_workers.py` absent** — **acceptable**. Codex report and spec both mark these as Phase 4-7 deferred work. Acceptance criteria #12, #13, #14, #16 are not yet met, but that is expected.

- **Source-backed runtime copy deferred** — **acceptable**. Spec §10 explicitly defers until after canary.

- **Spec tests #9-25 partially absent** — tests for `ftd_validate`, `ftd_git`, `ftd_resources` are absent because those modules are deferred. **Acceptable.** Watchdog tests #5-8 are absent despite the watchdog being implemented. **Not acceptable** — watchdog behavior is in scope for Phase 1/2.

- **`active_pm_runner_pid` written by `spawn_pm_runner` alongside `active_pm_wrapper_pid` (ftd_lib.py:709)** — acceptable as explicit legacy compat field per spec §6.1.8 ("preserve legacy `active_pm_runner_pid = proc.pid` temporarily only for compatibility"). ✓

---

## Commands Run

- Reviewed all files via Read tool — no shell commands executed (review only, as constrained)
