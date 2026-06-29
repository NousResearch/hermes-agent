# FTD Autonomous Developer Plan — Oppositional Architecture Review

**Date:** 2026-05-14
**Reviewer:** Claude Code (adversarial architecture review)
**Input:** `docs/ftd-autonomous-developer-plan-2026-05-14.md`
**Basis:** Plan text + code inspection of `~/.hermes/scripts/ftd_lib.py`, `ftd_start.py`, `ftd_set_state.py`, `ftd_stop.py`, `ftd_status.py`, `ftd_pm_runner_wrapper.py`, `tests/test_ftd_control_plane.py`

---

## 1. Executive Verdict

**Viable as-is: No.** The plan is architecturally coherent, but the gap between the written design and the existing code is large enough to make autonomous operation unsafe. The two most critical files (`ftd_watchdog.py`, `ftd_liveness_check.py`) are absent, causing all control-plane tests to fail at collection time. The default config template assigns child Kanban work to profiles that do not exist. The PM termination path in `ftd_stop.py` targets the wrong PID. None of this is fatal to the plan's design — but it means Phase 0 is larger than described.

### Top 5 blockers / highest-leverage fixes

1. **`ftd_liveness_check.py` is missing and its absence breaks the entire test suite.** `test_ftd_control_plane.py` imports it unconditionally at load time. `pytest` will fail to collect any tests until this file exists, meaning the control-plane tests cannot be run even to verify the parts that do work.

2. **`ftd_stop.py` terminates the wrapper PID, not the hermes process.** `spawn_pm_runner` records the wrapper's PID as `active_pm_runner_pid`. The actual hermes session PID is written later by the wrapper as `active_pm_child_pid`. `ftd_stop.py` sends SIGTERM to `active_pm_runner_pid` (wrapper), which terminates the wrapper but leaves the hermes child process running. A "stopped" FTD project can still have a live, autonomous hermes session consuming tokens and writing to the repo.

3. **Default config template hardcodes nonexistent profiles.** `ftd_lib.default_repo_config` emits `implementer_assignee: "galtcode"`, `reviewer_assignee: "galtcode"`, `researcher_assignee: "galtresearch"`, `ops_assignee: "galtops"`. Only `default` exists. Every child Kanban card the PM creates with these assignees will stall in `todo` state forever, silently failing the entire sprint.

4. **`ensure_project_watchdog` cannot detect a dead cron job.** When a stored `job_id` exists, it calls `hermes cron resume <id>` with `check=False` and returns the stored ID regardless of whether the resume succeeded or the job was deleted. The watchdog is effectively silently dead but the system believes it is running.

5. **Double lock window in `ftd_start.py` creates a spawn race.** The first `project_lock` block (lines 53–72) is released before `create_board_if_needed` and `kanban_create` (root task) run. A second concurrent `ftd_start` can pass through the first lock check, observe the same pre-ACTIVE state, and both processes reach the second `project_lock` (line 100) and attempt to spawn a PM runner in sequence, producing two runners.

---

## 2. Critical Gaps

### 2.1 `ftd_liveness_check.py` doesn't exist; tests cannot run at all

`test_ftd_control_plane.py:load_ftd_modules` does `import ftd_liveness_check` unconditionally. This is not a test-level import — it happens in the shared fixture called by every test. Running `pytest tests/test_ftd_control_plane.py` currently fails at collection with `ModuleNotFoundError`. No control-plane tests pass.

The last test (`test_liveness_flags_live_generic_worker_on_active_pm_sprint_task`) calls `ftd_liveness_check.sprint_task_control_plane_problem(state, task)` — this function must exist with that exact signature.

### 2.2 `ftd_watchdog.py` doesn't exist; `READY_NEXT_SPRINT` is a dead end

The entire "silent rollover" path is broken. When a PM runner sets `READY_NEXT_SPRINT`, nothing advances to the next sprint. `write_project_watchdog_wrapper` emits `import ftd_watchdog` into the generated per-project script. That script will crash on import, and cron will deliver a notification on every 2-minute tick.

### 2.3 PM termination targets wrong PID

`spawn_pm_runner` → `proc = subprocess.Popen(run_cmd, ...)` where `run_cmd` is the *wrapper* script (`ftd_pm_runner_wrapper.py`). The wrapper PID is stored in `state["active_pm_runner_pid"]`. Inside the wrapper, the actual `hermes chat` subprocess is spawned as a child, and its PID is stored in `state["active_pm_child_pid"]`. `ftd_stop.py:main` calls `terminate_process(state.get("active_pm_runner_pid"))`, killing the wrapper, not hermes. The hermes child is orphaned: `start_new_session=True` means it has no controlling terminal and ignores SIGHUP.

### 2.4 State transition enforcement exists only in the PM prompt

Section 7.3's seven-item closeout checklist ("No PM runner may set a terminal state until...") is only in the PM's natural-language prompt. `ftd_set_state.py` accepts any valid state from any caller with no precondition checks. An LLM that skips the handoff update, skips board reconciliation, or crashes mid-closeout can set `READY_NEXT_SPRINT` and the watchdog will happily spawn the next sprint with a stale or missing handoff.

### 2.5 `COMPLETE` state: watchdog job is paused forever, never cleaned up

`ftd_set_state.py` pauses the cron job when state is `COMPLETE`. The cron job is never deleted. It will fire silently every 2 minutes indefinitely, consuming scheduler slots, until manually removed. There is no "delete cron on COMPLETE/OFF" path.

---

## 3. Control-Plane Risks

### 3.1 Stale cron job ID treated as live

`ensure_project_watchdog` retrieves a stored `cron_id` and calls `hermes cron resume <id>` with `check=False`. If the job was deleted externally (e.g., board cleanup, hermes reinstall), the resume call fails silently and the function returns the stale ID. FTD enters `ACTIVE`, the watchdog never fires, and liveness degradation is undetected.

**Mitigation needed:** Verify the job exists after resume (e.g., `hermes cron list` + grep for `job_id`) and recreate if missing.

### 3.2 Double lock window allows concurrent sprint spawn

Between the first and second `project_lock` in `ftd_start.py`, an unguarded code path creates the Kanban board and root task. Two concurrent `ftd_start` invocations can both pass the first lock check (before either sets state to ACTIVE), both create the root task (idempotent due to idempotency key), then both enter the second lock in sequence. The second arrival will find `ACTIVE` with a live PID and bail, but by then the first spawned PM has already read `SPAWNING_NEXT_SPRINT` and created sprint task 1. The second arrival increments sprint counter to 2 (it re-reads state from the first attempt) and creates sprint task 2 before being blocked. Board has orphan sprint 1 and no runner for it.

**Mitigation:** Set state to `STARTING` inside the first lock block and check for it in the second, or consolidate into a single lock block.

### 3.3 `SPAWNING_NEXT_SPRINT` is written before `kanban_create`, not after

`create_next_sprint` increments `sprint_counter` and saves `spawning_sprint_counter` into state, then calls `kanban_create`. If `kanban_create` raises, the state file has an incremented sprint counter and a `SPAWNING_NEXT_SPRINT` flag, but no real task ID. The watchdog will see `SPAWNING_NEXT_SPRINT` and attempt recovery, but `spawning_sprint_task_id` is absent. Recovery logic in `ftd_watchdog.py` (once written) must handle this case explicitly.

### 3.4 `STOPPING` state does not terminate PM runner

`ftd_set_state.py` handles `STOPPING` by pausing the watchdog and removing `active_pm_runner_pid` from state — but does not call `terminate_process`. A PM runner in `ACTIVE` remains running after the state transitions to `STOPPING`. Only `ftd_stop.py` terminates a runner, and as noted above, it targets the wrapper PID, not the hermes child.

### 3.5 Project lock is `fcntl`-based, not multi-machine safe

If the `~/.hermes` directory is on a network filesystem or sync'd by iCloud/Dropbox, `fcntl.LOCK_EX` may not provide correct exclusivity. This is a minor risk in the current single-machine context but worth noting for any future remote-execution path.

### 3.6 `max_sprints_without_benjamin_review: 6` is config-only, never enforced

No code reads the `max_sprints_without_benjamin_review` or `max_active_hours_without_checkpoint` keys. The caps described in Section 10 as "safety fuses" do not exist in implementation. A runaway FTD will sprint indefinitely in the wrong direction.

---

## 4. Worker Orchestration Risks

### 4.1 `--accept-hooks` in PM spawn command

`spawn_pm_runner` passes `--accept-hooks` to `hermes chat`. This auto-approves all configured hermes shell hooks for the PM session. If a hook is configured to run on tool use, file write, or session end — including any hooks configured globally, not just FTD-scoped — the PM session will accept them without user review. This is a silent privilege escalation vector.

### 4.2 Wrapper `preexec_fn` + `start_new_session=True` is deprecated behavior

`ftd_pm_runner_wrapper.py:main` calls `subprocess.Popen(cmd, start_new_session=True, preexec_fn=_ignore_interactive_signals_in_child)`. Python 3.12 emits a `DeprecationWarning` when `preexec_fn` is used; Python 3.14 will make it an error in multi-threaded contexts. The combination with `start_new_session=True` also means the preexec function runs after the session is already detached, making the signal ignores redundant.

### 4.3 No retry on transient Kanban API failures

`kanban_create`, `create_board_if_needed`, and `dispatch_board` all call `ftd_lib.run()` without retry. A transient HTTP error, rate limit, or network blip during sprint creation causes FTD to fail hard and transition to `ERROR`. There is no back-off or retry.

### 4.4 Context provided to next sprint is raw HANDOFF.md tail

`create_next_sprint` inlines the last 6,000 bytes of `HANDOFF.md` directly into the Kanban sprint card body. If the handoff has grown large (accumulated sprint notes, long risk sections), the truncation at 6,000 bytes may cut mid-sentence or omit the "Next sprint recommendation" section that is supposed to appear near the end. No structural extraction is performed.

### 4.5 Worker health checks are manual, not automated pre-sprint

Sections 16.1 and 16.2 describe smoke-testing Claude and Codex before assigning work. Nothing in the current `ftd_watchdog.py` / `ftd_start.py` pipeline enforces this. There is no code path that runs the smoke prompts. A Codex auth expiry or Claude CLI logout will not be detected until a worker reports failure, which is a full sprint turn lost.

### 4.6 `models:` section in config template uses nonexistent model names

`ftd_lib.default_repo_config` emits `pm_runner: "gpt-5.5"` and `hard_review: "gpt-5-pro-or-opus"`. Neither `read_config_flag` nor `read_config_int` reads the `models:` key. No script uses these values. The model names are not valid hermes model identifiers. This section is dead configuration that will mislead anyone reading the config file.

---

## 5. Kanban / Profile / Dispatcher Risks

### 5.1 All three specialist profiles are missing; all child cards will stall

`ftd_lib.default_repo_config` hardcodes:
```yaml
implementer_assignee: "galtcode"
reviewer_assignee: "galtcode"
researcher_assignee: "galtresearch"
ops_assignee: "galtops"
```
`hermes profile list` returns only `default`. Cards assigned to nonexistent profiles are never claimed by any worker. The PM creates child tasks, the board fills up, nothing runs. The sprint eventually exhausts `max_pm_turns` or times out. The plan's mitigation (§17.9: "use available profiles only") is not implemented anywhere in the code.

### 5.2 `dispatch_board` dispatches all tasks without tenant/assignee filter

`ftd_lib.dispatch_board` runs `hermes kanban --board <board> dispatch --max <n>`. This dispatches the next available task on the board regardless of tenant. If the PM has created both FTD child tasks and administrative tasks (e.g., a `blocking-research` task that should wait), dispatch will pick them up in queue order. More critically, if a board is accidentally shared or misconfigured, dispatch will claim non-FTD tasks.

### 5.3 Board-tenant isolation is not enforced at dispatch time

Each FTD project gets a board slug like `ftd-<repo>-<hash>`. The slug can collide if two repos have the same name and the SHA1 first 8 hex characters collide (1 in 4 billion, unlikely but non-zero). More practically, the `dispatch_board` call doesn't filter by `--tenant ftd:<project-id>`, so if hermes `dispatch` doesn't scope to tenant by default, child tasks from two FTD projects sharing a board (theoretical) could cross-contaminate.

### 5.4 Stale `running` cards with dead worker PIDs block queue

There is no watchdog behavior described for detecting child cards stuck in `running` state with a dead worker PID. The liveness sentinel (§15) lists this as a check, but the implementation path is unclear. A single crashed Codex worker leaves a card permanently `running`, blocking the sprint from closing without manual board intervention.

---

## 6. Environment Cleanup Risks

### 6.1 Entire cleanup system is Phase 3 — not yet designed

Phase 3 is listed last among implementation phases. The resource ledger files, tracking helpers, and inventory card described in Section 13 do not exist. Until Phase 3 is complete, every sprint runs without any resource registration. When Phase 3 is finally implemented, there will be no ledger entries for resources created by Phases 0–2 sprints. The first cleanup pass will see only ambiguous/unowned resources and report nothing cleanable — silently leaving stale processes and containers behind.

### 6.2 `cleanup_command` in ledger is an arbitrary string executed by the cleanup agent

The ledger schema (§13.2) includes `"cleanup_command": "..."`. If any sprint creates a resource with a PM-supplied cleanup command, the cleanup subagent executes an arbitrary shell string. This is a command injection vector if the PM runner is compromised or confused. No sandboxing or allowlist is proposed for cleanup commands.

### 6.3 Worktrees: `owned-worktrees.json` ledger only helps if worktrees are registered before use

If a Codex worker creates a git worktree (allowed by the plan) without the PM first registering it in the ledger, the worktree is invisible to cleanup. The registration step is described as PM responsibility, but no enforcement exists. An `ACTIVE` → `ERROR` transition mid-sprint before registration runs leaves an orphan worktree permanently.

### 6.4 No TTL enforcement on ledger entries

The ledger schema has a `"ttl": "..."` field. Nothing reads or enforces it. Stale ledger entries from previous sprints accumulate. If a resource with `safe_cleanup: true` was already externally destroyed (e.g., Docker container stopped manually), the cleanup runner will attempt to re-run the cleanup command on a nonexistent resource, potentially producing errors or claiming false success.

---

## 7. Git / Release / Process Risks

### 7.1 No CI; local verification is the only gate

Section 11 and §17.8 acknowledge this. The risk is that "local verification" is PM-prompt-driven: the PM is instructed to run tests, but there is no enforcement. A PM that reports "tests passed" without running them will be believed if the handoff validator doesn't check the `Verification history` field for actual command output, not just presence of text.

### 7.2 `ftd_set_state.py` has no git-state precondition

A PM can call `ftd_set_state.py --state READY_NEXT_SPRINT` with uncommitted changes in the working tree. Nothing checks `git status` before accepting the state transition. Verified-sprint commits are the PM's responsibility, but an LLM mid-context-exhaustion can skip the commit step without triggering any machine-enforced gate.

### 7.3 Branch naming is not standardized or validated

The plan asks the question (§19.3) but does not answer it. `ftd_start.py` does not create or switch branches. The PM runner is told to commit/push "on the current branch." If the repo is on `main` at FTD start, all autonomous work accumulates directly on `main`, violating §11 ("do not merge to main by default").

### 7.4 Push path: `push_branch_only: true` is a config key that nothing reads

`default_repo_config` emits `push_branch_only: true`. `ftd_lib.read_config_flag` can read boolean keys by name but no code reads `push_branch_only`. The PM prompt says "push only if allow_push and push_if_allow_push" but says nothing about branch restrictions. The PM could push to main.

### 7.5 `never_force_push: true` is also unread config

Same issue as above. The config key exists for documentation purposes only. Nothing in the scripts enforces it.

---

## 8. Missing Features or Subsystems

### 8.1 No per-sprint sprint-count-to-checkpoint tracker

The `max_sprints_without_benjamin_review: 6` safety fuse requires a counter that increments on `READY_NEXT_SPRINT` and resets on `FEATURE_CHECKPOINT_READY_FOR_BENJAMIN`. No such counter exists in the state schema.

### 8.2 No handoff validator

Section 9 specifies "Add a handoff validator that refuses terminal state if required fields are missing or stale." `ftd_set_state.py` does not call any validator. The PM can set any terminal state with an empty or stub handoff.

### 8.3 No structured sprint outcome in `SPRINTS.md`

`ftd_lib` creates `SPRINTS.md` but provides no helper to append structured sprint records. The PM is instructed to append to it, but the format is ad hoc. A helper (`append_sprint_record`) with a defined schema would make sprint history machine-readable for future liveness/audit checks.

### 8.4 No global FTD liveness sentinel for multiple projects

§19.5 asks about this. Currently the design is per-project only. If 3 projects are running and one silently enters a stale `ACTIVE` state (watchdog paused), only that project's watchdog would detect it — and it's already broken if the watchdog is what detects it. A global sentinel script that iterates `iter_states()` and reports cross-project anomalies would catch watchdog-silenced degradation.

### 8.5 No mechanism to discover and record which profiles actually exist

The plan's mitigation for missing profiles (§17.9) says "discover profiles at sprint start." No `hermes profile list` call is made in `ftd_start.py` or the PM prompt. The PM is expected to do this at runtime, but the prompt doesn't include it in preflight steps (§8.2).

### 8.6 No rate-limit or token-cost tracking

A FTD project running 6 sprints with 120 turns each, plus Codex missions, can accumulate substantial API cost. There is no cost cap, turn-budget alert, or daily spend limit. The `max_active_hours_without_checkpoint: 12` config exists but is not enforced.

---

## 9. Edge Cases Not Covered

### 9.1 PM runner times out at `max_pm_turns` mid-closeout

If the PM reaches `max_pm_turns` before it writes the handoff and calls `ftd_set_state.py`, hermes exits 0 (turn limit, not error). The wrapper sees `returncode == 0` and does not transition to `ERROR`. State remains `ACTIVE`. The watchdog sees a live PID... but the process is gone. This is indistinguishable from the PM still running except that `active_pm_runner_pid` is dead. The liveness check would catch the dead PID, but only if `ftd_liveness_check.py` exists.

**Mitigation:** The wrapper should transition to `ERROR` if the PM exits 0 while state is still `ACTIVE` (not just nonzero). Zero exit on mid-closeout is a closeout failure.

### 9.2 Cron watchdog fires while PM is mid-closeout (in `project_lock`)

The watchdog will attempt `project_lock` to read state. The PM also holds `project_lock` during state transitions (via `ftd_set_state.py`). The watchdog blocks, which is correct. But if the PM crashes while holding the lock (e.g., `ftd_set_state.py` Popen'd and killed mid-write), `fcntl.LOCK_EX` is released automatically by the OS on file handle close. The state file may be in a partially written state (the `.tmp` rename in `save_json` protects the file, but the lock would be released before rename completes if the process is killed at exactly the wrong moment).

### 9.3 Two simultaneous `ftd_set_state.py` calls from PM + watchdog

A PM running with `--yolo` could call `ftd_set_state.py` at exactly the same time as the watchdog attempts a state transition. Both will acquire `project_lock` sequentially. The second write will overwrite the first. If the PM sets `READY_NEXT_SPRINT` and the watchdog simultaneously sets `SPAWNING_NEXT_SPRINT` (having seen the old READY state), the PM's write arrives second and reverts back to READY. Watchdog fires again and spawns again.

### 9.4 Repo path changes (rename, symlink resolution)

`project_id_for` uses `repo.expanduser().resolve()` + SHA1 of the resolved path. If the repo is moved or its parent directory renamed, the project ID changes. The old state file is orphaned. The watchdog's generated script hardcodes the project ID. The cron job keeps calling a watchdog that opens a project ID that returns empty state. No error is raised; the watchdog silently no-ops forever.

### 9.5 `ftd_start --continue-paused` after ERROR with live child process

If FTD is in `ERROR` state because the PM exited nonzero, but the actual hermes child process is still running (e.g., wrapper died but hermes survived), `ftd_start --continue-paused` will spawn a new PM runner without killing the existing one. Two runners will both write to the same repo and board. The `process_alive` check uses `active_pm_runner_pid` (wrapper PID, now dead), not `active_pm_child_pid` (hermes, still running).

### 9.6 Kanban `--idempotency-key` behavior on replay

If sprint 3 is created but the process crashes before `save_state` is called, re-running the watchdog will increment to sprint 3 again and call `kanban_create` with the same idempotency key `ftd:<id>:sprint:3`. If the Kanban API returns the existing task ID, the sprint picks up correctly. If the Kanban API treats a duplicate idempotency key as an error, the sprint never advances. The plan assumes idempotent behavior but does not test it.

### 9.7 Hermes session output not captured by the wrapper

`subprocess.Popen(run_cmd, ..., stdout=log_fh, stderr=subprocess.STDOUT)` in `spawn_pm_runner` captures *wrapper* stdout. The actual hermes session output is written to the log by hermes itself (passed via `-Q`). If hermes is configured to write to a different log path or suppresses output in `-Q` mode, the log file may be empty. There is no verification that the log file is being written to before concluding a sprint completed.

---

## 10. Recommended Changes to the Plan

Ordered by priority:

1. **Implement `ftd_liveness_check.py` first, before any other Phase 0 work.** Without it, no tests run. The test suite provides the only safety net for Phase 0 changes.

2. **Fix `ftd_stop.py` to terminate `active_pm_child_pid`, not just `active_pm_runner_pid`.** Add a `terminate_process(state.get("active_pm_child_pid"))` call before the wrapper SIGTERM. Add a test verifying the child PID is terminated.

3. **Replace the default config template's nonexistent profile names with `default` or make them conditional.** Do a `hermes profile list` at `ftd_start` time and store available profiles in state. Gate child card assignment on profile availability check. Fail loudly at start if `use_kanban: true` and no usable profiles exist.

4. **Consolidate `ftd_start.py`'s two `project_lock` blocks into one.** Move `create_board_if_needed` and the root `kanban_create` inside the single lock. State must be written to `STARTING` before any lock release.

5. **Add sprint-past-checkpoint counter to project state.** Increment in `create_next_sprint`. Reset on `FEATURE_CHECKPOINT_READY_FOR_BENJAMIN`. Check in `create_next_sprint` and transition to checkpoint state if the cap is reached. This is the only way the §10 safety fuse is enforced.

6. **Add a handoff validator callable from `ftd_set_state.py`.** Block `READY_NEXT_SPRINT`, `FEATURE_CHECKPOINT_READY_FOR_BENJAMIN`, and `COMPLETE` transitions if HANDOFF.md is missing, its `Last updated` timestamp is older than the current sprint's `started_at`, or required sections are absent.

7. **Make `ensure_project_watchdog` verify the cron job is live after resume.** After `hermes cron resume <id>`, call `hermes cron list` or an equivalent status check. If the job is not found, delete the stale config entry and create a new job. Return the verified live job ID.

8. **Add `COMPLETE` / `OFF` cron job deletion path.** `pause_watchdog_if_known` should offer a `delete=True` variant. `COMPLETE` and `OFF` states should delete, not pause, the cron job.

9. **Replace `models:` section placeholder names with real model identifiers or remove the section.** `gpt-5.5` and `gpt-5-pro-or-opus` are not hermes model IDs. Either use real IDs (`claude-opus-4-7`, `claude-sonnet-4-6`, etc.) or remove the section until it is actually read by code.

10. **Add branch enforcement to the PM prompt preflight.** Before any work begins, require the PM to confirm it is on a non-main branch and create one if needed. Add this as a preflight step in §8.2 alongside the existing git status check.

---

## 11. Tests That Must Be Added

These are concrete test behaviors, ordered roughly by the control-plane failure mode they prevent:

1. **`test_ftd_stop_terminates_hermes_child_not_just_wrapper`** — Set state with both `active_pm_runner_pid` (wrapper, dead) and `active_pm_child_pid` (child, alive). Call `ftd_stop.main()`. Assert `terminate_process` was called with the child PID, and that `active_pm_child_pid` is cleared from state.

2. **`test_ftd_set_state_ready_next_sprint_while_state_active_advances_sprint_counter`** — When `ftd_set_state` is called with `READY_NEXT_SPRINT`, a sprint-since-checkpoint counter increments. When the counter reaches the configured cap, the next watchdog tick transitions to `FEATURE_CHECKPOINT_READY_FOR_BENJAMIN`.

3. **`test_ftd_watchdog_detects_dead_pm_runner_and_sets_error`** — Project in `ACTIVE` with a dead PID. Watchdog tick sets state to `ERROR`.

4. **`test_ftd_watchdog_advances_ready_next_sprint_to_spawning`** — Project in `READY_NEXT_SPRINT`. Watchdog tick creates next sprint card and sets `SPAWNING_NEXT_SPRINT`.

5. **`test_ftd_watchdog_recovers_stale_spawning_next_sprint`** — Project stuck in `SPAWNING_NEXT_SPRINT` with no live wrapper PID for longer than TTL. Watchdog transitions to `ERROR` and notifies.

6. **`test_ftd_watchdog_pauses_on_checkpoint_state`** — Project in `FEATURE_CHECKPOINT_READY_FOR_BENJAMIN`. Watchdog does not spawn, does not advance state, calls `pause_watchdog_if_known`.

7. **`test_ftd_watchdog_silent_on_off_state`** — Project `OFF`. Watchdog emits no stdout and takes no action.

8. **`test_ftd_start_consolidates_lock_no_double_spawn`** — Simulate two concurrent `ftd_start` calls using threads. Assert only one PM runner is spawned and only one sprint task is created.

9. **`test_handoff_validator_blocks_terminal_state_without_updated_handoff`** — Call `ftd_set_state` with `READY_NEXT_SPRINT` when HANDOFF.md is missing or has a stale timestamp. Assert `SystemExit` with a message naming the missing field.

10. **`test_ensure_project_watchdog_recreates_on_dead_job_id`** — Store a stale `cron_id` in global config. Mock `hermes cron resume` to return nonzero. Assert `ensure_project_watchdog` creates a new job and updates the stored ID.

11. **`test_dispatch_board_only_in_active_state`** — Watchdog should call `dispatch_board` only when state is `ACTIVE`. Assert dispatch is not called for `READY_NEXT_SPRINT`, `ERROR`, or `STOPPING`.

12. **`test_pm_runner_exit_zero_while_active_transitions_to_error`** — Wrapper detects PM exited 0 while state is still `ACTIVE`. Assert state transitions to `ERROR` (currently only nonzero exits trigger this in `ftd_pm_runner_wrapper.py`).

13. **`test_create_next_sprint_idempotent_on_kanban_failure`** — `kanban_create` raises `RuntimeError`. Assert sprint counter in the state file is not advanced (i.e., `save_state` was not called before the failure).

14. **`test_ftd_liveness_flags_active_with_dead_pid`** — `ftd_liveness_check` called with ACTIVE state and a PID that `process_alive` returns False for. Assert problem string is returned naming the dead PID.

15. **`test_ftd_stop_transitions_to_off_not_stopping`** — Current `ftd_stop.py` sets state to `OFF` directly, bypassing `STOPPING`. If `STOPPING` is intended as a transition state with runner termination semantics, a test should enforce that the full stop sequence runs through `STOPPING` → `OFF`.

---

## 12. Final Recommendation

**Build the control-plane floor before any sprint pipeline work.**

The minimum viable system — one that can safely start, silently roll over sprints, and stop without orphaning processes — requires exactly four files to be correct: `ftd_watchdog.py`, `ftd_liveness_check.py`, and fixes to `ftd_stop.py` and `ftd_start.py`. These four changes unlock the test suite and make the core loop (ACTIVE → READY_NEXT_SPRINT → ACTIVE) trustworthy.

Do not implement the Claude/Codex pipeline (Phase 2) until the control plane passes its own tests with the missing files in place. A broken watchdog + broken stop + broken liveness check means any work in Phase 2 will be autonomously building on sand. The sprint will look like it's running while the watchdog silently fails, and you will not notice until you check manually.

The plan's own §20 says this. The review confirms it. The gap between §20 and reality is that Phase 0 is not just "write two missing files" — it also requires fixing the stop/PID confusion, the double-lock race, and the nonexistent-profile config, all of which would cause obvious failures on the first real run.
