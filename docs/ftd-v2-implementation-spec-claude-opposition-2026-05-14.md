# FTD V2 Spec — Oppositional Review

**Date:** 2026-05-14
**Reviewer:** Claude Code (oppositional systems architect)
**Input spec:** `docs/ftd-v2-implementation-spec-draft-2026-05-14.md`
**Runtime files inspected:**
- `/Users/johngalt/.hermes/scripts/ftd_lib.py`
- `/Users/johngalt/.hermes/scripts/ftd_start.py`
- `/Users/johngalt/.hermes/scripts/ftd_stop.py`
- `/Users/johngalt/.hermes/scripts/ftd_status.py`
- `/Users/johngalt/.hermes/scripts/ftd_set_state.py`
- `/Users/johngalt/.hermes/scripts/ftd_pm_runner_wrapper.py`
- `/Users/johngalt/.hermes/scripts/tests/test_ftd_control_plane.py`

---

## 1. Executive Verdict

The spec is well-structured, self-aware about old architecture debt, and targets the right bugs. Its posture is correct: replace, don't patch.

However, it has five categories of concrete problems that will cause Codex to produce broken or incomplete work without further clarification:

1. **Naming confusion between wrapper PID, runner PID, and child PID** — the spec introduces a three-field schema that contradicts how the current code actually assigns those fields.
2. **Three bugs the spec claims to fix are actually present in more dangerous forms than described** — specifically the exit-zero-while-ACTIVE case, the `set_state` PID-clearing side effect, and the `create_next_sprint` counter-before-Kanban ordering.
3. **The Phase A source-backed copy strategy is a development trap** — copying without changing creates two-version drift during the actual implementation phases.
4. **Several "new module" boundaries are underspecified** — Codex cannot correctly implement `ftd_watchdog.py` or `ftd_liveness_check.py` from the spec alone without resolving open questions about the Hermes cron/Kanban CLI output format.
5. **The Claude interactive tmux health check is untestable and wrong for a control-plane floor** — it belongs in Phase D at earliest and must not block Phase B tests.

Overall recommendation: **implement with spec revisions in sections 13 and 14 of this review applied first**. The core design is sound; the blockers are tactical.

---

## 2. Top Blocking Issues

### 2.1 The three-PID naming scheme contradicts the current code

The spec (§3.1) describes three distinct state fields:

```json
"active_pm_runner_pid": 123,
"active_pm_runner_wrapper_pid": 123,
"active_pm_child_pid": 456
```

But in the current code:

- `spawn_pm_runner` (`ftd_lib.py:590`) spawns the wrapper and stores `proc.pid` into `state["active_pm_runner_pid"]`. At spawn time, `active_pm_runner_pid` IS the wrapper PID.
- The wrapper itself (`ftd_pm_runner_wrapper.py:78`) later writes `state["active_pm_runner_wrapper_pid"] = os.getpid()`, which sets the same value again.
- The child (real Hermes) PID is written by the wrapper at `ftd_pm_runner_wrapper.py:77` into `active_pm_child_pid`.

So `active_pm_runner_pid == active_pm_runner_wrapper_pid` once the wrapper starts. The naming implies they can differ, which they cannot in the current design. If the spec intends to keep three fields, it must either:

- **Rename:** `active_pm_runner_pid` → `active_pm_wrapper_pid` at spawn time; remove the redundant second write in the wrapper; keep `active_pm_child_pid` for the real Hermes process.
- **Or collapse:** keep only `active_pm_wrapper_pid` and `active_pm_child_pid`.

Without resolving this, the liveness checks in `ftd_liveness_check.py` and `ftd_stop.py` will be implemented against a schema that doesn't match reality, and test assertions will be wrong.

### 2.2 The exit-zero-while-ACTIVE bug is worse than described

The spec (§6.2, item 3) says: "If child exits while state is still ACTIVE... transition to ERROR **regardless of exit code**." This is correct.

But the current wrapper at `ftd_pm_runner_wrapper.py:116–128` only converts nonzero exits to ERROR:

```python
if (
    returncode != 0
    and state.get("state") == "ACTIVE"
    and state.get("active_sprint_task_id") == args.sprint_task_id
):
    state["state"] = "ERROR"
```

The exit-zero-while-ACTIVE bug is confirmed present. The spec correctly identifies it. **However**, there is a compounding interaction the spec does not address:

`ftd_set_state.py:43` unconditionally pops `active_pm_runner_pid` for any non-ACTIVE state:

```python
if args.state != "ACTIVE":
    state.pop("active_pm_runner_pid", None)
```

This means when the PM calls `ftd_set_state.py --state READY_NEXT_SPRINT`, the PID is cleared from state — but the wrapper's child process is still alive at that moment. The wrapper's `finally` block then runs its checks against the updated state (sprint task id may still match), but `active_pm_runner_pid` is gone. If the wrapper was checking for PID match to guard against stale exits (§6.2, item 5), this guard is already broken by the PM itself calling set_state.

**Fix required in spec:** The wrapper must not rely on `active_pm_runner_pid` being present in state to determine "stale vs. current." It must use `active_pm_child_pid` or a `current_wrapper_run_id` token written at spawn time.

### 2.3 `create_next_sprint` advances the counter before the Kanban call

`ftd_lib.py:634–635`:

```python
sprint = int(state.get("sprint_counter") or 0) + 1
state["sprint_counter"] = sprint
state["spawning_sprint_counter"] = sprint
...
save_state(state)  # line 669 — saves incremented counter
return task_id     # line 671 — task_id from Kanban call
```

Actually, the save at line 669 happens inside `create_next_sprint` after the Kanban call on line 656. But the counter is mutated in-memory at line 635 and written to state at line 669 after the Kanban call. If `kanban_create` raises (network/auth error), the in-memory state has an incremented counter, and if the caller saved state before this point, the counter is advanced without a corresponding task. The watchdog's `READY_NEXT_SPRINT` handler calls `create_next_sprint` and then must handle this atomicity failure. The spec's acceptance criterion 14 ("Create/reuse sprint task idempotent no counter advance on Kanban failure") is correct, but the fix is not described. The idempotency key alone does not fix the counter.

**Concrete fix required in spec:** Counter increment must be written to state only after the Kanban call succeeds. The in-memory mutation is fine; the state save must be ordered: Kanban first, then `state["sprint_counter"] = sprint`, then save.

### 2.4 `ensure_project_watchdog` silently ignores dead cron IDs

`ftd_lib.py:205–207`:

```python
if job_id:
    run([HERMES_BIN, "cron", "resume", job_id], check=False)
    return job_id
```

`check=False` means if the job was deleted, the resume fails silently and the stale `job_id` is returned as if it were live. The spec (§4.4) says to verify the job exists after resume. But the spec does not specify what `hermes cron list --all` output looks like, or how to parse the job ID from it. Codex cannot implement `cron_job_exists(job_id)` without knowing the CLI output schema.

**Required addition to spec:** Either document the exact `hermes cron list` output format, or specify that `cron_job_exists` should call `hermes cron status <id>` (if that subcommand exists) and check exit code only.

### 2.5 Existing watchdog wrapper wrappers crash on import today

`ftd_lib.py:188`:

```python
"import ftd_watchdog\n"
```

`ftd_watchdog.py` does not exist. Any project that has an existing generated watchdog wrapper (`ftd_watchdog_<project-id>.py`) will crash at import with `ModuleNotFoundError` on every cron tick, generating noise before Phase B is complete.

The spec's implementation order does not address this. If Phase A copies the existing runtime files without creating `ftd_watchdog.py`, every active project's watchdog immediately starts failing.

**Required:** Either Phase A must create a minimal stub `ftd_watchdog.py` that returns 0 and is silent, or existing watchdog wrapper generation must be gated on the module existing.

---

## 3. Where the Draft Still Relies Too Much on Old Architecture

### 3.1 The PM prompt in `ftd_lib.py` is still doing enforcement work

`ftd_lib.py:489–544` (`build_pm_prompt`) contains extensive policy text: "Never install packages unless...", "Push only if...", "Never deploy, expose public network...". The spec (§6.3) correctly identifies this and says to add more fields to the prompt. But the spec still frames these as additions to the PM prompt, not as replacements by machine validators.

The current approach is: PM reads the prompt, PM decides. V2 claims "State transitions are machine-validated" and "Branch/push policy is enforced by code, not prompt text." But then §6.3 says "Move long PM policy text out of `ftd_lib.py` into a template/helper function if practical" and adds more policy text. This is contradictory. Enforcement belongs in `ftd_validate.py` and `ftd_git.py`. The PM prompt should reference the validator commands, not re-state the policy.

### 3.2 `dispatch_board()` has no ACTIVE guard

`ftd_lib.py:700–704`:

```python
def dispatch_board(state: Dict[str, Any]) -> None:
    ...
    run([HERMES_BIN, "kanban", "--board", state["board"], "dispatch", "--max", max_dispatch], check=False)
```

No check that `state["state"] == "ACTIVE"`. No check that PM runner is alive. The spec (§8.3) correctly mandates this restriction, but the current implementation dispatches unconditionally when called. The spec does not explicitly say "fix this function"; it says to add guards in the watchdog. The risk is that any caller other than the watchdog bypasses the guard. The guard must be inside `dispatch_board` itself, not only in the watchdog.

### 3.3 `ftd_stop.py` only terminates the wrapper, not the child

Confirmed by code. `ftd_stop.py:22`: `pid = state.get("active_pm_runner_pid")` — this is the wrapper PID. `terminate_process(pid)` sends SIGTERM to the wrapper's process group. But the wrapper spawned the child with `start_new_session=True` (`ftd_pm_runner_wrapper.py:72`), placing the child in a separate session and process group. Killing the wrapper's process group does not reach the child.

The spec correctly identifies this. But §5.2 says to terminate `active_pm_child_pid` first, then wrapper/runner. The spec needs to explicitly note: **the child may outlive the wrapper kill by minutes**. Stop must poll for child death, not just send SIGTERM.

### 3.4 Old config still hardcodes nonexistent profiles

`ftd_lib.py:305–310` in `default_repo_config()`:

```yaml
kanban:
  implementer_assignee: "galtcode"
  reviewer_assignee: "galtcode"
  researcher_assignee: "galtresearch"
  ops_assignee: "galtops"
```

The spec correctly flags this. But the spec's fix (§8.2) only defines the V2 config schema and says "PM should prefer direct CLI invocation." It doesn't say to change the default generated config in `default_repo_config()`. This is a concrete gap: if `default_repo_config()` is not changed, every new project starts with nonexistent profiles in config.

---

## 4. Where the Draft Should Reuse Existing Code

### 4.1 `project_lock` is correct — reuse it everywhere

`ftd_lib.py:84–94` (`project_lock`) uses `fcntl.flock` correctly with a context manager. All new modules must import and use this. The spec correctly relies on it.

**One caveat the spec misses:** `fcntl.flock` does not timeout. If a process holds the lock and crashes without releasing it, the lock file remains and the next caller blocks forever. This is not a problem on crash (file locks are released on process exit by the OS), but it is a problem if the process hangs while holding the lock. The spec should note this limitation, especially for the watchdog which runs on a cron tick and may be killed by the scheduler if it hangs.

### 4.2 `save_json` atomic write is correct — do not replace it

`ftd_lib.py:143–147` writes to `.tmp` then renames. This is correct. New modules that write JSON files must use `save_json`, not raw `write_text`.

### 4.3 `process_alive` is correct but incomplete

`ftd_lib.py:601–614` correctly handles `PermissionError` (process alive, we can't signal it) vs `ProcessLookupError` (dead). Reuse this. The spec proposes no change, which is correct.

### 4.4 `board_slug`, `project_id_for`, `slugify` are stable — leave them alone

These functions are used by existing state files. Changing their logic would invalidate all existing project IDs and board names. The spec correctly treats them as stable. Do not touch.

---

## 5. Where the Draft Should Replace Old Code

### 5.1 Replace `ftd_stop.py` entirely — don't patch it

The current `ftd_stop.py` is 38 lines. It has three distinct bugs:

1. Wrong PID targeted (wrapper vs child)
2. No `STOPPING` state transition before termination
3. No verification that the child actually died
4. No lock around the termination/state-write sequence
5. Missing `--timeout-seconds`, `--delete-watchdog`, `--json` flags the spec requires

Patching this file to add all required behavior will result in harder-to-read code than a clean rewrite. The CLI entrypoint contract is simple: take repo path, kill things, set OFF. Rewrite.

### 5.2 Replace `ftd_set_state.py`'s interior, not just add a gate

The current `ftd_set_state.py` is 56 lines with no validation. The spec says to "add validation gates." But the PID-clearing behavior at line 43 is also wrong:

```python
if args.state != "ACTIVE":
    state.pop("active_pm_runner_pid", None)
```

This should not clear PID on non-ACTIVE states unless the PM runner is confirmed dead. If the PM sets `READY_NEXT_SPRINT` and the wrapper is still running (which it is — the PM finishes its last command and then calls set_state before exiting), the PID is cleared while the process is alive. The watchdog then sees a state with no PID and no way to verify liveness until the wrapper exits and writes `last_pm_runner_exit`.

**Correct behavior:** Clear PID fields only in `ftd_stop.py` (explicit stop) and in the wrapper's `finally` block (on confirmed exit). `ftd_set_state.py` should not touch PID fields.

### 5.3 `build_pm_prompt` should shrink, not grow

The spec says to move policy text out. Don't add new fields like "worker health check requirement" and "branch protection requirement" as more prompt text. Replace those with: "Before implementation, run `python3 ~/.hermes/scripts/ftd_workers.py --check` and confirm both workers are healthy. Before closeout, run `python3 ~/.hermes/scripts/ftd_validate.py <repo>` and fix any blocking issues." The PM should call the validators, not memorize their rules.

---

## 6. Missing Files/Features/Subsystems

### 6.1 `ftd_watchdog.py` — no specified output for the global `--all` mode

The spec says `--all` mode should "check all projects for compatibility" and only the project-specific mode requires `--project-id`. But it doesn't specify what "compatibility" means in global mode, what the output format is, or how it differs from `ftd_liveness_check.py`. This overlap will cause Codex to implement redundant or contradictory behavior.

**Recommendation:** `ftd_watchdog.py` should handle only project-specific mode (always requires `--project-id`). Drop the `--all` flag from `ftd_watchdog.py`. `ftd_liveness_check.py` handles global cross-project checks. The current spec blurs this boundary.

### 6.2 No schema migration

The spec defines `schema_version: 2`. Existing state files have no `schema_version` field (V1). The spec does not describe how `load_state` handles a V1 file. If the watchdog loads a V1 file and looks for `schema_version`, it gets `None`. If it looks for `watchdog.cron_id`, it gets `None` (the cron ID is stored in `global_config["project_watchdogs"]` in V1, not in the project state).

**Required:** A migration function: `migrate_state_to_v2(state) -> state`. This is not optional — every active project will hit this on the first watchdog tick after upgrade.

### 6.3 No lock timeout / watchdog self-watchdog

If the watchdog hangs while holding `project_lock`, the next cron tick will queue behind the lock and also hang. After enough ticks, the cron system may accumulate zombie processes. The spec has no mechanism to detect or break this. A simple mitigation: the watchdog script should set `SIGALRM` for a hard timeout (e.g., 90 seconds) before acquiring the lock.

### 6.4 `ftd_liveness_check.py` — task dict structure undefined

The test at `test_ftd_control_plane.py:124` passes a `task` dict with keys `id`, `status`, `assignee`, `worker_pid`, `title`. The spec (§4.5) says the function must exist and return a message string. But:

- Where does the `task` dict come from in production? The function is called with in-memory data in tests, but the actual watchdog would need to query Kanban for the sprint task's current state.
- Does `hermes kanban list --board ... --json` return `worker_pid`? If the Kanban CLI doesn't expose worker PID, the detection is impossible regardless of what the function looks like.

**Required:** Specify what Kanban CLI command produces the task dict and what fields are guaranteed available.

### 6.5 No `ftd_resume.py` or `ftd_recover.py`

The spec adds `--continue-paused` to `ftd_start.py`. But `ftd_start.py --continue-paused` runs the full start transaction including creating a new sprint. For ERROR recovery, the operator often wants to: inspect state, fix the problem, then resume from the existing sprint without creating a new one. A dedicated `ftd_resume.py` or a `--from-existing-sprint` flag on `ftd_start.py` is missing.

### 6.6 No handling of concurrent worktrees on the same repo

If the user has multiple git worktrees from the same repo, `git_root` resolves to different paths per worktree, producing different project IDs. This means two active FTD projects for the same logical repo, with conflicting Kanban boards, conflicting sprint counters, and competing watchdogs. The spec mentions this risk in §18.10 but provides no mitigation. The `repo_inode_fingerprint` field in the state schema is marked "optional platform-specific" with no implementation spec.

### 6.7 No `ftd_board.py` decision is deferred incorrectly

The spec calls `ftd_board.py` "optional but recommended." But §8.3 requires dispatch to be guarded by state/liveness. `dispatch_board()` in `ftd_lib.py` has no such guard. Either add the guard to `ftd_lib.py:dispatch_board` explicitly, or require `ftd_board.py` in Phase B. Marking it optional means Codex may skip it, leaving the unguarded `dispatch_board` in place.

---

## 7. File-by-File Critique

### `ftd_lib.py`

**Keep:** `project_lock`, `save_json`/`load_json`, `slugify`, `project_id_for`, `board_slug`, `project_state_path`, `process_alive`, `git_root`, `iter_states`, `now_iso`, `ensure_dirs`, `VALID_STATES`, `PAUSE_STATES`, `BENJAMIN_REVIEW_STATES`, `FTD_CONTROL_ASSIGNEE`, `ftd_control_assignee`.

**Fix without moving:**
- `default_repo_config()`: change `implementer_assignee`, `reviewer_assignee`, `researcher_assignee`, `ops_assignee` to `null`; add `child_dispatch_mode: "direct_cli"`.
- `dispatch_board()`: add guard `if state.get("state") != "ACTIVE": return`. Add guard that PM runner PID is alive.
- `ensure_project_watchdog()`: after `cron resume check=False`, call a `cron_job_exists(job_id)` helper; if dead, clear `job_id` and fall through to create.
- `create_next_sprint()`: move `state["sprint_counter"] = sprint` and `state["spawning_sprint_counter"] = sprint` assignments to AFTER the `kanban_create` call succeeds.
- `spawn_pm_runner()`: rename the stored field from `active_pm_runner_pid` to `active_pm_wrapper_pid` (or align with spec's chosen naming — just pick one and commit).

**Move out (to new modules):** `build_pm_prompt`, `CORE_PM_SKILLS`, `CORE_TOOLSETS`, `write_project_watchdog_wrapper`, the `--Hermes cron` helpers. These don't belong in a "stdlib-safe primitives" module. The docstring at line 3 says "stdlib-only" but the cron and Kanban calls use `subprocess` against the Hermes CLI, which is not a stdlib primitive.

**Don't move yet:** `kanban_create`, `ensure_repo_files`, `default_handoff`, `read_config_flag` — these are called by start/stop and moving them mid-development creates churn without benefit.

### `ftd_start.py`

**Critical bug not described in spec:** There is a double-ACTIVE check (first at lines 56–67, second at lines 103–113) with a gap in between. Between the first lock release and second lock acquisition, `create_board_if_needed` and root task `kanban_create` run outside any lock. A second concurrent `ftd_start.py` call can:

1. Pass the first lock check (state is STARTING from first caller)
2. Wait on the second lock
3. After first caller releases second lock, second caller sees STARTING (not ACTIVE), passes the pause-state check, and proceeds to create a duplicate sprint task.

The `STARTING` state with a `start_token` approach described in §5.1 is the right fix. But it requires that the first lock write `STARTING` before releasing for external calls, and that the second caller treats `STARTING` as "already in progress" and refuses. The spec needs to spell out the exact token check.

**Another bug not in spec:** `resume_watchdog_if_known(project_id)` at line 160 is called **outside** the project lock, after `spawn_pm_runner` returns. If `spawn_pm_runner` set state to ACTIVE and saved it, but then the watchdog ticks before `resume_watchdog_if_known` is called, the watchdog runs against an ACTIVE state with a new PM runner but a paused watchdog — not catastrophic, but messy.

### `ftd_stop.py`

Confirmed: only kills `active_pm_runner_pid` (wrapper PID), does not kill `active_pm_child_pid` (real Hermes process). Child continues running after `ftd_stop.py` completes. State is set to `OFF` with a live child process. The watchdog (if not paused) will then tick, see OFF, and be silent — but the Hermes child is burning tokens in the background.

The spec's fix is correct. Additionally:

- No lock around the load→terminate→save sequence. Concurrent stop+watchdog can corrupt state.
- `terminate_process` uses `os.killpg(pid_int, signal.SIGTERM)` — this is correct for process group kill, but only if `pid_int` is the PGID. With `start_new_session=True` in the wrapper, the wrapper's PID == its PGID, so this is correct for the wrapper. For the child PID, it depends on whether the child used `start_new_session=True` (it did, `ftd_pm_runner_wrapper.py:72`). So `os.killpg(child_pid, SIGTERM)` is correct.
- After SIGTERM, no polling for death. Add a poll loop with timeout before SIGKILL fallback.

### `ftd_status.py`

Only shows `active_pm_runner_pid`, not `active_pm_child_pid` or `active_pm_runner_wrapper_pid`. Currently shows no watchdog info, no git state, no worker health. The spec's expansion is correct. No additional concerns — this file is thin enough to expand cleanly.

Note: `ftd_status.py:37` calls `git_root(Path(args.repo).expanduser())` which runs a subprocess. If git is not available or the path is not a repo, it raises. Status should never crash.

### `ftd_set_state.py`

**The PID-clearing at line 43 is the worst bug in this file** — worse than the missing validation. The spec addresses the missing validation but does not explicitly call out the PID-clearing side effect. See §2.2 of this review.

The `--override-validation` flag is not in the current `argparse`. It must be added. The spec correctly specifies it.

Missing from spec: what happens if `ftd_set_state.py` is called for `COMPLETE` — should it delete or just pause the watchdog? The spec table in §3.2 says COMPLETE means "delete/disable watchdog based on config" but `ftd_set_state.py` currently calls `pause_watchdog_if_known` for COMPLETE (it's in `PAUSE_STATES` which is in `SAFETY_PAUSE_STATES`).

### `ftd_pm_runner_wrapper.py`

**The `preexec_fn=_ignore_interactive_signals_in_child` issue:** The spec says to avoid it "if possible." Let's be specific: `preexec_fn` is called after `fork()` but before `exec()`. Its purpose is to set signal dispositions in the child before exec. But `exec()` resets all signal dispositions to `SIG_DFL` anyway, so the `preexec_fn` signal settings have **no effect on the actual Hermes process** after exec. The only effect is if `subprocess.Popen` is used without exec (i.e., if the command is a shell built-in or if `shell=True`). Since it's a list command, exec is used, so `preexec_fn` here is completely inert and can be removed.

The real isolation is provided by `start_new_session=True`. Remove `preexec_fn` entirely. This simplifies the code and removes the deprecation concern.

**The unconditional PID clearing at lines 114–115:**

```python
state.pop("active_pm_child_pid", None)
state.pop("active_pm_runner_wrapper_pid", None)
```

This runs in the `finally` block regardless of whether a newer runner has been spawned. If the watchdog spawned a new runner (changing `active_pm_child_pid` and `active_pm_runner_wrapper_pid`), this stale wrapper will clear the new runner's PIDs. The spec (§6.2, item 5) correctly identifies this and says to "remove active child/wrapper PID fields only if they match the current runner." But it doesn't specify what the match key is. **Recommendation:** At spawn time, write a `runner_run_id = uuid.uuid4().hex[:8]` into state. The wrapper receives this as `--run-id`. In `finally`, only clear PIDs if `state.get("runner_run_id") == args.run_id`.

### `tests/test_ftd_control_plane.py`

**All tests that call `load_ftd_modules` will fail** because `ftd_liveness_check` does not exist (line 21: `import ftd_liveness_check`). This means the entire test file is currently broken. The spec acknowledges this but does not acknowledge that `load_ftd_modules` is a shared helper used by all tests in the file — fixing the import requires creating `ftd_liveness_check.py` first.

**`init_git_repo` is broken for tests that run git commands.** The helper creates `(path / ".git").mkdir()` — a directory, not a real git repo. `git rev-parse --show-toplevel` requires a real git repo initialized with `git init`. Any test that exercises code paths reaching `git_root()` will fail with a non-zero git exit. Tests monkeypatching `git_root` avoid this, but it's fragile. **Fix:** Use `subprocess.run(["git", "init", str(path)])` in `init_git_repo` to create a real (minimal) git repo.

---

## 8. State-Machine and Watchdog Edge Cases

### 8.1 `SPAWNING_NEXT_SPRINT` recovery is underspecified

The spec's watchdog algorithm for `SPAWNING_NEXT_SPRINT`:

```text
if fresh -> silent
if stale with task id/no runner -> either spawn existing task or ERROR depending evidence
```

"Depending evidence" is not implementable by Codex. Define the exact decision:
- If `spawning_sprint_task_id` is set and age < T minutes → silent.
- If `spawning_sprint_task_id` is set and age >= T minutes and no `active_pm_child_pid` alive → spawn with existing task ID, clear `spawning_` fields, set ACTIVE.
- If `spawning_sprint_task_id` is not set → ERROR (counter advanced with no task).
- What is T? Spec must define it (suggest 5 minutes).

### 8.2 STARTING state stale detection threshold not defined

The spec says "If stale → ERROR; else silent" for `STARTING`. What is stale? A start that was interrupted after writing `STARTING` but before completing could be 10 seconds old or 10 hours old. Define a threshold (suggest: 5 minutes — long enough for Kanban create to complete, short enough to detect a crashed start).

### 8.3 `sprints_since_benjamin_checkpoint` increment timing race

The spec (§12) says this counter "increments on successful `READY_NEXT_SPRINT` closeout or on watchdog rollover, not both." But `ftd_set_state.py` runs before the watchdog rollover. If the PM calls `ftd_set_state.py --state READY_NEXT_SPRINT` and the watchdog is what increments the counter during the subsequent `READY_NEXT_SPRINT` → `SPAWNING_NEXT_SPRINT` → `ACTIVE` transition, then both `ftd_set_state.py` (if it also increments) and the watchdog would double-count. The spec says increment in only one place — but it must specify which one explicitly. Recommendation: watchdog increments during the `READY_NEXT_SPRINT` → `SPAWNING_NEXT_SPRINT` transition (single place, deterministic).

### 8.4 Checkpoint notification de-duplication relies on state hash, but hash is not defined

The spec (§4.3) says to de-duplicate on `reason_hash: "sha256"`. Hash of what, exactly? If the state reason string changes even slightly between ticks (e.g., timestamp embedded in the reason), the hash changes and a new notification fires. Define the hash input precisely: `sha256(state + "|" + state_reason_without_timestamps)` or similar.

### 8.5 Watchdog acquires project lock but lock is also held by PM runner calls to `ftd_set_state.py`

The watchdog acquires `project_lock` to read/write state. `ftd_set_state.py` also acquires `project_lock`. If the PM calls `ftd_set_state.py` while the watchdog is mid-tick, one blocks on the lock. This is correct behavior for a lock. But the watchdog could be doing external calls (Kanban create, cron resume) while holding the lock (if coded naively). These external calls may take seconds, holding the lock and blocking the PM's set_state. **Spec should explicitly state: hold the lock only for state reads and writes; release before external calls (Kanban, cron, hermes) and re-acquire after.**

---

## 9. Worker Orchestration Edge Cases

### 9.1 Claude interactive tmux health check is wrong for a control-plane floor

§7.1 specifies a Claude health check that:
1. Creates a tmux session
2. Pastes a prompt
3. Captures the pane
4. Kills the session

This is not automatable in tests. It requires a display, tmux, and interactive terminal inference. It will fail in CI, in headless cron environments, and in any context where the tty is not a real terminal. It is also slow (seconds per check).

**For the control-plane floor (Phase B), Claude health check should be:**
```bash
claude --version  # confirms binary exists and is executable
```
The tmux smoke test belongs in Phase D as an optional pre-sprint verification, not as a blocker for the watchdog/liveness/stop fixes.

### 9.2 Codex health check uses `-o <tmpfile>` but spec doesn't specify cleanup

§7.1: `codex exec --sandbox read-only -o <tmpfile> 'Say exactly OK. Do not edit files.'` — `<tmpfile>` must be created, read, and deleted. The spec doesn't mention cleanup of the temp file. If health checks run frequently, temp files accumulate. Use `tempfile.mkstemp()` in a try/finally.

### 9.3 Worker health check result staleness

The spec stores `worker_health` with `checked_at`. But the spec doesn't define how stale a check can be before it's considered invalid. If Codex was healthy 3 hours ago but is now broken, the PM will happily dispatch work to it based on stale health state. Define a `max_health_age_seconds` (suggest 300 seconds for within-sprint use, require fresh check at sprint start).

### 9.4 PM prompt assumption that profiles exist

The current `spawn_pm_runner` (`ftd_lib.py:547`) does not do profile discovery. The spec (§5.1, item 9) says to "discover available profiles with `hermes profile list` and store them." But `build_pm_prompt` currently makes no reference to discovered profiles. After discovery, the prompt must be conditional: if only `default` exists, the prompt should say "use direct CLI workers only; do not assign Kanban child tasks to non-default profiles." Currently the PM gets a prompt referencing `galtcode`/`galtresearch` via the old config, or gets a bare prompt with no guidance on workers.

---

## 10. Kanban/Profile/Dispatcher Edge Cases

### 10.1 `hermes kanban dispatch` tenant filtering is not confirmed to exist

§8.3: "If Hermes CLI dispatch lacks tenant/assignee filtering, V2 must either: add support in Hermes Kanban dispatch, or avoid generic `dispatch`..." — this is a deferred decision that Codex cannot make. The spec must state which path is chosen. Inspecting `ftd_lib.py:700–704`, `dispatch_board` calls `hermes kanban --board <board> dispatch --max <n>` with no tenant filter. If the board is project-private (only FTD tasks on it), tenant filtering is not needed. If the board is shared, this dispatches non-FTD tasks. **Recommendation:** Enforce that each FTD project gets a dedicated board (current behavior), and remove the tenant-filter ambiguity from the spec. The guard should be "board is project-private" not "tenant filter applied."

### 10.2 `idempotency_key` for sprint task does not prevent duplicate Kanban tasks if counter is wrong

The idempotency key is `ftd:{project_id}:sprint:{sprint}`. If `sprint_counter` is advanced incorrectly (see §2.3), the new sprint number generates a new idempotency key even if the old sprint task was never actually created. The key is only idempotent for the same sprint number. This means the fix for §2.3 (counter advance after Kanban success) is load-bearing for idempotency.

### 10.3 Root task created outside the lock, before STARTING is written

`ftd_start.py:88–98`: `kanban_create` for the root task runs outside `project_lock`, before the second lock section writes `STARTING`. If two concurrent starts run:
- Both create root tasks (idempotency key `ftd:{project_id}:root` deduplicates this, assuming Kanban honors it)
- Both proceed to the second lock section
- One writes STARTING, creates sprint task, spawns PM

The root task idempotency key saves this case. But the spec should explicitly acknowledge that the root task idempotency key is the deduplication mechanism for this race, not the lock.

---

## 11. Git/Resource Cleanup/Security Edge Cases

### 11.1 `ftd_git.py` branch isolation cannot be retroactively enforced

§9: "FTD must not implement directly on default branch unless `require_branch_isolation: false` is explicit. If repo is on default branch at start, create/switch to `ftd/<project>/<objective-slug>` before implementation."

But FTD doesn't implement code — the PM (via Claude/Codex) does. `ftd_git.py` can check at start time whether the repo is on the default branch, but it cannot prevent the PM from later checking out the default branch. The enforcement must be: (a) at start, create and switch to the FTD branch; (b) at closeout validation, check that the repo is not on the default branch; (c) in PM prompt, instruct to stay on FTD branch. The spec describes all three but doesn't sequence them clearly for Codex.

### 11.2 Resource ledger cleanup via `git_worktree_remove` is destructive

§10.4: "worktree path recorded and under approved base directory" is auto-cleanable. But `git worktree remove` fails if the worktree has uncommitted changes (by default). The cleanup code must either:
- Use `git worktree remove --force` (risky, destroys uncommitted work), or
- Check for dirty state before removal and skip/report if dirty.

The spec does not address this. Codex will likely use `--force` for simplicity. Specify explicitly: **do not use `--force` for worktree removal; if dirty, report as ambiguous resource and do not clean.**

### 11.3 `ftd_resources.py` cleanup method `rm_repo_tmpdir` needs path validation

§10.4: auto-clean "temp dir under `.fulltime-dev/tmp/`". The cleanup code must validate that the path being removed is strictly under `<repo>/.fulltime-dev/tmp/`, not a symlink pointing outside. A resolved path check (`Path(path).resolve().is_relative_to(repo / ".fulltime-dev" / "tmp")`) is required. Specify this in the resource module spec.

### 11.4 `drop_named_test_db` cleanup method has no authentication spec

§10.2: `cleanup_method: "drop_named_test_db"` is listed as an enum value. But dropping a database requires a connection, credentials, and a DB type (postgres, mysql, sqlite). The spec gives no schema for how the DB connection is recorded. This cleanup method cannot be implemented from the spec as written. Either specify the DB record schema or remove this method from V2 scope.

### 11.5 Security: PM prompt exposes allow_push and allow_install state as f-string

`ftd_lib.py:510–516`: prompt embeds `{allow_push_state}` and `{allow_install_override}` directly. If these are `True` when they should be `False` (e.g., due to a state file tampering), the PM acts as if they're authorized. This is already a known limitation of prompt-as-policy. The fix is validators in `ftd_validate.py`, but the spec should explicitly note this as a residual risk even with V2.

---

## 12. Test Plan Gaps

### 12.1 No test for `create_next_sprint` counter advance on Kanban failure

Test 14 in the spec ("test_create_next_sprint_idempotent_no_counter_advance_on_kanban_failure") is listed but there is no test body or description of how to trigger the Kanban failure. Codex needs a concrete scenario: mock `kanban_create` to raise, then assert `sprint_counter` is unchanged.

### 12.2 No test for `ensure_project_watchdog` silent-resume-of-dead-job

The spec lists "test_watchdog_recreates_missing_cron_job" (test 7) but the current `ensure_project_watchdog` silently returns a stale job ID. The test must: set a stale `cron_id` in config, mock `hermes cron resume` to fail, mock `cron_job_exists` to return False, and assert a new job is created and the config is updated.

### 12.3 `init_git_repo` does not create a real git repo

As noted in §7, `(path / ".git").mkdir()` is not a git repo. Tests that reach `git_root()` will fail. Fix by using `git init`.

### 12.4 No test for concurrent start (double-spawn prevention)

The spec lists "test_start_reservation_prevents_double_start" (test 13) but this requires simulating concurrent calls. A simple approach: write `STARTING` state with a `start_token`, then call `ftd_start.main()` and assert it refuses. The actual concurrency race is hard to test, but the reservation check is testable.

### 12.5 No test for wrapper stale-PID guard (spec §6.2 item 5)

Test 11 ("test_wrapper_does_not_clear_new_runner_pid_from_stale_exit") is listed but its implementation requires the `runner_run_id` mechanism described in this review (§7 critique of wrapper). The test can be written even before the mechanism exists: write state with a new `runner_run_id`, run the wrapper `finally` block with an old run ID, assert PIDs are not cleared. This test will fail until the `runner_run_id` mechanism is implemented.

### 12.6 No integration test for full start → READY_NEXT_SPRINT → watchdog rollover

The Phase G canary test is described but there's no plan for it to be automated. Without automation, it's a manual checklist that will be skipped under time pressure. At minimum, a test that mocks Hermes subprocess calls and exercises the full state machine path should be required before marking Phase B complete.

### 12.7 No test for `dispatch_board` ACTIVE guard

After adding the ACTIVE guard to `dispatch_board`, there should be a test: call with `state["state"] == "READY_NEXT_SPRINT"`, assert no `hermes kanban dispatch` subprocess is launched.

### 12.8 No test for `ftd_set_state.py` not clearing PIDs

After fixing the PID-clearing behavior, a test: set state ACTIVE with `active_pm_child_pid=123`, call `ftd_set_state.py --state READY_NEXT_SPRINT`, assert `active_pm_child_pid` is still 123 in the saved state.

---

## 13. Concrete Changes Recommended to the Spec

**Blocking changes — Codex cannot implement correctly without these:**

1. **§3.1 PID naming:** Commit to `active_pm_wrapper_pid` and `active_pm_child_pid` as the two fields. Remove the confusingly-named `active_pm_runner_pid`. Update all references in start, stop, status, wrapper, watchdog, liveness, and tests. Or keep `active_pm_runner_pid` as the wrapper PID (current behavior) and drop `active_pm_runner_wrapper_pid` as redundant. Either is fine; pick one.

2. **§6.2:** Add: "The wrapper must use a `runner_run_id` token (written to state at spawn time, passed as `--run-id` to wrapper) to guard against clearing a newer runner's PID fields in the `finally` block."

3. **§5.1 start transaction:** Add: "Write `STARTING` with a `start_token = uuid4().hex[:8]` before releasing the lock for external calls. A second concurrent start call that finds `STARTING` with a recent (< 5 minute) `start_token` must refuse rather than proceeding." This is the actual reservation mechanism.

4. **§4.4:** Add: "Specify the exact CLI invocation and output format for `cron_job_exists`. If `hermes cron status <id>` returns exit 0 for live jobs and nonzero for dead/missing jobs, use exit code only. If not available, use `hermes cron list --json` and parse the job ID from the JSON array."

5. **§4.5:** Add: "Specify what Kanban CLI command produces the `task` dict for `sprint_task_control_plane_problem`. If `hermes kanban --board <board> list --json --task-id <id>` returns the task dict, use that. Document the exact JSON fields guaranteed to be present."

6. **Remove `preexec_fn` from wrapper spec:** Add to §6.2: "Remove `preexec_fn=_ignore_interactive_signals_in_child` from `subprocess.Popen`. The signal setting is inert after exec and the `start_new_session=True` already provides session isolation."

7. **§12 checkpoint counter:** Specify exactly: "Watchdog increments `sprints_since_benjamin_checkpoint` during the `READY_NEXT_SPRINT` → `SPAWNING_NEXT_SPRINT` transition. `ftd_set_state.py` does not increment it."

8. **§5.2 stop:** Add: "After sending SIGTERM to child and wrapper, poll every 0.5 seconds up to `--timeout-seconds` (default 10) for process death. If child still alive after timeout, send SIGKILL. Record what was killed and whether SIGKILL was required."

**Non-blocking improvements:**

9. **§1.3 Phase A source-backed structure:** Downgrade this from Phase A to a post-implementation step. Develop directly in `~/.hermes/scripts/` during Phases B–F. Move to source-backed copies after Phase G canary succeeds. Rationale: two-version drift during development is harder to manage than late migration.

10. **§7.1 Claude health check:** Move tmux-based Claude health to Phase D, optional. Phase B health check for Claude: `claude --version` only. Codex smoke test only: `command -v codex && codex --version`.

11. **§10.2 `drop_named_test_db`:** Remove from V2 scope or add DB connection schema. Mark as "future" in the cleanup method enum.

12. **§8 Kanban dispatch tenant filter:** Explicitly state: "FTD projects use dedicated per-project boards. Tenant filter is not applied to `dispatch_board`. The board-is-private assumption is the isolation mechanism."

---

## 14. Recommended Implementation Order

The spec's Phase A (source-backed structure) should be done last, not first. The correct order:

### Phase 1 — Make the test suite not crash (1–2 hours)

1. Create stub `ftd_liveness_check.py` with `sprint_task_control_plane_problem` returning the expected message format. No other logic.
2. Fix `init_git_repo` in the test file to use `git init`.
3. Run tests. `test_liveness_flags_live_generic_worker_on_active_pm_sprint_task` should pass. Others should pass or be skipped cleanly.

This unblocks all subsequent test-driven development.

### Phase 2 — Fix the three critical bugs (2–4 hours)

1. Fix `ftd_stop.py`: acquire lock, set STOPPING, terminate `active_pm_child_pid` first (with SIGTERM + poll + SIGKILL), then wrapper, clear PIDs, set OFF, pause/delete watchdog.
2. Fix `ftd_pm_runner_wrapper.py`: add `--run-id` parameter; add exit-zero-while-ACTIVE → ERROR; guard PID clear by `runner_run_id` match; remove `preexec_fn`.
3. Fix `spawn_pm_runner`: write `runner_run_id`; rename stored field to `active_pm_wrapper_pid` (or confirm naming choice from §13 item 1).
4. Fix `ftd_set_state.py`: remove PID-clearing line; add `--override-validation` flag (gate is always open for now — fill in Phase 4).
5. Fix `create_next_sprint`: counter advance after Kanban success.
6. Fix `dispatch_board`: add ACTIVE guard.
7. Fix `default_repo_config`: remove hardcoded nonexistent profiles.
8. Tests: test 9, 10, 11, 12, and the new §12.8 test should now pass.

### Phase 3 — Create minimal watchdog and liveness (4–6 hours)

1. Implement `ftd_watchdog.py` with the state machine algorithm from §4.2. Drop the `--all` mode; require `--project-id`.
2. Implement full `ftd_liveness_check.py` beyond the stub.
3. Fix `ensure_project_watchdog` with actual job-existence check.
4. Fix `STARTING` reservation with `start_token` in `ftd_start.py`.
5. Add schema migration for V1 → V2 state files.
6. Tests: 1–8, 13 should pass.

### Phase 4 — Validators and state gate (2–3 hours)

1. Implement `ftd_validate.py` with handoff, board, git, and cleanup stubs.
2. Wire into `ftd_set_state.py`.
3. Tests: 15, 16, 17.

### Phase 5 — Git policy (1–2 hours)

1. Implement `ftd_git.py`.
2. Tests: 19, 20.

### Phase 6 — Resource ledger (2–3 hours)

1. Implement `ftd_resources.py` with process, worktree, tmpdir cleanup methods. Skip `drop_named_test_db`.
2. Tests: 21, 22.

### Phase 7 — Worker health and PM prompt update (1–2 hours)

1. Implement `ftd_workers.py` with Codex version check and `claude --version` only.
2. Update `build_pm_prompt` to shrink policy text and add validator command references.
3. Add profile discovery to `ftd_start.py`.
4. Tests: 18.

### Phase 8 — Status expansion (1 hour)

1. Expand `ftd_status.py` with `--json`, `--liveness`, `--watchdog` flags.
2. Tests: 23.

### Phase 9 — Canary (1–2 hours)

1. Manual end-to-end in a disposable git repo.
2. Verify all acceptance criteria from §16.

### Phase 10 — Source-backed copy (deferred)

1. Move to `hermes-agent/local/ftd/` after canary passes.
2. Write `install_runtime.py`.
3. Add to Hermes repo and commit.

**Do not do Phase 10 before Phase 9.** Source-backing half-finished code creates churn.

---

*End of oppositional review. All findings are based on direct code inspection of the runtime files. No runtime commands were executed. No files were modified.*
