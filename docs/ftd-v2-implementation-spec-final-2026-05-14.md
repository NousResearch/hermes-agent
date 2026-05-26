# Full-Time Developer V2 Implementation Spec — Final

**Date:** 2026-05-14
**Author:** John Galt / Hermes Agent
**Status:** Final planning/specification artifact. No implementation performed.
**Input artifacts:**

- Draft spec: `docs/ftd-v2-implementation-spec-draft-2026-05-14.md`
- Claude opposition: `docs/ftd-v2-implementation-spec-claude-opposition-2026-05-14.md`
- Runtime scripts inspected: `/Users/johngalt/.hermes/scripts/ftd_*.py`
- Runtime tests inspected: `/Users/johngalt/.hermes/scripts/tests/test_ftd_control_plane.py`

---

## 0. Executive Decision

V2 is a **controlled replacement/refactor** of the existing FTD control plane. It is not a continuation of the old architecture by default.

Keep existing operator entrypoints so current commands and cron wrappers do not break:

- `ftd_start.py`
- `ftd_stop.py`
- `ftd_status.py`
- `ftd_set_state.py`
- `ftd_pm_runner_wrapper.py`
- `ftd_lib.py`

But rewrite or heavily modify their internals where the current implementation is unsound.

The first implementation goal is **not** Claude planner + Codex worker automation. The first goal is a working, testable control plane:

> liveness, watchdog, correct PID semantics, safe stop, wrapper exit handling, state-transition validation, cron verification, profile discovery, branch/resource safety, and tests.

Until that works, any autonomous worker pipeline is theater.

---

## 1. Non-Negotiable Invariants

1. **One active PM runner per FTD project.** Concurrent starts must not create duplicate runners.
2. **Fresh PM context per sprint.** Continuity lives in `.fulltime-dev/HANDOFF.md`, not transcript memory.
3. **Cron is deterministic supervision only.** Cron must not become the developer.
4. **Kanban is a durable child-task ledger, not PM lifecycle owner.**
5. **`ftd-control-plane` PM/root/sprint cards are nonspawnable records.** Generic Kanban workers must not claim them.
6. **Workers are not trusted.** Galt PM verifies worker claims by inspecting diffs, running commands, and dogfooding when feasible.
7. **State transitions are validated by code.** Prompt instructions are not enforcement.
8. **Stop kills the real child process, not just the wrapper.**
9. **No cleanup without ownership proof.** No arbitrary shell cleanup commands.
10. **No default references to nonexistent worker profiles.** Runtime must discover profiles.
11. **Branch/push safety is enforced in code.** No force push, no default-branch work unless explicit config/user policy allows it.
12. **Benjamin is not notified on every sprint.** Notify only for checkpoint, approval, blocker, error, completion, or enforced caps.
13. **No implementation should happen from this spec until explicitly requested.**

---

## 2. Existing Implementation Assessment

### 2.1 Existing files and V2 decisions

| File | Current finding | V2 decision |
|---|---|---|
| `/Users/johngalt/.hermes/scripts/ftd_lib.py` | Useful primitives mixed with policy, prompts, cron, Kanban, process logic. | **Modify heavily.** Keep stable primitives; fix broken helpers; avoid making it larger. |
| `/Users/johngalt/.hermes/scripts/ftd_start.py` | Useful CLI; unsafe transaction shape; side effects outside lock; double-start race. | **Rewrite internals, preserve CLI.** Add STARTING reservation token. |
| `/Users/johngalt/.hermes/scripts/ftd_stop.py` | Too small and wrong; kills wrapper PID only. | **Rewrite, preserve CLI.** Must kill child first, then wrapper, with polling/SIGKILL fallback. |
| `/Users/johngalt/.hermes/scripts/ftd_status.py` | Thin status; lacks child PID, watchdog, liveness, JSON. | **Expand, preserve CLI.** |
| `/Users/johngalt/.hermes/scripts/ftd_set_state.py` | No validation; incorrectly clears runner PID on non-ACTIVE states. | **Rewrite interior.** Must not clear PID fields; add validators and override flag. |
| `/Users/johngalt/.hermes/scripts/ftd_pm_runner_wrapper.py` | Good concept; flawed exit behavior; stale PID clearing; no run token. | **Modify.** Add runner run ID, exit-zero ACTIVE erroring, guarded PID clearing. |
| `/Users/johngalt/.hermes/scripts/tests/test_ftd_control_plane.py` | Currently fails because `ftd_liveness_check.py` is missing; git test helper fake `.git` is not a real repo. | **Expand/fix.** First implementation should make tests collect. |
| `/Users/johngalt/.hermes/scripts/ftd_watchdog.py` | Missing. Generated wrappers import it and currently crash. | **Create.** |
| `/Users/johngalt/.hermes/scripts/ftd_liveness_check.py` | Missing. Tests import it and currently fail. | **Create.** |

### 2.2 Current verified hard failures

Verified by inspection and/or test run:

1. `ftd_liveness_check.py` does not exist; `pytest ~/.hermes/scripts/tests/test_ftd_control_plane.py` fails at import.
2. `ftd_watchdog.py` does not exist; generated watchdog wrappers import a missing module.
3. `ftd_stop.py` kills `active_pm_runner_pid`; current code uses that as wrapper PID, while real Hermes child is `active_pm_child_pid`.
4. `ftd_pm_runner_wrapper.py` only converts nonzero exit while `ACTIVE` to `ERROR`; exit 0 while still `ACTIVE` is silently accepted.
5. `ftd_pm_runner_wrapper.py` clears child/wrapper PID fields unconditionally in `finally`, even if a newer runner was spawned.
6. `ftd_set_state.py` clears `active_pm_runner_pid` on any non-ACTIVE state, while wrapper/child may still be alive.
7. `create_next_sprint()` mutates `sprint_counter` before the Kanban task exists; the save happens after `kanban_create`, but callers and retry paths still need atomic semantics specified.
8. `ensure_project_watchdog()` resumes stored cron IDs with `check=False` and returns stale/deleted cron IDs as live.
9. `dispatch_board()` has no internal `ACTIVE`/PM-alive guard.
10. `default_repo_config()` hardcodes nonexistent profiles (`galtcode`, `galtresearch`, `galtops`) while only `default` exists on this install.

---

## 3. V2 System Architecture

```text
explicit start/resume
  -> ftd_start.py lock-protected STARTING reservation
  -> repo-local .fulltime-dev/ files
  -> external state ~/.hermes/fulltime-dev/projects/<project-id>.json
  -> per-project no-agent watchdog cron
  -> fresh Hermes PM runner for sprint
  -> PM optionally uses Claude Code for planning/review and Codex for implementation
  -> PM verifies, reconciles Kanban, updates handoff, validates closeout
  -> ftd_set_state.py records terminal/continuation state only after validation
  -> watchdog either silently starts next sprint or pauses/notifies
```

### 3.1 Responsibility split

| Component | Owns | Does not own |
|---|---|---|
| `ftd_start.py` | initialization, reservation, board/root/sprint setup, spawn | ongoing development |
| `ftd_watchdog.py` | per-project deterministic state transitions and PM liveness | coding/planning |
| `ftd_liveness_check.py` | global anomaly detection across projects | mutation except reporting |
| Hermes PM runner | orchestration, decomposition, verification, final quality | durable daemon lifecycle |
| Claude Code | planning and oppositional review when healthy | final acceptance |
| Codex CLI | bounded implementation missions when healthy | final acceptance, push/merge |
| Kanban | child task ledger and audit trail | PM control-plane ownership |
| validators | machine-enforced transition gates | broad reasoning |
| resource ledger | ownership proof for cleanup | deleting unknown resources |

---

## 4. State Schema

### 4.1 Naming decision: two PIDs plus run token

Do **not** keep the confusing three-PID schema from the draft.

V2 uses:

```json
{
  "active_pm_wrapper_pid": 123,
  "active_pm_child_pid": 456,
  "active_pm_run_id": "8char-or-uuid-token"
}
```

Compatibility:

- Existing V1 state may contain `active_pm_runner_pid`. Treat it as legacy wrapper PID.
- Migration copies `active_pm_runner_pid` → `active_pm_wrapper_pid` if the new field is absent.
- New code should write `active_pm_wrapper_pid`, not `active_pm_runner_pid`.
- Status/liveness may display legacy fields for diagnosis but should normalize state internally.

Reason: in current code `active_pm_runner_pid` is actually the wrapper PID. Keeping that name invites repeated bugs.

### 4.2 Project state file

Path:

```text
~/.hermes/fulltime-dev/projects/<project-id>.json
```

Required V2 fields:

```json
{
  "schema_version": 2,
  "project_id": "repo-8hex",
  "repo": "/absolute/repo/path",
  "repo_realpath": "/resolved/repo/path",
  "logical_repo_id": "optional remote/root identity for multi-worktree detection",
  "board": "ftd-repo-8hex",
  "tenant": "ftd:repo-8hex",
  "state": "ACTIVE",
  "objective": "...",
  "root_task_id": "t_...",
  "active_sprint_task_id": "t_...",
  "sprint_counter": 3,
  "sprints_since_benjamin_checkpoint": 2,
  "started_at": "iso",
  "updated_at": "iso",
  "last_state_set_at": "iso",
  "active_started_at": "iso",
  "last_benjamin_checkpoint_at": null,
  "active_pm_wrapper_pid": 123,
  "active_pm_child_pid": 456,
  "active_pm_run_id": "abc12345",
  "active_pm_runner_log": "...",
  "active_pm_runner_prompt": "...",
  "last_pm_runner_exit": {},
  "watchdog": {
    "cron_id": "...",
    "deliver": "discord:<channel>:<thread>",
    "schedule": "every 2m",
    "script": "ftd_watchdog_<project-id>.py",
    "verified_at": "iso"
  },
  "available_profiles": ["default"],
  "worker_health": {},
  "git": {},
  "last_notification": {}
}
```

### 4.3 Migration

Create a migration helper, likely in `ftd_lib.py` initially:

```python
def migrate_state_to_v2(state: dict) -> dict:
    ...
```

Migration rules:

1. If `schema_version` missing, treat as V1.
2. If `active_pm_runner_pid` exists and `active_pm_wrapper_pid` absent, copy it to `active_pm_wrapper_pid`.
3. Do not delete legacy fields during first migration; keep them for diagnosis until all tools are updated.
4. Add `schema_version: 2`.
5. Populate `repo_realpath` from `Path(repo).resolve()` if possible.
6. Move known project watchdog config from global config into state if available, but keep global config fallback for compatibility.
7. Migration must be idempotent.

Every state-loading entrypoint must either call migration or use a `load_project_state()` wrapper that does.

---

## 5. State Machine

### 5.1 Valid states

| State | Meaning | Watchdog behavior |
|---|---|---|
| `OFF` | Disabled/stopped | Silent; no dispatch/spawn |
| `STARTING` | Start transaction reserved | If recent, silent/refuse duplicate start; if stale, `ERROR` |
| `ACTIVE` | PM runner should be alive | Verify liveness; dispatch child work only if PM alive |
| `READY_NEXT_SPRINT` | PM closeout succeeded; internal rollover | Spawn next sprint silently unless checkpoint cap reached |
| `SPAWNING_NEXT_SPRINT` | Transient rollover in progress | Recover or error if stale |
| `FEATURE_CHECKPOINT_READY_FOR_BENJAMIN` | Substantial user-testable bundle | Pause project watchdog and notify once |
| `CHECKPOINT_READY_FOR_BENJAMIN` | Legacy alias | Normalize/treat as feature checkpoint |
| `APPROVAL_REQUIRED` | Explicit safety gate | Pause and notify |
| `BLOCKED_NEEDS_INPUT` | External blocker | Pause and notify |
| `ERROR` | Control-plane/runtime error | Pause and notify |
| `COMPLETE` | Objective complete | Notify once; delete/disable watchdog per policy |
| `STOPPING` | Stop transaction in progress | Finish stop or flag stale stop |

### 5.2 Staleness thresholds

Define constants:

```python
STARTING_STALE_SECONDS = 300
SPAWNING_STALE_SECONDS = 300
WATCHDOG_HARD_TIMEOUT_SECONDS = 90
WORKER_HEALTH_MAX_AGE_SECONDS = 300
```

Use config overrides later; hardcoded constants are acceptable for Phase 1/2.

### 5.3 Checkpoint counter rule

Only the watchdog increments `sprints_since_benjamin_checkpoint`, during `READY_NEXT_SPRINT -> SPAWNING_NEXT_SPRINT`.

`ftd_set_state.py` must not increment it.

This prevents double-counting.

### 5.4 Notification de-duplication

Use deterministic notification keys:

```text
notification_key = sha256(project_id + "|" + state + "|" + normalized_reason + "|" + active_sprint_task_id)
```

`normalized_reason` must strip timestamps and volatile log-tail content. Prefer explicit `last_reason`/`last_summary` strings written by state setters.

State stores:

```json
{
  "last_notification": {
    "state": "ERROR",
    "key": "sha256...",
    "sent_at": "iso"
  }
}
```

Identical key: no repeat notification.

---

## 6. File-by-File Implementation Spec

### 6.1 Modify `ftd_lib.py`

Keep:

- `project_lock`
- `save_json` / `load_json`
- `slugify`
- `project_id_for`
- `board_slug`
- `project_state_path`
- `process_alive`
- `git_root`
- `iter_states`
- `now_iso`
- `ensure_dirs`
- constants: `VALID_STATES`, `PAUSE_STATES`, `BENJAMIN_REVIEW_STATES`, `FTD_CONTROL_ASSIGNEE`
- `ftd_control_assignee`

Modify:

1. Add `migrate_state_to_v2()` and ensure state loads normalize V1.
2. Add `active_pm_wrapper_pid` support; treat `active_pm_runner_pid` as legacy.
3. Update `default_repo_config()`:

```yaml
kanban:
  pm_assignee: "ftd-control-plane"
  child_dispatch_mode: "direct_cli"
  default_profile_assignee: "default"
  implementer_assignee: null
  reviewer_assignee: null
  researcher_assignee: null
  ops_assignee: null
  strict_profile_validation: false
  max_dispatch_per_tick: 4
```

4. Fix `dispatch_board(state)` internally:

```python
if state.get("state") != "ACTIVE": return
if not process_alive(state.get("active_pm_child_pid")) and not process_alive(state.get("active_pm_wrapper_pid")): return
```

Then call `hermes kanban --board <board> dispatch --max <n>` only on dedicated project board.

5. Fix `ensure_project_watchdog()`:
   - if stored cron ID exists, attempt resume;
   - verify existence;
   - if missing/dead, recreate;
   - update state/global config with verified cron ID and timestamp.

6. Add `cron_job_exists(job_id)`.

Preferred detection order:

```text
A. If `hermes cron list --json --all` exists, parse JSON and match `id`/`job_id`.
B. Else use `hermes cron list --all` text and match the exact job ID as a token.
C. Treat parse failure as unknown; recreate only when resume command failed clearly. Otherwise report liveness warning.
```

Do not invent nonexistent `hermes cron status <id>` unless verified by implementation.

7. Fix `create_next_sprint()` atomicity:
   - compute `next_sprint = old + 1` locally;
   - call `kanban_create(... idempotency_key=f"ftd:{project_id}:sprint:{next_sprint}")`;
   - only after `kanban_create` succeeds, set `sprint_counter`, `spawning_sprint_counter`, `active_sprint_task_id`, `spawning_sprint_task_id`, and save.

8. Fix `spawn_pm_runner()`:
   - generate `active_pm_run_id = uuid.uuid4().hex`;
   - pass `--run-id` to wrapper;
   - write `active_pm_wrapper_pid = proc.pid`;
   - preserve legacy `active_pm_runner_pid = proc.pid` temporarily only for compatibility, but new code must read `active_pm_wrapper_pid` first.

Avoid moving too many functions in the first implementation. Refactor boundaries after behavior is correct.

### 6.2 Rewrite `ftd_start.py` internals

Preserve CLI flags:

- `repo`
- `--objective`
- `--allow-push`
- `--allow-install`
- `--yolo`
- `--no-spawn`
- `--continue-paused`
- `--pm-assignee` compatibility flag
- `--watchdog-deliver`
- `--no-watchdog-ensure`

Add later if needed:

- `--from-existing-sprint`
- `--resume-existing-sprint`

Start algorithm:

```text
resolve repo git root
project_id = project_id_for(repo)
start_token = uuid4
acquire project_lock
  load/migrate state
  if ACTIVE and child/wrapper alive: refuse
  if STARTING and recent token exists: refuse
  if pause state and not --continue-paused: refuse
  write STARTING reservation with start_token, objective, started_at/current timestamp
release lock
ensure repo files
read config flags
verify config does not conflict with start flags/standing safety rules
discover profiles and configured missing assignees
create/verify project-private board
create/reuse root task idempotently
ensure/recreate project watchdog unless disabled
create/reuse sprint task idempotently
acquire lock
  reload state
  verify start_token still matches
  if --no-spawn: set READY_NEXT_SPRINT and spawn_existing_sprint_task_id
  else spawn_pm_runner and set ACTIVE
release lock
resume watchdog if configured and desired
print status
```

If any failure occurs after `STARTING` reservation, write `ERROR` with reason unless the process is killed before it can write.

A second start that finds recent `STARTING` must refuse. This is the actual duplicate-start guard.

### 6.3 Rewrite `ftd_stop.py`

Preserve CLI entrypoint.

Required flags:

```bash
--keep-watchdog
--delete-watchdog
--timeout-seconds 10
--json
```

Default behavior: pause watchdog on `OFF`; delete only with `--delete-watchdog` or complete-state cleanup policy.

Algorithm:

```text
resolve repo/project
acquire lock
  load/migrate state
  write STOPPING with stop reason and timestamp
  capture active_pm_child_pid, active_pm_wrapper_pid, legacy active_pm_runner_pid
release lock
terminate child process group first
poll every 0.5s until dead or timeout
if still alive, SIGKILL child process group
terminate wrapper process group
poll; SIGKILL if necessary
acquire lock
  reload state
  clear active PID/run fields only if they match captured values
  set OFF
  write last_summary with killed/not-found/SIGKILL details
release lock
pause/delete watchdog according to flags
print exact result
```

Use `os.killpg(pid, SIGTERM)` because both wrapper and child are launched as session/process-group leaders. Fallback to `os.kill(pid, SIGTERM)` if group kill fails.

### 6.4 Expand `ftd_status.py`

Preserve existing human output. Add:

```bash
--json
--kanban
--open-only
--active-sprint-only
--liveness
--watchdog
```

Status must never crash just because repo path is invalid or no git root exists. It should print/report the error and continue for other states.

Fields to show:

- project ID, repo, state
- board, tenant, root task, active sprint task
- sprint counter and checkpoint counters
- wrapper PID and alive/dead
- child PID and alive/dead
- active PM run ID
- last PM exit
- watchdog cron ID/status/deliver/script
- worker health
- git branch/head/dirty summary if available
- resource ledger summary
- liveness problems if requested

### 6.5 Rewrite `ftd_set_state.py` interior

Preserve CLI and add:

```bash
--override-validation
--override-reason TEXT
--json
```

Critical rule:

> `ftd_set_state.py` must not clear PID fields. PID fields are cleared by the wrapper after confirmed child exit, or by `ftd_stop.py` during explicit stop.

Algorithm:

```text
resolve repo/project
load/migrate state under lock
if target state requires validation and no override:
  run validators
  if blocking failures: exit nonzero and do not mutate state
write state, last_summary, last_reason, last_state_set_at
if sprint_task_id supplied, update active_sprint_task_id
write validation evidence/override metadata
release lock
resume watchdog for READY_NEXT_SPRINT/SPAWNING/ACTIVE/STARTING
pause watchdog for pause states/OFF/STOPPING
print result
```

Validation is introduced incrementally. In the first implementation phase, add the flag and do not enforce all gates until `ftd_validate.py` exists.

### 6.6 Modify `ftd_pm_runner_wrapper.py`

Add required arg:

```bash
--run-id <active_pm_run_id>
```

At child start:

- write `active_pm_child_pid`
- write `active_pm_wrapper_pid`
- write `active_pm_run_id`

On child exit/finally:

1. write `last_pm_runner_exit` metadata including run ID, wrapper PID, child PID, sprint task, started/ended, exit code, signal, command prefix.
2. reload state.
3. only clear PID fields if `state.active_pm_run_id == args.run_id`.
4. if state is still `ACTIVE` for the same sprint and same run ID, set `ERROR` **regardless of exit code**.
5. if PM intentionally changed state to a terminal/continuation state, do not override.
6. do not clear a newer runner's PID fields.

Signal handling note:

- Keep `start_new_session=True`.
- Claude claimed `preexec_fn` signal ignoring is inert after `exec`; that is overstated. Ignored signal dispositions can survive exec on POSIX, while caught dispositions usually reset. Do **not** remove the signal behavior blindly.
- Implementation should either keep it with a test demonstrating PM child survives parent `SIGINT`/`SIGHUP`, or remove it only after replacing it with a tested signal/session strategy.

### 6.7 Create `ftd_watchdog.py`

Required path:

```text
/Users/johngalt/.hermes/scripts/ftd_watchdog.py
```

CLI:

```bash
python3 ~/.hermes/scripts/ftd_watchdog.py --project-id <id> [--json]
python3 ~/.hermes/scripts/ftd_watchdog.py --repo <path> [--json]
```

Do **not** implement global `--all` in the first version. Global checks belong in `ftd_liveness_check.py`.

Cron contract:

- stdout empty + exit 0 = healthy/no notification
- stdout non-empty + exit 0 = notify with stdout
- nonzero exit = cron error notification

State handling:

```text
set hard timeout alarm
resolve project
load/migrate state
case OFF: silent
case STARTING:
  if recent: silent
  if stale: set ERROR and emit
case ACTIVE:
  if child/wrapper dead: set ERROR and emit
  else dispatch_board(state) and silent
case READY_NEXT_SPRINT:
  if checkpoint cap reached: set FEATURE_CHECKPOINT_READY_FOR_BENJAMIN and emit
  else increment sprints_since_benjamin_checkpoint
       set SPAWNING_NEXT_SPRINT with timestamp
       create/reuse next sprint task
       spawn PM runner
       set ACTIVE
       silent unless failure
case SPAWNING_NEXT_SPRINT:
  if recent: silent
  if stale and spawning_sprint_task_id exists and no live runner: spawn existing task and set ACTIVE
  if stale without task evidence: set ERROR and emit
case checkpoint/approval/block/error/complete:
  pause/delete this project's watchdog as appropriate
  emit once using notification key
case STOPPING:
  if stale: emit liveness/control-plane warning
```

Do not hold `project_lock` while doing slow external CLI calls if avoidable. Use short critical sections: read/mark intent, release, call external command, reacquire and commit result.

### 6.8 Create `ftd_liveness_check.py`

Required path:

```text
/Users/johngalt/.hermes/scripts/ftd_liveness_check.py
```

CLI:

```bash
python3 ~/.hermes/scripts/ftd_liveness_check.py [--json] [--project-id <id>]
```

Silent on OK.

Minimum functions required by current tests:

```python
def sprint_task_control_plane_problem(state: dict, task: dict) -> str | None:
    ...
```

If state is `ACTIVE`, task is the active PM sprint task, task status is `running`, task assignee is not `ftd-control-plane`, and both generic worker PID and PM PID are alive, return a message containing:

- `generic Kanban worker PID <pid>`
- `PM runner PID <pid>` or `PM wrapper PID <pid>` compatible with legacy test wording

Production liveness checks:

- missing `ftd_watchdog.py`
- missing generated project watchdog wrapper
- `ACTIVE` with dead child/wrapper PID
- `READY_NEXT_SPRINT` older than threshold with missing/paused watchdog
- stale `STARTING` / `SPAWNING_NEXT_SPRINT`
- checkpoint/error state still dispatching work
- project watchdog cron missing/paused when project should run
- runner log contradiction: success state but log ends in `Traceback`, `KeyboardInterrupt`, or nonzero exit metadata
- PM sprint task claimed by generic Kanban worker
- superseded main-worktree FTD still active while component worktrees exist

Kanban production task dict source must be specified by implementation after inspecting CLI support. Preferred:

```bash
hermes kanban --board <board> show <task-id> --json
```

If that command does not exist, use the least fragile available JSON output. If no JSON exposes `worker_pid`, liveness should degrade gracefully and report that this check is unavailable.

### 6.9 Create `ftd_validate.py`

Purpose: closeout/state transition validators.

CLI:

```bash
python3 ~/.hermes/scripts/ftd_validate.py /repo --target-state READY_NEXT_SPRINT [--json]
```

Validator categories:

- handoff freshness/required sections
- sprint log entry
- Kanban board reconciliation
- git branch/dirty/commit/push policy
- test/check evidence
- dogfood/manual verification evidence or explicit reason skipped
- cleanup/resource inventory
- worker review evidence for code changes

Validation output schema:

```json
{
  "ok": false,
  "blocking": [],
  "warnings": [],
  "evidence": {}
}
```

### 6.10 Create `ftd_git.py`

Responsibilities:

- detect default branch
- enforce branch isolation at start and closeout
- create/switch FTD branch when configured
- block force push
- block direct default-branch push/merge unless explicit config permits
- record branch/base/head/push evidence

Default config:

```yaml
git:
  default_branch: "main"
  require_branch_isolation: true
  branch_prefix: "ftd/"
  commit_verified_sprints: true
  push_if_allow_push: false
  push_branch_only: true
  never_force_push: true
  merge_to_default_branch: false
```

### 6.11 Create `ftd_resources.py`

Purpose: ownership ledger and safe cleanup helpers.

Repo-local paths:

```text
.fulltime-dev/resources/
  cleanup-ledger.json
  ambiguous-resources.md
```

Ledger entry:

```json
{
  "kind": "process|container|worktree|tmpdir|port",
  "id": "stable-id",
  "project_id": "...",
  "sprint": 3,
  "created_at": "iso",
  "ttl_seconds": 7200,
  "safe_cleanup": true,
  "cleanup_method": "kill_process_group|docker_stop_label|git_worktree_remove|rm_repo_tmpdir",
  "metadata": {}
}
```

Do not implement `drop_named_test_db` in V2 unless a DB-specific credential/schema design is added. Mark it future.

Safety rules:

- no free-form cleanup command strings
- no unknown container/process cleanup
- no worktree `--force`; if dirty, report ambiguous and do not remove
- tmpdir cleanup must verify resolved path is under `<repo>/.fulltime-dev/tmp/` and not symlink-escaped

### 6.12 Create `ftd_workers.py`

Phase 1 worker health should be cheap and noninteractive:

```bash
command -v codex && codex --version
command -v claude && claude --version
```

Do not make tmux Claude inference a Phase 1/2 control-plane blocker.

Phase D / actual sprint worker health may add:

- Codex harmless prompt smoke test with temp output file cleaned in `finally`
- Claude interactive tmux smoke test when the sprint actually intends to use Claude Code

Worker health records must have TTL:

```json
{
  "status": "ok|failed|unknown",
  "checked_at": "iso",
  "max_age_seconds": 300,
  "evidence": "..."
}
```

### 6.13 `ftd_board.py`

Create only if Kanban logic grows too large for `ftd_lib.py`.

For first implementation, it is acceptable to keep board helpers in `ftd_lib.py`, but **not** optional to fix `dispatch_board()` guards.

---

## 7. Repo-Local `.fulltime-dev/` Contract

Required structure:

```text
.fulltime-dev/
  config.yaml
  HANDOFF.md
  SPRINTS.md
  CHECKPOINT.md
  RUNBOOK.md
  state/local-status.json
  resources/cleanup-ledger.json
  resources/ambiguous-resources.md
  reviews/sprint-###-claude-plan.md
  reviews/sprint-###-claude-review.md
  reviews/sprint-###-pm-final-review.md
  dogfood/reports/sprint-###.md
  dogfood/screenshots/
  logs/
  archive/
```

### 7.1 Handoff required sections

```markdown
# Full-Time Developer Handoff

## Project
## Current objective
## Current state
## Completed since last Benjamin checkpoint
## Current implementation facts
## Verification history
## Worker health
## Open risks
## Deferred work
## Environment/resource state
## Git state
## Next sprint recommendation
## Resume instructions
```

Handoff is compact continuity, not transcript dump.

---

## 8. Kanban Design

### 8.1 Dedicated project boards

FTD V2 assumes **one dedicated board per FTD project/worktree**. This is the isolation mechanism.

Because current `dispatch_board()` does not apply tenant filtering, do not write the spec as if tenant filtering is enforced. Tenant remains useful metadata, but board privacy is the operational boundary.

### 8.2 Control-plane cards

Root and sprint PM cards:

- assigned to `ftd-control-plane`
- never dispatched by generic Kanban dispatcher
- used as durable records and recovery anchors

### 8.3 Child tasks

Child tasks may be created for implementation, review, dogfood, or research. If only `default` profile exists, PM should prefer direct Claude/Codex CLI worker invocation instead of inventing `galtcode`-style assignees.

### 8.4 Dispatch guard

`dispatch_board(state)` must do nothing unless:

- state is `ACTIVE`
- current PM child/wrapper process is alive
- board is the project-private FTD board

---

## 9. Claude / Codex Worker Pipeline

This is not Phase 1.

### 9.1 Claude planner

When healthy and used, Claude writes:

```text
.fulltime-dev/reviews/sprint-###-claude-plan.md
```

Planner must output:

- concise sprint goals
- Codex missions
- acceptance criteria
- likely files affected
- test/dogfood plan
- non-goals
- risks

### 9.2 Codex implementer

Codex gets bounded missions. It does not own final acceptance, commit, push, or merge.

Policy:

- work inside repo/worktree only
- no package install unless allowed
- no push/merge
- output mission summary to file if supported

### 9.3 Claude reviewer

Claude writes:

```text
.fulltime-dev/reviews/sprint-###-claude-review.md
```

Findings classified:

- `BLOCKING`
- `SHOULD_FIX`
- `NON_BLOCKING`

Blocking findings become Codex fix missions, then re-review.

### 9.4 PM verification

PM independently verifies with commands/diffs/dogfood. Worker reports are not evidence.

---

## 10. Source-Backed Runtime Decision

The draft proposed source-backed copies under:

```text
/Users/johngalt/.hermes/hermes-agent/local/ftd/
```

Final decision: **defer source-backed migration until after the control plane passes canary.**

Reason: during repair, two copies create drift. Implement the control-plane fixes in runtime scripts first. After canary passes, copy the stabilized runtime code into a tracked `local/ftd/` area and add an install/sync script.

Phase after canary:

```text
local/ftd/
  ftd_*.py
  tests/
  install_runtime.py
  README.md
```

Runtime remains:

```text
~/.hermes/scripts/ftd_*.py
```

---

## 11. Tests Required

First make current tests collect. Then expand.

### 11.1 Immediate test repair

1. Create minimal `ftd_liveness_check.py` with `sprint_task_control_plane_problem()`.
2. Fix test helper `init_git_repo()` to run `git init` instead of creating a fake `.git` directory.
3. Run:

```bash
python3 -m pytest /Users/johngalt/.hermes/scripts/tests/test_ftd_control_plane.py -q
```

### 11.2 Required tests

1. `test_liveness_module_imports`
2. `test_liveness_flags_active_with_dead_child_pid`
3. `test_liveness_flags_live_generic_worker_on_active_pm_sprint_task`
4. `test_watchdog_ready_next_sprint_spawns_next_pm`
5. `test_watchdog_does_not_dispatch_in_ready_next_sprint`
6. `test_watchdog_pauses_checkpoint_state_and_notifies_once`
7. `test_watchdog_recreates_missing_cron_job`
8. `test_watchdog_errors_stale_spawning_next_sprint`
9. `test_stop_terminates_child_then_wrapper`
10. `test_wrapper_exit_zero_while_active_sets_error`
11. `test_wrapper_does_not_clear_new_runner_pid_from_stale_exit`
12. `test_start_refuses_live_active_child_pid`
13. `test_start_reservation_prevents_double_start`
14. `test_create_next_sprint_no_counter_advance_on_kanban_failure`
15. `test_set_state_blocks_ready_without_fresh_handoff`
16. `test_set_state_blocks_success_with_unreconciled_cards`
17. `test_set_state_allows_error_with_minimal_failure_evidence`
18. `test_profile_discovery_rejects_missing_configured_assignees`
19. `test_git_policy_blocks_default_branch_when_isolation_required`
20. `test_git_policy_blocks_force_push`
21. `test_resource_cleanup_rejects_arbitrary_cleanup_command`
22. `test_resource_cleanup_only_cleans_owned_tmpdir`
23. `test_status_json_contains_watchdog_liveness_git_resource_fields`
24. `test_dispatch_board_noops_unless_active_and_pm_alive`
25. `test_set_state_does_not_clear_active_pid_fields`

Specific mocking guidance:

- For sprint counter failure: mock `kanban_create` to raise; assert persisted `sprint_counter` unchanged.
- For stale cron: set stale cron ID; mock resume failure and job-exists false; assert new job created.
- For concurrent start: prewrite recent `STARTING` with token; assert second start refuses.
- For stale wrapper: write new `active_pm_run_id`; simulate old wrapper finalization; assert new PID fields remain.

---

## 12. Implementation Order

### Phase 1 — Test suite collects and liveness stub exists

1. Create minimal `ftd_liveness_check.py`.
2. Fix `init_git_repo` test helper.
3. Run tests and capture baseline.

### Phase 2 — Critical process/state bugs

1. Rewrite `ftd_stop.py` child/wrapper kill semantics.
2. Add `active_pm_run_id` and PID naming migration.
3. Fix wrapper exit-zero-while-ACTIVE => `ERROR`.
4. Guard wrapper PID clearing by run ID.
5. Remove `ftd_set_state.py` PID clearing.
6. Fix `create_next_sprint` counter ordering.
7. Fix `dispatch_board` ACTIVE/PM-alive guard.
8. Fix generated default profile config.

### Phase 3 — Watchdog/liveness floor

1. Create `ftd_watchdog.py` project mode only.
2. Expand `ftd_liveness_check.py`.
3. Add state migration.
4. Add cron verification/recreation.
5. Add STARTING reservation token to `ftd_start.py`.

### Phase 4 — Validation gates

1. Create `ftd_validate.py`.
2. Wire it into `ftd_set_state.py`.
3. Validate handoff, sprint log, Kanban reconciliation, git, resource/dogfood evidence.

### Phase 5 — Git policy

1. Create `ftd_git.py`.
2. Enforce branch isolation and push rules at start/closeout.

### Phase 6 — Resource ledger

1. Create `ftd_resources.py`.
2. Implement only safe owned process/worktree/tmpdir/container helpers.
3. No DB cleanup in V2.

### Phase 7 — Worker health and prompt shrink

1. Create `ftd_workers.py` with cheap version checks first.
2. Add optional richer Codex/Claude smoke tests for real sprint use.
3. Shrink PM prompt to reference validator/worker commands instead of restating all policy.

### Phase 8 — Status expansion

1. Add JSON/liveness/watchdog/git/resource fields to `ftd_status.py`.

### Phase 9 — Canary

Use a disposable local git repo:

1. start FTD
2. verify one wrapper and one child only
3. stop FTD and verify both dead
4. force `READY_NEXT_SPRINT`; watchdog rolls silently
5. force checkpoint; watchdog pauses/notifies once
6. kill child; watchdog marks `ERROR`
7. liveness silent on healthy and noisy on constructed failures
8. no unknown resources cleaned

### Phase 10 — Source-backed copy

Only after canary passes:

1. copy stabilized FTD runtime into `hermes-agent/local/ftd/`
2. add `install_runtime.py`
3. add README
4. commit/push only with Benjamin approval or explicit FTD project policy

---

## 13. Acceptance Criteria

V2 is ready for real autonomous use only when:

1. `ftd_watchdog.py` exists and runs project mode.
2. `ftd_liveness_check.py` exists, tests import it, and healthy output is silent with exit 0.
3. `ftd_stop.py` kills both child and wrapper and verifies death.
4. PM child exit while state remains `ACTIVE` becomes `ERROR` regardless of exit code.
5. Wrapper cannot clear newer runner PID fields.
6. `ftd_set_state.py` does not clear PID fields.
7. Start uses recent `STARTING` reservation to prevent double starts.
8. `create_next_sprint` does not persist counter advance before task exists.
9. Stale/deleted cron job IDs are detected and recreated or reported.
10. `dispatch_board` only dispatches when state is `ACTIVE` and PM is alive.
11. Default generated config does not reference nonexistent profiles.
12. Terminal/continuation states validate handoff/board/git/resource evidence or record explicit override.
13. Branch isolation and no-force-push rules are enforced in code.
14. Cleanup cannot execute arbitrary shell strings.
15. Status JSON exposes process, watchdog, state, git, worker, and liveness data.
16. Claude/Codex worker health is recorded before real use.
17. Checkpoint caps are machine-enforced.
18. Disposable canary passes.

---

## 14. Final Notes from Claude Opposition and Galt Review

### 14.1 Accepted Claude findings

I accept these as correct and integrated:

- PID naming was ambiguous; final spec uses wrapper PID + child PID + run ID.
- `ftd_set_state.py` PID clearing is a real bug and must be fixed.
- `create_next_sprint` atomicity needed a concrete fix.
- `ensure_project_watchdog` stale cron behavior needed implementation detail.
- Missing `ftd_watchdog.py` and `ftd_liveness_check.py` must be first-class blockers.
- Source-backed copies should be deferred until runtime is stable, not done first.
- `dispatch_board` guard belongs inside the function, not only in callers.
- V1→V2 schema migration is required.
- `STARTING` and `SPAWNING_NEXT_SPRINT` staleness thresholds must be explicit.
- `drop_named_test_db` is too underspecified for V2.
- Worktree cleanup must not use force on dirty worktrees.

### 14.2 Claude finding modified

Claude said `preexec_fn` signal ignoring is inert after exec. That is not reliably correct as stated: ignored signal dispositions can survive exec on POSIX. Final spec does **not** mandate removing it blindly. It mandates a tested signal/session strategy.

### 14.3 Main architecture correction

The draft leaned toward many new modules immediately. The final spec keeps the direction but orders the work differently:

1. fix runtime/control-plane bugs first;
2. add missing watchdog/liveness;
3. add validators;
4. only then worker orchestration;
5. source-back after canary.

That is less elegant than a greenfield design, but it reduces operational risk.

---

## 15. Codex Handoff Summary

When implementation is approved, hand Codex this final file and tell it:

1. Do **not** implement Claude/Codex sprint planning first.
2. Start with Phase 1 and Phase 2.
3. Write tests before or alongside each control-plane fix.
4. Preserve CLI entrypoints.
5. Do not delete old state files.
6. Do not push or commit without approval.
7. Do not install packages.
8. Verify after every phase with tests and a disposable canary.
