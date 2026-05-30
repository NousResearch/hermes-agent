# Full-Time Developer V2 Implementation Spec — Draft

**Date:** 2026-05-14
**Author:** John Galt / Hermes Agent
**Status:** Draft for oppositional Claude Code review. No implementation performed.
**Target reader:** Codex implementer with full local file access.
**Primary target runtime:** Benjamin's Galt Hermes install on macOS.
**Repo context:** `/Users/johngalt/.hermes/hermes-agent` is source repo; current FTD runtime scripts live outside the repo in `/Users/johngalt/.hermes/scripts/`.

---

## 0. Hard Position

Do **not** preserve the old FTD architecture by default. Preserve only the parts that directly serve the V2 design and are simpler/safer than replacing them.

The current implementation is an incomplete prototype, not a foundation to trust. It has useful pieces:

- repo-local `.fulltime-dev/` convention
- external project state under `~/.hermes/fulltime-dev/`
- project IDs / board slugs / locks
- reserved nonspawnable `ftd-control-plane` assignee
- PM runner wrapper concept
- project-scoped watchdog wrapper concept
- start/status/stop/operator CLI affordances

But it also has hard failures:

- missing `ftd_watchdog.py`
- missing `ftd_liveness_check.py`
- test suite fails immediately because `ftd_liveness_check` is missing
- `ftd_stop.py` terminates wrapper PID, not actual Hermes child PID
- wrapper treats exit code 0 while state remains `ACTIVE` as acceptable
- `ftd_set_state.py` trusts prompts instead of enforcing closeout preconditions
- default child assignees are nonexistent profiles (`galtcode`, `galtresearch`, `galtops`) while only `default` exists
- cron job resume does not verify stored job IDs are live
- start path has avoidable race windows and side effects outside the project lock
- cleanup is aspirational, not enforced

V2 should therefore be a **controlled replacement/refactor**, not a patch pile.

---

## 1. Current Implementation Inventory

### 1.1 Existing runtime files to inspect/use

Current runtime files verified present under `/Users/johngalt/.hermes/scripts/`:

| File | Current status | V2 decision |
|---|---:|---|
| `ftd_lib.py` | Exists; too broad; useful helpers mixed with config, prompts, state, cron, Kanban, process control | **Modify heavily**. Keep core helpers only; split new behavior into focused helper functions. |
| `ftd_start.py` | Exists; creates repo files/board/root/sprint and spawns PM; race windows | **Modify heavily**. Preserve CLI entrypoint but rewrite start transaction. |
| `ftd_stop.py` | Exists; wrong PID termination | **Modify**. Preserve CLI entrypoint; fix process-tree termination/state semantics. |
| `ftd_status.py` | Exists; minimal status only | **Modify**. Preserve CLI entrypoint; add JSON/status/liveness/watchdog fields. |
| `ftd_set_state.py` | Exists; prompt-only state trust | **Modify heavily**. Preserve CLI entrypoint; add validation gates and explicit override mode. |
| `ftd_pm_runner_wrapper.py` | Exists; useful wrapper/metadata concept; flawed exit semantics and signal handling | **Modify**. Keep concept; fix active-exit handling and process-group termination. |
| `tests/test_ftd_control_plane.py` | Exists; currently fails due missing liveness module; limited coverage | **Modify/expand**. Keep existing tests; add V2 tests before or alongside code changes. |

### 1.2 New runtime files V2 must create

| New file | Required? | Purpose |
|---|---:|---|
| `ftd_watchdog.py` | Yes | Deterministic project/global watchdog. No-agent cron target. Silent on healthy/no-op. |
| `ftd_liveness_check.py` | Yes | Global sentinel; detects stale/dead/split-brain conditions. Silent on OK. Tests import this. |
| `ftd_validate.py` | Yes | Closeout/state transition validators: handoff, git, board, verification, cleanup, resources. |
| `ftd_resources.py` | Yes | Owned-resource ledger helpers; no arbitrary cleanup shell strings. |
| `ftd_workers.py` | Yes | Claude/Codex health checks and worker invocation contracts. |
| `ftd_git.py` | Yes | Branch/commit/push policy helpers; enforce `never_force_push`, branch-only push, default branch protection. |
| `ftd_board.py` | Optional but recommended | Kanban wrapper helpers, tenant filtering, control-card detection, idempotent creation. |
| `ftd_spec.md` or generated runbook templates | Optional | Current spec/reference material; not required at runtime. |

Reason for new helper files: `ftd_lib.py` is already a god module. V2 should not make it worse. Keep `ftd_lib.py` as stdlib-safe primitives and stable compatibility API; put policy subsystems in dedicated modules.

### 1.3 Source-control decision

Current runtime scripts live under `~/.hermes/scripts/`, outside the Hermes repo. That is operationally convenient but bad for review, rollback, and Codex handoff.

V2 should support **both**:

1. **Runtime location:** `/Users/johngalt/.hermes/scripts/ftd_*.py` because existing operator commands and cron wrappers call those paths.
2. **Source-backed copies:** add tracked copies under the Hermes repo, recommended path:

```text
/Users/johngalt/.hermes/hermes-agent/local/ftd/
  ftd_lib.py
  ftd_start.py
  ftd_stop.py
  ftd_status.py
  ftd_set_state.py
  ftd_pm_runner_wrapper.py
  ftd_watchdog.py
  ftd_liveness_check.py
  ftd_validate.py
  ftd_resources.py
  ftd_workers.py
  ftd_git.py
  ftd_board.py
  tests/test_ftd_control_plane.py
  README.md
```

Then provide a sync/install command:

```bash
python3 local/ftd/install_runtime.py --to ~/.hermes/scripts
```

If we do not want tracked repo changes yet, Codex can implement directly in `~/.hermes/scripts/`; but that repeats the old problem: important operational code outside source control. My recommendation is source-backed local module plus copied runtime wrappers.

Do not upstream this to public Hermes core until it is clean, generalizable, and stripped of Benjamin/Galt-specific policy.

---

## 2. V2 Architecture

### 2.1 System shape

```text
Explicit user start/resume
  -> ftd_start.py transaction
  -> repo-local .fulltime-dev/ config/handoff/resources
  -> external project state ~/.hermes/fulltime-dev/projects/<id>.json
  -> per-project no-agent cron watchdog
  -> fresh Galt PM Hermes runner for each sprint
  -> Claude Code interactive/tmux planner/reviewer when healthy
  -> Codex CLI implementer when healthy
  -> Galt PM verifies, reconciles board, updates handoff, commits/pushes if policy allows
  -> ftd_set_state.py validates closeout and transitions
  -> watchdog starts next sprint silently OR pauses/notifies on checkpoint/block/error/approval/complete
```

### 2.2 Responsibility split

| Component | Responsibility | Not responsible for |
|---|---|---|
| Cron watchdog | State transitions, liveness, next sprint spawn, pause/notify | Coding, broad reasoning, planning |
| Galt PM runner | Sprint orchestration, decomposition, verification, final accountability | Being an unbounded daemon |
| Claude Code | Sprint planning and oppositional review | Primary implementation by default |
| Codex CLI | Bounded implementation missions | Final acceptance |
| Kanban | Durable child task graph/audit trail | Owning PM lifecycle |
| Liveness sentinel | Cross-project anomaly detection | Fixing state destructively |
| Resource ledger | Ownership proof for cleanup | Pattern-based deletion |

### 2.3 Core design changes from old architecture

1. **Control-plane first.** No sprint pipeline until start/stop/watchdog/liveness/state validation pass tests.
2. **PM control cards are records, not dispatcher work.** `ftd-control-plane` remains nonspawnable.
3. **Child worker tasks cannot depend on nonexistent profiles.** Runtime discovers `hermes profile list`; if only `default` exists, either use direct CLI workers or assign child Kanban to `default` only when that is intended.
4. **State transitions are machine-validated.** PM prompt instructions are guidance, not enforcement.
5. **Resource cleanup is ownership-ledger based.** No `docker system prune`, no arbitrary `cleanup_command` string execution.
6. **Stop must stop the real child.** The wrapper and child PIDs are separate; V2 treats both explicitly.
7. **Cron job existence is verified.** Stored cron ID is not truth.
8. **Checkpoint caps are enforced.** Config-only safety keys are not acceptable.
9. **Branch/push policy is enforced.** Do not rely on PM prompt memory.
10. **Spec and runtime must match.** Tests assert the behavior described here.

---

## 3. State Model

### 3.1 External project state file

Path:

```text
~/.hermes/fulltime-dev/projects/<project-id>.json
```

Required schema fields:

```json
{
  "schema_version": 2,
  "project_id": "repo-8hex",
  "repo": "/absolute/repo/path",
  "repo_realpath": "/resolved/repo/path",
  "repo_inode_fingerprint": "optional-platform-specific",
  "board": "ftd-repo-8hex",
  "tenant": "ftd:repo-8hex",
  "state": "ACTIVE",
  "objective": "...",
  "root_task_id": "t_...",
  "active_sprint_task_id": "t_...",
  "sprint_counter": 3,
  "sprints_since_benjamin_checkpoint": 2,
  "active_started_at": "iso",
  "last_benjamin_checkpoint_at": "iso|null",
  "started_at": "iso",
  "updated_at": "iso",
  "last_state_set_at": "iso",
  "active_pm_runner_pid": 123,
  "active_pm_runner_wrapper_pid": 123,
  "active_pm_child_pid": 456,
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
  "worker_health": {
    "claude": {"status": "ok|failed|unknown", "checked_at": "iso", "evidence": "..."},
    "codex": {"status": "ok|failed|unknown", "checked_at": "iso", "evidence": "..."}
  },
  "git": {
    "default_branch": "main",
    "active_branch": "ftd/...",
    "base_head": "sha",
    "last_verified_commit": "sha|null",
    "last_push": {"remote": "origin", "branch": "...", "at": "iso", "result": "..."}
  }
}
```

### 3.2 Valid states

Keep current names for compatibility, but define them sharply:

| State | Meaning | Watchdog behavior |
|---|---|---|
| `OFF` | Disabled | Silent; no dispatch/spawn |
| `STARTING` | Lock-protected initialization/reservation | If stale, error/notify |
| `ACTIVE` | PM runner should be alive | Verify PM liveness, dispatch child work if allowed |
| `READY_NEXT_SPRINT` | Prior PM closeout succeeded; continue silently | Create/spawn next sprint; no child dispatch before spawn |
| `SPAWNING_NEXT_SPRINT` | Transient transition | Complete spawn or recover stale transition |
| `FEATURE_CHECKPOINT_READY_FOR_BENJAMIN` | User-testable bundle ready | Pause watchdog and notify once |
| `CHECKPOINT_READY_FOR_BENJAMIN` | Legacy alias | Normalize or treat as feature checkpoint |
| `APPROVAL_REQUIRED` | Safety gate | Pause and notify |
| `BLOCKED_NEEDS_INPUT` | External blocker | Pause and notify |
| `ERROR` | Control-plane or unrecoverable runtime error | Pause and notify |
| `COMPLETE` | Objective complete | Notify, delete/disable watchdog based on config |
| `STOPPING` | Stop transaction in progress | Terminate child/wrapper, settle to `OFF` |

### 3.3 Transition invariants

`ftd_set_state.py` must reject terminal/continuation transitions unless validators pass, except with an explicit operator override flag.

Validated states:

- `READY_NEXT_SPRINT`
- `FEATURE_CHECKPOINT_READY_FOR_BENJAMIN`
- `CHECKPOINT_READY_FOR_BENJAMIN`
- `APPROVAL_REQUIRED`
- `BLOCKED_NEEDS_INPUT`
- `ERROR`
- `COMPLETE`

Required validation before success/checkpoint/complete:

1. `HANDOFF.md` exists and has required sections.
2. `HANDOFF.md` `Last updated` is after current sprint start.
3. `SPRINTS.md` contains an entry for current sprint.
4. Git status inspected and recorded.
5. Tests/checks recorded or explicitly justified.
6. Dogfood/manual verification recorded or explicitly justified.
7. Current sprint task/card state reconciled.
8. Open child cards are either done, blocked with same blocker/error, or explicitly superseded/obsolete.
9. Resource ledger cleanup/inventory recorded.
10. If push is configured, push result recorded.

Relaxed validation for `ERROR`/`BLOCKED_NEEDS_INPUT`:

- handoff update should be attempted but not required if the failure is exactly that handoff/state is corrupted.
- must record the failing command/reason in state and/or log.

Operator override:

```bash
python3 ~/.hermes/scripts/ftd_set_state.py /repo --state ERROR --override-validation --reason "manual recovery"
```

Override must write `validation_override: true`, caller, timestamp, reason.

---

## 4. Cron / Watchdog Design

### 4.1 Project-scoped watchdogs

One no-agent cron job per active FTD project/worktree.

Generated wrapper remains useful:

```text
~/.hermes/scripts/ftd_watchdog_<project-id>.py
```

Wrapper imports `ftd_watchdog` and calls:

```bash
python3 ~/.hermes/scripts/ftd_watchdog.py --project-id <project-id>
```

### 4.2 `ftd_watchdog.py` behavior

Create new file: `/Users/johngalt/.hermes/scripts/ftd_watchdog.py`.

CLI:

```bash
python3 ~/.hermes/scripts/ftd_watchdog.py [--project-id ID | --repo PATH | --all] [--json]
```

Default behavior:

- if called without arguments, check all projects for compatibility
- for generated wrappers, project-specific mode is required
- empty stdout and exit 0 means healthy/no notification
- non-empty stdout means cron notification
- nonzero exit means cron error notification

Algorithm per project:

```text
acquire project_lock(project_id)
load state
if missing -> emit error only for project-specific invocation; global mode reports missing separately
normalize legacy checkpoint state if needed
case state:
  OFF: silent
  STARTING: if stale -> ERROR; else silent
  ACTIVE:
    verify PM child/wrapper liveness
    if dead -> ERROR + emit
    verify active sprint task is not claimed by generic worker
    dispatch child board tasks with tenant/filter only if PM alive and state still ACTIVE
    silent
  READY_NEXT_SPRINT:
    if checkpoint cap reached -> FEATURE_CHECKPOINT_READY_FOR_BENJAMIN + emit
    set SPAWNING_NEXT_SPRINT
    create/reuse next sprint task idempotently
    spawn PM runner
    set ACTIVE
    silent unless spawn failure
  SPAWNING_NEXT_SPRINT:
    if fresh -> silent
    if stale with task id/no runner -> either spawn existing task or ERROR depending evidence
  FEATURE_CHECKPOINT_READY_FOR_BENJAMIN / APPROVAL_REQUIRED / BLOCKED_NEEDS_INPUT / ERROR / COMPLETE:
    pause or delete this project's watchdog as policy requires
    emit one notification unless already notified for same state hash
  STOPPING:
    perform/verify stop or emit if stale
```

### 4.3 Notification de-duplication

Add state fields:

```json
{
  "last_notification": {
    "state": "ERROR",
    "reason_hash": "sha256",
    "sent_at": "iso"
  }
}
```

Watchdog should not spam identical checkpoint/error notifications every tick.

### 4.4 Cron ID verification

Modify `ensure_project_watchdog()` in `ftd_lib.py`:

- after resume, verify the job exists with `hermes cron list --all` or structured cron CLI output if available.
- if missing, recreate job and update config.
- if paused and desired state requires active, resume and verify enabled.
- store `verified_at`.

Add helper:

```python
def cron_job_exists(job_id: str) -> bool: ...
def ensure_project_watchdog(state, deliver='origin', schedule='every 2m') -> str: ...
def pause_project_watchdog(project_id) -> None: ...
def delete_project_watchdog(project_id) -> None: ...
```

### 4.5 Global liveness sentinel

Create `ftd_liveness_check.py`.

This is separate from project watchdogs and should be suitable for a single ops cron job delivered to `#cron-jobs`.

CLI:

```bash
python3 ~/.hermes/scripts/ftd_liveness_check.py [--json] [--project-id ID]
```

Silent on OK.

Detect:

- missing `ftd_watchdog.py`
- missing generated project watchdog wrapper
- project in `ACTIVE` but wrapper/child PID dead
- project in `ACTIVE` but PM control card claimed by generic worker
- project in `READY_NEXT_SPRINT` older than threshold with watchdog paused/missing
- stale `SPAWNING_NEXT_SPRINT`
- project watchdog cron ID missing/paused when it should be active
- checkpoint/error state still dispatching work
- recent runner log ends in `Traceback`, `KeyboardInterrupt`, or nonzero exit that contradicts success state
- stale monolithic/main-worktree project active while component worktrees superseded it
- child cards running with dead worker PID if Kanban exposes PID

Required test compatibility:

```python
def sprint_task_control_plane_problem(state: dict, task: dict) -> str | None:
    ...
```

Existing test expects a message containing both generic Kanban worker PID and PM runner PID.

---

## 5. Start/Stop/Status Design

### 5.1 `ftd_start.py` rewrite

Preserve CLI, change internals.

Required behavior:

1. Resolve git root.
2. Compute project ID from resolved path.
3. Acquire project lock **before any side-effectful FTD state change**.
4. Load existing state.
5. Refuse live `ACTIVE` based on both wrapper and child PIDs.
6. Refuse pause states unless `--continue-paused`.
7. Write `STARTING` reservation with objective and timestamp before releasing lock for external calls, or keep lock through external calls if acceptable.
8. Ensure repo files exist.
9. Discover available profiles with `hermes profile list` and store them.
10. Ensure/create board/root task idempotently.
11. Ensure project watchdog and verify job exists.
12. Create/reuse sprint task idempotently.
13. If `--no-spawn`, set `READY_NEXT_SPRINT` with existing task id.
14. Else spawn PM runner and set `ACTIVE`.

Race policy:

- Do not create root/sprint tasks outside a state reservation.
- If external calls are outside lock, state must show `STARTING` with `start_token` so a second start refuses instead of duplicating work.
- Any start failure after reservation must transition to `ERROR` with reason.

### 5.2 `ftd_stop.py` rewrite

Preserve CLI.

Required behavior:

1. Resolve repo/project.
2. Acquire lock.
3. Set `STOPPING` with reason.
4. Terminate `active_pm_child_pid` process group first.
5. Terminate `active_pm_runner_wrapper_pid` and `active_pm_runner_pid` as backup.
6. Wait briefly and verify dead.
7. Clear active PID fields.
8. Set `OFF`.
9. Pause/delete watchdog according to flags.
10. Print exactly what was killed/not found.

CLI flags:

```bash
--keep-watchdog
--delete-watchdog
--timeout-seconds 10
--json
```

Default: pause watchdog on `OFF`; delete only on `COMPLETE` or explicit `--delete-watchdog`.

### 5.3 `ftd_status.py` expansion

Preserve CLI; add machine-readable output.

New flags:

```bash
--json
--kanban
--open-only
--active-sprint-only
--liveness
--watchdog
```

Status should show:

- state
- wrapper PID and liveness
- child PID and liveness
- active sprint task
- board/tenant
- watchdog cron id/status/deliver
- sprint counters/checkpoint caps
- last PM exit
- last notification
- worker health summary
- branch/head/dirty summary
- resource ledger summary
- liveness problems if `--liveness`

---

## 6. PM Runner / Wrapper Design

### 6.1 Keep wrapper concept

`ftd_pm_runner_wrapper.py` is useful and should remain. It creates durable exit metadata that the watchdog can inspect.

### 6.2 Required fixes

1. Store and distinguish:
   - wrapper PID
   - child PID
   - process group IDs if available
2. On child exit, reload state and inspect current state.
3. If child exits while state is still `ACTIVE` for the same sprint task, transition to `ERROR` **regardless of exit code**.
4. If state was changed to `READY_NEXT_SPRINT`, checkpoint, blocked, approval, complete, or error by the PM before exit, record metadata and do not override.
5. Remove active child/wrapper PID fields only if they match the current runner. Avoid clearing a newer runner's PID after stale wrapper exit.
6. Avoid unsafe/deprecated `preexec_fn` if possible. If signal ignoring is required, implement with supported process/session handling and clear tests.

### 6.3 PM prompt improvements

Move long PM policy text out of `ftd_lib.py` into a template/helper function if practical.

PM prompt must add:

- profile discovery result
- worker health check requirement
- branch protection requirement
- no direct push to default branch unless explicitly configured
- resource ledger requirement before spawning servers/containers/worktrees
- closeout validator command preview
- exact expected review artifacts:
  - `.fulltime-dev/reviews/sprint-###-claude-plan.md`
  - `.fulltime-dev/reviews/sprint-###-claude-review.md`
  - `.fulltime-dev/dogfood/reports/sprint-###.md`
  - `.fulltime-dev/resources/cleanup-ledger.json`

---

## 7. Claude / Codex Worker Pipeline

### 7.1 Worker health checks

Create `ftd_workers.py`.

Codex health:

```bash
command -v codex
codex --version
codex exec --sandbox read-only -o <tmpfile> 'Say exactly OK. Do not edit files.'
```

Claude health for Galt default must prefer interactive tmux, not print mode:

```bash
tmux new-session -d -s ftd-claude-smoke-<project> -x 160 -y 50 'cd <repo> && claude --permission-mode plan --no-chrome'
# accept trust if needed
# paste: Say exactly: OK. Do not use tools.
# capture pane and verify OK or explicit failure
# exit/kill tmux session
```

Do not treat TUI launch or `claude auth status` as sufficient.

Worker health results must be stored in state and handoff.

### 7.2 Sprint planning

Claude Code planner writes:

```text
.fulltime-dev/reviews/sprint-###-claude-plan.md
```

Planner constraints:

- no code edits during planning
- must read handoff/config/git status/diff/test commands
- must output concise Codex missions with acceptance criteria
- must identify required tests/dogfood
- must name likely files affected
- must name non-goals

### 7.3 Codex implementation

Codex receives bounded mission prompts. PM should prefer one mission per coherent change.

Codex invocation policy:

- inside git repo/worktree only
- `--sandbox workspace-write` for implementation
- use output file with `-o` where supported
- no `--dangerously-bypass-approvals-and-sandbox` unless external disposable sandbox exists and policy permits
- no package installs unless config permits
- no push/merge from Codex; PM owns commit/push

### 7.4 Claude review

Claude Code reviewer writes:

```text
.fulltime-dev/reviews/sprint-###-claude-review.md
```

Reviewer checks:

- spec compliance
- DRY violations
- over-verbose/over-engineered code
- bugs/races/security/privacy
- missing tests
- missing dogfood
- resource leaks
- branch/push risk

Findings must be classified:

- `BLOCKING`
- `SHOULD_FIX`
- `NON_BLOCKING`

Blocking findings create Codex fix missions; Claude re-reviews after fixes.

### 7.5 PM final verification

PM must independently run commands and inspect diff. Worker self-report is not evidence.

Required verification record includes:

```text
command: <exact command>
exit: <code>
summary: <result>
artifact/log: <path if large>
```

---

## 8. Kanban Design

### 8.1 Keep Kanban, but demote it

Kanban is not the PM lifecycle owner. It is the child task ledger and parallel work queue.

Root and sprint PM cards remain durable records but are assigned to `ftd-control-plane` and never dispatched as ordinary workers.

### 8.2 Profile/assignee strategy

Current default config hardcodes nonexistent profiles. V2 must not.

Default generated config:

```yaml
kanban:
  pm_assignee: "ftd-control-plane"
  child_dispatch_mode: "direct_cli" # direct_cli | profile | mixed
  default_profile_assignee: "default"
  implementer_assignee: null
  reviewer_assignee: null
  researcher_assignee: null
  ops_assignee: null
  max_dispatch_per_tick: 4
```

Policy:

- If only `default` exists, PM should prefer direct Claude/Codex CLI invocation for planner/reviewer/implementer rather than inventing profile workers.
- If specialist profiles exist, store them in state and use only exact discovered names.
- If config names unavailable profiles, start should warn/error based on `strict_profile_validation`.

### 8.3 Dispatch restrictions

`dispatch_board(state)` must dispatch only when:

- state is `ACTIVE`
- PM runner is alive
- board matches state board
- tenant filter is applied if Hermes CLI supports it
- control-plane cards are not dispatchable

If Hermes CLI dispatch lacks tenant/assignee filtering, V2 must either:

- add support in Hermes Kanban dispatch, or
- avoid generic `dispatch` and let PM invoke direct CLI workers, or
- ensure board is strictly project-private and only child cards are dispatchable.

Do not pretend tenant isolation exists if the CLI does not enforce it.

### 8.4 Board reconciliation

V2 validator must query board state. Success transitions require no contradictory open sprint cards.

If board query is unavailable, validation fails closed unless operator override is provided.

---

## 9. Git Policy

Create `ftd_git.py`.

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

Rules:

1. FTD must not implement directly on default branch unless `require_branch_isolation: false` is explicit.
2. If repo is on default branch at start, create/switch to `ftd/<project>/<objective-slug>` before implementation.
3. PM may commit verified work if configured.
4. PM may push branch if both `autonomy.allow_push` and `git.push_if_allow_push` are true.
5. PM must never force-push unless explicit config and user approval; default is hard deny.
6. PM must not merge to default branch without explicit policy/approval.
7. Handoff records branch/base/HEAD/commit/push.

Validator must block success closeout if:

- dirty changes exist and `commit_verified_sprints` is true but no commit recorded
- work is on default branch while branch isolation is required
- push was attempted to default branch while `push_branch_only` is true

---

## 10. Resource / Environment Cleanup

Create `ftd_resources.py` and repo-local resource state:

```text
.fulltime-dev/resources/
  owned-processes.json
  owned-containers.json
  owned-ports.json
  owned-worktrees.json
  owned-databases.json
  owned-tempdirs.json
  cleanup-ledger.json
  ambiguous-resources.md
```

### 10.1 Principle

Only clean positively identified FTD-owned resources.

Do not use arbitrary shell cleanup strings from ledger.

### 10.2 Ledger schema

```json
{
  "schema_version": 1,
  "resources": [
    {
      "kind": "process|container|worktree|database|tmpdir|port",
      "id": "stable-id",
      "project_id": "...",
      "sprint": 3,
      "created_at": "iso",
      "ttl_seconds": 7200,
      "safe_cleanup": true,
      "cleanup_method": "kill_process_group|docker_stop_label|git_worktree_remove|rm_repo_tmpdir|drop_named_test_db",
      "metadata": {}
    }
  ]
}
```

`cleanup_method` is an enum implemented by code. No free-form command strings.

### 10.3 Cleanup gate

Every sprint closeout must run inventory:

- known owned resources cleaned or explicitly retained with reason
- ambiguous resources reported but not cleaned
- cleanup result written to handoff/sprint record

### 10.4 What V2 may auto-clean

- process group launched by FTD helper and recorded
- container with exact FTD label and project ID
- worktree path recorded and under approved base directory
- temp dir under `.fulltime-dev/tmp/`
- test DB with exact generated name and recorded creation

### 10.5 What V2 must not auto-clean

- unknown Docker containers
- arbitrary processes on a port
- files outside repo or outside `.fulltime-dev/tmp/`
- global caches
- DBs without exact FTD ownership marker
- worktrees not recorded by ledger

---

## 11. Repo-local Files

V2 `.fulltime-dev/` structure:

```text
.fulltime-dev/
  config.yaml
  HANDOFF.md
  SPRINTS.md
  CHECKPOINT.md
  RUNBOOK.md
  state/
    local-status.json
  resources/
    owned-processes.json
    owned-containers.json
    owned-ports.json
    owned-worktrees.json
    owned-databases.json
    owned-tempdirs.json
    cleanup-ledger.json
    ambiguous-resources.md
  reviews/
    sprint-###-claude-plan.md
    sprint-###-claude-review.md
    sprint-###-pm-final-review.md
  dogfood/
    reports/sprint-###.md
    screenshots/
  logs/
  archive/
```

### 11.1 Handoff contract

Keep handoff compact. It is not a transcript.

Required sections:

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

Validator checks headings and timestamps.

---

## 12. Checkpoint Policy

Benjamin should not be interrupted after every sprint.

Machine-enforced caps:

```yaml
checkpoint_policy:
  notify_benjamin_on_each_sprint: false
  require_feature_bundle_for_benjamin_checkpoint: true
  min_completed_feature_count: 2
  checkpoint_on_user_facing_bundle: true
  checkpoint_on_safety_boundary: true
  checkpoint_on_blocker: true
  checkpoint_on_error: true
  max_sprints_without_benjamin_review: 6
  max_active_hours_without_checkpoint: 12
```

Enforcement:

- `sprints_since_benjamin_checkpoint` increments on successful `READY_NEXT_SPRINT` closeout or on watchdog rollover, not both.
- resets when state becomes `FEATURE_CHECKPOINT_READY_FOR_BENJAMIN` or Benjamin explicitly resumes after checkpoint.
- watchdog refuses further rollover if cap reached; writes checkpoint state and notifies.
- active-hour cap computed from `active_started_at` or last checkpoint timestamp.

---

## 13. Security / Safety Gates

Even in YOLO mode, V2 must gate:

- deletion outside FTD-owned resources
- package installs unless `allow_install: true`
- deploys unless `allow_deploy: true`
- public network exposure unless explicitly allowed
- cloud processing of sensitive identifiers/CJIS
- force push
- merge to default branch
- modifying global Hermes secrets/config files unless explicitly part of objective and approved

`--yolo` means fewer prompts for routine project work, not absence of policy.

---

## 14. Tests Required Before Runtime Use

Add/expand tests under the FTD source-backed test location, mirrored to runtime tests if needed.

Minimum tests:

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
14. `test_create_next_sprint_idempotent_no_counter_advance_on_kanban_failure`
15. `test_set_state_blocks_ready_without_fresh_handoff`
16. `test_set_state_blocks_success_with_unreconciled_cards`
17. `test_set_state_allows_error_with_minimal_failure_evidence`
18. `test_profile_discovery_rejects_missing_configured_assignees`
19. `test_git_policy_blocks_default_branch_when_isolation_required`
20. `test_git_policy_blocks_force_push`
21. `test_resource_cleanup_rejects_arbitrary_cleanup_command`
22. `test_resource_cleanup_only_cleans_owned_tmpdir`
23. `test_status_json_contains_watchdog_liveness_git_resource_fields`

Run:

```bash
python3 -m py_compile ~/.hermes/scripts/ftd_*.py
python3 -m pytest ~/.hermes/scripts/tests/test_ftd_control_plane.py -q
```

If source-backed under repo:

```bash
python3 -m pytest local/ftd/tests/ -q
```

---

## 15. Implementation Order for Codex

### Phase A — Source-backed structure and tests

1. Create `local/ftd/` source-backed copy structure or explicitly document why not.
2. Copy existing runtime files into source-backed location.
3. Add tests for missing liveness import, stop PID bug, wrapper exit-0 bug.
4. Do not change runtime behavior yet except if needed for importability in tests.

### Phase B — Control-plane floor

1. Create `ftd_liveness_check.py` minimal but tested.
2. Create `ftd_watchdog.py` minimal state handler.
3. Fix `ftd_stop.py` child/wrapper termination.
4. Fix wrapper active-exit behavior.
5. Fix start reservation/locking.
6. Fix cron verification/recreation.
7. Make tests pass.

### Phase C — Validation gates

1. Create `ftd_validate.py`.
2. Add handoff validator.
3. Add board reconciliation validator.
4. Add git policy validator.
5. Add cleanup/inventory validator stub.
6. Wire into `ftd_set_state.py`.

### Phase D — Worker pipeline

1. Create `ftd_workers.py`.
2. Add Codex smoke test helper.
3. Add Claude interactive tmux smoke test helper.
4. Add planner/reviewer output file contract.
5. Update PM prompt.

### Phase E — Resource ledger and cleanup

1. Create `ftd_resources.py`.
2. Add resource register/list/cleanup helpers.
3. Add no-free-form-cleanup enforcement.
4. Add closeout cleanup report.

### Phase F — Observability/status

1. Expand `ftd_status.py`.
2. Add JSON output.
3. Add liveness integration.
4. Add global liveness cron setup instructions.

### Phase G — Canary

Use a disposable local git repo. Verify:

1. start creates exactly one PM runner
2. stop kills child and wrapper
3. `READY_NEXT_SPRINT` rolls to new sprint silently
4. checkpoint pauses/notifies once
5. dead child becomes `ERROR`
6. liveness sentinel silent on healthy, noisy on constructed unhealthy cases
7. no unknown resources cleaned

---

## 16. Acceptance Criteria

V2 is not ready until all are true:

1. Tests pass.
2. `ftd_watchdog.py` and `ftd_liveness_check.py` exist and run.
3. `ftd_stop.py` kills the actual Hermes child process.
4. Zero-exit while still `ACTIVE` becomes `ERROR`.
5. Stale cron IDs are recreated or reported.
6. Start cannot create duplicate active PM runners under concurrent calls.
7. Terminal state transitions validate handoff/board/git/resource evidence.
8. Default config does not reference nonexistent profiles.
9. Branch isolation is enforced by code, not prompt text.
10. Cleanup cannot execute arbitrary shell strings.
11. Status JSON exposes enough information for watchdog/liveness/debugging.
12. Claude/Codex worker health is recorded before use.
13. Benjamin is notified only on checkpoint/block/error/approval/complete or caps, not routine internal rollovers.
14. A disposable canary repo completes start -> sprint closeout -> rollover -> stop.

---

## 17. Explicit File-by-File Worklist

### Modify existing

- `/Users/johngalt/.hermes/scripts/ftd_lib.py`
  - keep: project ID, board slug, locks, JSON load/save, process helpers, repo file creation
  - modify: default config, cron verification, create_next_sprint atomicity, dispatch restriction
  - move out: validation, resources, worker health, git policy if feasible

- `/Users/johngalt/.hermes/scripts/ftd_start.py`
  - preserve CLI
  - rewrite start transaction/reservation/profile discovery

- `/Users/johngalt/.hermes/scripts/ftd_stop.py`
  - preserve CLI
  - fix child/wrapper process termination

- `/Users/johngalt/.hermes/scripts/ftd_status.py`
  - preserve CLI
  - add JSON/liveness/watchdog/git/resource fields

- `/Users/johngalt/.hermes/scripts/ftd_set_state.py`
  - preserve CLI
  - add validation gates and override flag

- `/Users/johngalt/.hermes/scripts/ftd_pm_runner_wrapper.py`
  - preserve wrapper concept
  - fix exit-0-active behavior and stale PID clearing

- `/Users/johngalt/.hermes/scripts/tests/test_ftd_control_plane.py`
  - preserve existing tests
  - expand with V2 tests above

### Create new

- `/Users/johngalt/.hermes/scripts/ftd_watchdog.py`
- `/Users/johngalt/.hermes/scripts/ftd_liveness_check.py`
- `/Users/johngalt/.hermes/scripts/ftd_validate.py`
- `/Users/johngalt/.hermes/scripts/ftd_resources.py`
- `/Users/johngalt/.hermes/scripts/ftd_workers.py`
- `/Users/johngalt/.hermes/scripts/ftd_git.py`
- `/Users/johngalt/.hermes/scripts/ftd_board.py` if Kanban code grows past simple wrappers

### Prefer creating source-backed copies

- `/Users/johngalt/.hermes/hermes-agent/local/ftd/...`

If source-backed copies are created, runtime wrappers under `~/.hermes/scripts/` should be generated/synced, not hand-edited independently.

---

## 18. Main Risks

1. **Overengineering into a second agent framework.** Mitigation: cron stays deterministic; PM remains Hermes; worker CLIs stay external.
2. **Prompt text pretending to be enforcement.** Mitigation: validators and tests.
3. **Source/runtime drift.** Mitigation: source-backed copies plus install/sync script.
4. **Claude interactive orchestration brittleness.** Mitigation: health checks, fallback to Codex/direct PM review, record failure.
5. **Kanban dispatcher conflicts.** Mitigation: `ftd-control-plane`, project-private boards, dispatch only in `ACTIVE`.
6. **Cleanup destroying unrelated data.** Mitigation: owned-resource ledger and enum cleanup methods only.
7. **Infinite self-improvement loop.** Mitigation: checkpoint caps and canary/rollback required for FTD self-modification.
8. **Cost runaway.** Mitigation: sprint caps, PM turn caps, checkpoint caps, worker health before wasted delegation.
9. **Dirty repo ambiguity.** Mitigation: preflight and branch isolation before implementation.
10. **Old state incompatibility.** Mitigation: schema migration and explicit legacy handling.

---

## 19. Final Recommendation in This Draft

Implement V2 as a disciplined replacement/refactor:

- keep CLI entrypoint names for operator compatibility
- keep useful state/lock/Kanban/control-lane conventions
- create missing watchdog/liveness/validator/resource/worker/git modules
- move important runtime code into source-backed files before large changes
- make the control plane pass tests before touching Claude/Codex sprint automation

The correct first Codex implementation task is **not** “add Claude planner and Codex worker.” It is:

> Build the FTD V2 control-plane floor: liveness module, watchdog module, correct stop semantics, wrapper active-exit erroring, start reservation, cron verification, profile discovery guard, and tests.
