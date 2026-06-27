# Autonomous Full-Time Developer Plan

**Date:** 2026-05-14
**Author:** John Galt / Hermes Agent
**Status:** Research and planning draft. No implementation performed.
**Scope:** Design the next-generation Full-Time Developer (FTD) system for long-running autonomous software development using Hermes, cron, Kanban, Claude Code CLI, and Codex CLI.

---

## 1. Executive Summary

The Full-Time Developer should be a **state-machine supervised, sprint-scoped autonomous development system**.

Core architecture:

```text
Explicit user start
  -> repo-local .fulltime-dev/ state + project-scoped Kanban board
  -> deterministic no-agent cron watchdog
  -> fresh Galt PM sprint runner per sprint/context window
  -> Claude Code plans sprint
  -> Codex implements bounded missions
  -> Claude Code reviews oppositional/spec-quality/security gaps
  -> Galt PM verifies, dogfoods, cleans environment, commits/pushes if allowed
  -> handoff written
  -> watchdog silently rolls into next sprint or pauses/notifies for checkpoint/block/error
```

The main design choice is to use **cron as supervisor/liveness/state-transition mechanism**, not as the developer and not as the general task dispatcher. Kanban remains the durable work queue for child implementation/review/dogfood tasks. The FTD PM control plane must be isolated from the generic Kanban dispatcher to avoid the previous split-brain conflict.

The immediate first implementation work should be **control-plane repair and hardening**, because the current local install contains `ftd_start.py`, `ftd_lib.py`, `ftd_pm_runner_wrapper.py`, `ftd_set_state.py`, and `ftd_status.py`, but lacks `ftd_watchdog.py` and `ftd_liveness_check.py` even though existing code/tests reference them.

---

## 2. Goals

1. Run autonomously for long periods without requiring Benjamin to inspect every sprint.
2. Preserve fresh context per sprint while maintaining continuity through durable handoffs.
3. Use Claude Code CLI for high-level sprint planning and oppositional review.
4. Use Codex CLI as primary bounded implementer.
5. Keep Galt PM as accountable orchestrator, not a passive relay.
6. Require independent review and actual verification before accepting worker claims.
7. Keep repo and machine clean: no runaway containers, stale DBs, orphan processes, port conflicts, or accumulating test artifacts.
8. Commit/push verified work routinely when repo FTD config allows it.
9. Notify Benjamin only for substantial feature checkpoints, approval gates, blockers, errors, or completion.
10. Allow FTD to improve itself safely, with strong control-plane tests and rollback discipline.

---

## 3. Non-Goals

1. Do not create an always-on uncontrolled agent that mutates repos without explicit start/resume.
2. Do not let cron become a developer. Cron should remain deterministic Python.
3. Do not let generic Kanban dispatcher claim FTD PM control cards.
4. Do not notify Benjamin after every sprint.
5. Do not auto-delete ambiguous resources or files.
6. Do not use YOLO as a reason to bypass explicit safety boundaries.
7. Do not treat worker self-reports as evidence.
8. Do not merge to `main` by default merely because a branch was pushed.

---

## 4. Current Local Findings

### 4.1 Verified files present

Existing local FTD scripts under `~/.hermes/scripts/`:

- `ftd_start.py`
- `ftd_lib.py`
- `ftd_pm_runner_wrapper.py`
- `ftd_set_state.py`
- `ftd_stop.py`
- `ftd_status.py`
- `tests/test_ftd_control_plane.py`

### 4.2 Verified files missing

Missing but referenced:

- `ftd_watchdog.py`
- `ftd_liveness_check.py`

This is a control-plane correctness gap. `ftd_lib.py` writes project watchdog wrappers that import `ftd_watchdog`, and tests import `ftd_liveness_check`; therefore project-scoped cron rollover/liveness cannot work as designed until these are restored or implemented.

### 4.3 Existing useful design already present

`ftd_lib.py` already contains important structure:

- project IDs from repo path
- project locks
- repo-local `.fulltime-dev/` bootstrapping
- external project state under `~/.hermes/fulltime-dev/projects/`
- project-scoped watchdog creation wrappers
- reserved `FTD_CONTROL_ASSIGNEE = "ftd-control-plane"`
- `READY_NEXT_SPRINT` silent rollover state
- `FEATURE_CHECKPOINT_READY_FOR_BENJAMIN` checkpoint state
- PM runner prompt with explicit closeout requirements
- wrapper that records PM child exit metadata

### 4.4 Current profile inventory

Current `hermes profile list` shows only:

- `default`

Existing default repo config templates mention `galtcode`, `galtresearch`, and `galtops`. If those profiles do not exist, cards assigned to them will stall. The FTD PM must discover available profiles at runtime or avoid assigning generic Kanban worker cards to nonexistent profiles.

---

## 5. Key Architectural Decision: Cron vs Kanban Dispatcher

### 5.1 Decision

Use **project-scoped no-agent cron watchdogs** for FTD supervision and PM lifecycle. Use **Kanban dispatcher** only for child work tasks during `ACTIVE` state.

### 5.2 Rationale

Cron is appropriate for deterministic liveness/state transitions:

- check project state
- detect dead runner
- create next sprint
- spawn fresh PM context
- pause and notify on checkpoint/block/error
- run silently when nothing changes

Kanban is appropriate for durable task graph/audit trail:

- implementation tasks
- review tasks
- dogfood tasks
- cleanup tasks
- research/spike tasks
- fix-followup tasks

### 5.3 Anti-pattern to avoid

```text
PM sprint task assigned to a real Hermes profile
AND watchdog also spawns PM runner
```

This causes split-brain. A generic worker can claim the PM sprint card and start behaving like the PM while the wrapper/watchdog also controls a PM runner.

### 5.4 Correct pattern

```text
FTD root/sprint PM cards -> assignee ftd-control-plane (reserved, non-spawnable)
watchdog/wrapper -> owns PM runner process lifecycle
Galt PM runner -> creates child cards assigned to actual workers/profiles
Kanban dispatcher -> only claims child cards, never PM/root control cards
```

---

## 6. Proposed System Components

### 6.1 Repo-local `.fulltime-dev/`

Recommended structure:

```text
.fulltime-dev/
  config.yaml
  HANDOFF.md
  SPRINTS.md
  CHECKPOINT.md
  RUNBOOK.md
  state/
    owned-processes.json
    owned-containers.json
    owned-ports.json
    owned-databases.json
    owned-worktrees.json
    cleanup-ledger.json
  reviews/
    sprint-###-claude-plan.md
    sprint-###-claude-review.md
    sprint-###-codex-review.md
  dogfood/
    reports/
    screenshots/
  logs/
  archive/
```

### 6.2 External FTD state

```text
~/.hermes/fulltime-dev/
  projects/<project-id>.json
  logs/<project-id>-sprint-###.log
  logs/<project-id>-sprint-###-prompt.md
  locks/<project-id>.lock
  config.json
```

### 6.3 Kanban board

Each FTD project uses a project-scoped board:

```text
board: ftd-<repo>-<hash>
tenant: ftd:<repo>-<hash>
```

Control cards use `ftd-control-plane`; child cards use actual available profiles or explicit CLI workers run by the PM.

### 6.4 Cron watchdog

Each project gets a no-agent cron job:

```text
schedule: every 2m
script: ~/.hermes/scripts/ftd_watchdog_<project-id>.py
no_agent: true
empty stdout: silent
non-empty stdout: delivered notification
```

The wrapper imports `ftd_watchdog` and passes `--project-id <id>`.

### 6.5 PM runner

A fresh Hermes PM runner is spawned per sprint:

```text
hermes [--yolo] chat -Q --source ftd --max-turns <bounded> --skills <core FTD skills> --toolsets terminal,file,delegation,browser,web,skills,session_search,todo,cronjob,messaging -q <PM prompt>
```

The PM runner is not trusted to run forever. It must close out with a durable state transition and handoff.

---

## 7. State Machine

### 7.1 Valid states

```text
OFF
STARTING
ACTIVE
READY_NEXT_SPRINT
SPAWNING_NEXT_SPRINT
FEATURE_CHECKPOINT_READY_FOR_BENJAMIN
CHECKPOINT_READY_FOR_BENJAMIN (legacy alias)
APPROVAL_REQUIRED
BLOCKED_NEEDS_INPUT
ERROR
COMPLETE
STOPPING
```

### 7.2 State semantics

| State | Meaning | Watchdog behavior |
|---|---|---|
| OFF | FTD disabled | silent |
| STARTING | initialization in progress | validate/advance or report stale |
| ACTIVE | PM runner should be alive | liveness check, dispatch child board tasks |
| READY_NEXT_SPRINT | prior sprint cleanly ended; continue silently | spawn next PM runner |
| SPAWNING_NEXT_SPRINT | transient lock-protected state | avoid duplicate spawn; recover if stale |
| FEATURE_CHECKPOINT_READY_FOR_BENJAMIN | substantial bundle ready | pause, notify |
| APPROVAL_REQUIRED | safety boundary | pause, notify |
| BLOCKED_NEEDS_INPUT | genuine external blocker | pause, notify |
| ERROR | unhealthy/error | pause, notify |
| COMPLETE | objective complete | pause, notify |
| STOPPING | shutdown in progress | terminate runner, settle to OFF |

### 7.3 State transition rule

No PM runner may set a terminal/continuation state until:

1. Handoff is updated.
2. Sprint log is updated.
3. Relevant reviews/dogfood reports are written.
4. Git state is inspected.
5. Relevant tests/checks are run or skipped with explicit evidence.
6. All sprint-related Kanban cards are reconciled.
7. Environment cleanup has run or been skipped with reason.

---

## 8. Sprint Lifecycle

### 8.1 Start/resume

Start requires explicit user approval:

```bash
python3 ~/.hermes/scripts/ftd_start.py /path/to/repo \
  --objective "..." \
  --watchdog-deliver 'origin or explicit Discord thread' \
  --allow-push \
  --yolo
```

If paused in checkpoint/block/error/approval state, resume requires:

```bash
python3 ~/.hermes/scripts/ftd_start.py /path/to/repo --continue-paused --objective "continue from handoff"
```

### 8.2 PM preflight

Every sprint starts with:

```bash
git status --short --branch --untracked-files=all
git log --oneline -5
python3 ~/.hermes/scripts/ftd_status.py /path/to/repo --kanban
hermes kanban --board <board> list --tenant <tenant>
```

Then read:

- `.fulltime-dev/config.yaml`
- `.fulltime-dev/HANDOFF.md`
- `.fulltime-dev/CHECKPOINT.md` if present
- recent review/dogfood reports relevant to current objective

If dirty state changes safety or scope, the first sprint task must reconcile it.

### 8.3 Claude planning

Claude Code CLI is the planner. It should receive:

- current handoff
- objective
- repo status/diff summary
- relevant files/code map
- existing tests/commands
- constraints from config
- instruction to produce concise, specific sprint plan

Expected output:

```text
.fulltime-dev/reviews/sprint-###-claude-plan.md
```

Planner must include:

- sprint goal
- exact Codex implementation missions
- acceptance criteria
- tests/checks to run
- dogfood required
- files likely affected
- explicit non-goals
- checkpoint recommendation

### 8.4 Codex implementation

Codex is primary implementer. Each mission should be narrow:

```text
Mission: one sentence
Context: repo, branch, files, relevant prior findings
Scope: allowed files/dirs
Forbidden: no delete/install/push/network/public exposure unless config allows
Tests: exact commands expected
Deliverables: changes, tests run, risks
Failure behavior: block and report, do not improvise destructively
```

Use isolated worktrees for parallel/conflicting changes. Use main repo only for serial low-conflict work.

### 8.5 Claude review

Claude Code performs oppositional review after Codex implementation:

- compare implementation to sprint plan
- identify bugs/regressions
- identify DRY violations
- identify over-verbose/over-engineered parts
- identify missing tests/dogfood
- identify security/privacy/resource risks
- recommend blocking vs non-blocking fixes

Review output:

```text
.fulltime-dev/reviews/sprint-###-claude-review.md
```

If blocking findings exist, PM creates Codex fix mission, then Claude re-reviews.

### 8.6 PM verification

PM independently verifies:

```bash
git status --short --untracked-files=all
git diff --stat
git diff
<relevant tests/lint/typecheck/build>
<manual dogfood where feasible>
```

Worker claims are not proof. Verification commands and results must be recorded.

### 8.7 Closeout

PM closeout:

1. Reconcile sprint Kanban cards.
2. Run environment cleanup/inventory.
3. Update `HANDOFF.md`.
4. Append `SPRINTS.md`.
5. Write/refresh `CHECKPOINT.md` if notifying Benjamin.
6. Commit verified work if configured.
7. Push if configured.
8. Set state.

---

## 9. Handoff Contract

`HANDOFF.md` should stay compact and structured. It is not a transcript.

Recommended schema:

```markdown
# Full-Time Developer Handoff

## Project
- Repo:
- Branch:
- HEAD:
- Board:
- Tenant:
- Sprint:
- Last updated:

## Current objective
One concise paragraph.

## Current state
State and why.

## Completed since last Benjamin checkpoint
Feature/fix bullets only.

## Current implementation facts
Architecture decisions, invariants, touched areas.

## Verification history
Exact command -> result. Dogfood action -> result. Review report links.

## Open risks
Precise risks only.

## Deferred work
Itemized with reason.

## Environment/resource state
Owned resources active or cleaned.

## Next sprint recommendation
Specific enough for a fresh PM context.

## Resume instructions
Exact command/state/board/thread.
```

Add a handoff validator that refuses terminal state if required fields are missing or stale.

---

## 10. Checkpoint Policy

Benjamin should not be notified after every sprint.

Notify only when:

1. A substantial user-testable feature bundle is ready.
2. Final review is needed.
3. Install/deploy/public exposure/deletion/sensitive-cloud approval is required.
4. Genuine ambiguity blocks progress.
5. Control-plane/worker error requires intervention.
6. Objective is complete.

Recommended defaults:

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

The max sprint/hour caps are safety fuses against infinite wrong-direction autonomy, not routine interruption triggers.

---

## 11. Git Policy

Recommended config:

```yaml
git:
  commit_verified_sprints: true
  push_if_allow_push: true
  push_branch_only: true
  never_force_push: true
  merge_to_main: false
```

Rules:

1. Commit after verified sprint if configured.
2. Push branch routinely if `allow_push: true` and `push_if_allow_push: true`.
3. Do not force-push.
4. Do not merge to main without explicit policy/approval.
5. Record branch, base, commits, and verification in handoff.
6. If no CI exists, local verification is mandatory and must be recorded.

---

## 12. Permissions and Safety

### 12.1 YOLO semantics

FTD PM may run with `hermes --yolo` for ordinary approved project work. This should reduce interruptions, not erase boundaries.

Still gated:

- deletion of non-FTD-owned files/resources
- package installs unless `allow_install: true`
- deploys unless `allow_deploy: true`
- public network exposure unless explicitly allowed
- sensitive identifiers/cloud processing
- force-push/merge-to-main unless explicitly allowed

### 12.2 Recommended autonomy config

```yaml
autonomy:
  allow_push: true
  allow_install: false
  allow_public_network: false
  allow_deploy: false
  allow_issue_creation: false
  allow_pr_creation: false

workflow:
  yolo_runner: true
  max_pm_turns: 120
  use_kanban: true

web:
  bind_host: "127.0.0.1"
  allow_public_exposure: false
```

---

## 13. Environment Cleanup Design

### 13.1 Principle

Cleanup must be **ownership-based**, not pattern-based.

Bad:

```bash
docker system prune
kill process on port 8000
rm -rf tmp-looking-directory
```

Good:

```text
resource was created by FTD project/sprint, recorded in ledger, has safe_cleanup=true, cleanup command is scoped
```

### 13.2 Owned resource ledger

When FTD starts a resource, record:

```json
{
  "kind": "process|container|database|port|worktree|tmpdir",
  "id": "...",
  "created_by": "ftd:<project-id>:sprint-###",
  "purpose": "...",
  "safe_cleanup": true,
  "cleanup_command": "...",
  "created_at": "...",
  "ttl": "..."
}
```

### 13.3 Cleanup subagent behavior

1. Inventory processes/ports/containers/worktrees/test DBs/temp dirs.
2. Compare to FTD ledger.
3. Clean only positively identified FTD-owned safe resources.
4. Report ambiguous resources; do not clean them.
5. Record cleanup results in `cleanup-ledger.json`.

### 13.4 Automatically cleanable examples

- process PID launched and recorded by current FTD sprint
- container with exact FTD project label
- DB named with exact FTD project/sprint prefix
- temp dir under `.fulltime-dev/tmp/`
- worktree recorded in `owned-worktrees.json`

### 13.5 Not automatically cleanable

- unknown Docker containers
- unknown processes merely because they use a port
- untracked repo files without review
- databases not created by FTD
- global caches
- files outside declared FTD-owned paths

---

## 14. Observability

Minimum operator commands:

```bash
python3 ~/.hermes/scripts/ftd_status.py /repo --kanban
hermes kanban boards list
hermes kanban --board <board> list
tail -f ~/.hermes/fulltime-dev/logs/<project>-sprint-###.log
```

Add status visibility for:

- project state
- active PM wrapper PID and child PID
- runner log path
- active sprint task
- watchdog cron job ID/status
- board slug
- last sprint summary
- last commit/push
- last verification
- last cleanup
- next recommended sprint

Dashboard should expose FTD board and state without binding publicly. Default localhost only.

---

## 15. Liveness Sentinel

Implement `ftd_liveness_check.py` as silent-on-OK no-agent sentinel.

Flag problems:

- `ACTIVE` but PM runner PID dead
- `ACTIVE` but PM control card claimed by generic worker
- `READY_NEXT_SPRINT` but watchdog paused/missing
- generated watchdog wrapper imports missing `ftd_watchdog`
- stale `SPAWNING_NEXT_SPRINT`
- wrapper exited nonzero
- stale lock or state update beyond TTL
- repeated Claude/Codex auth failure
- child board cards stuck in `running` with dead worker PID
- checkpoint/error states still dispatching work

This should be cron-runnable and safe to deliver only on non-empty output.

---

## 16. Worker Health Checks

Before assigning real work:

### 16.1 Claude Code

Do not rely on TUI opening or `claude auth status` alone. Run a real minimal prompt from the same environment, preferably interactive tmux on Galt:

```text
Say exactly: OK. Do not use tools.
```

If prompt fails, mark Claude unavailable and route planning/review to alternate worker or block if Claude is required.

### 16.2 Codex CLI

Verify:

```bash
command -v codex
codex --version
codex exec --sandbox read-only --output-last-message /tmp/codex-smoke.md 'Say OK and do not edit files.'
```

If Codex cannot run in repo/auth context, block implementation rather than pretending worker completed.

---

## 17. Edge Cases and Mitigations

### 17.1 Duplicate PM runners

Cause: overlapping cron/manual starts.

Mitigation:

- project lock
- `SPAWNING_NEXT_SPRINT`
- live PID checks
- wrapper exit metadata
- idempotency keys

### 17.2 PM/Kanban split-brain

Cause: sprint task assigned to real profile.

Mitigation:

- `ftd-control-plane`
- dispatcher skip non-profile lane
- liveness sentinel flags generic worker claim
- migration/recovery tool for old boards

### 17.3 Stale `READY_NEXT_SPRINT`

Cause: paused/missing watchdog.

Mitigation:

- liveness sentinel
- status reports watchdog job
- watchdog ensure/resume idempotent

### 17.4 Context degradation

Cause: long PM runner/context bloat.

Mitigation:

- fresh PM context per sprint
- compact handoff
- archive logs
- PM max-turn cap
- workers get focused mission orders

### 17.5 Worker auth failure

Cause: Claude/Codex logged in superficially but cannot run actual prompts.

Mitigation:

- real smoke prompts
- fallback routing
- record in handoff
- block if required worker unavailable

### 17.6 Destructive cleanup

Cause: pattern-based cleanup.

Mitigation:

- ownership ledger
- cleanup allowlist
- ambiguous resource report only
- approval gates

### 17.7 Infinite self-improvement loop

Cause: FTD modifying itself endlessly.

Mitigation:

- self-improvement is explicit sprint class
- acceptance criteria required
- control-plane tests required
- canary start/stop required
- rollback plan required

### 17.8 No CI pipeline

Cause: local-only independent development.

Mitigation:

- local verification contract
- branch push with verification summary
- no merge-to-main by default

### 17.9 Missing worker profiles

Cause: config references nonexistent profiles.

Mitigation:

- discover profiles at sprint start
- use available profiles only
- use direct CLI workers if no specialist profiles exist
- block/report if durable child Kanban cannot dispatch

---

## 18. Implementation Phases

### Phase 0: Control-plane repair

Goal: make FTD safe to run.

Tasks:

1. Implement/restore `ftd_watchdog.py`.
2. Implement/restore `ftd_liveness_check.py`.
3. Expand `test_ftd_control_plane.py`.
4. Add watchdog status to `ftd_status.py`.
5. Verify generated watchdog wrapper imports and runs.
6. Verify no duplicate PM runners.
7. Verify checkpoint/error pause semantics.
8. Verify `ACTIVE` with dead PM becomes `ERROR` or reported loudly.
9. Verify generic dispatcher cannot claim PM sprint cards.

Acceptance:

```bash
python -m pytest ~/.hermes/scripts/tests/test_ftd_control_plane.py -q
python3 ~/.hermes/scripts/ftd_liveness_check.py
python3 ~/.hermes/scripts/ftd_status.py
```

### Phase 1: Handoff discipline

Tasks:

1. Tighten `HANDOFF.md` schema.
2. Add handoff validator.
3. Add sprint log append helper.
4. Add checkpoint writer.
5. Add closeout checklist enforcement.
6. Refuse terminal state if board unreconciled.

### Phase 2: Claude-plan / Codex-implement / Claude-review pipeline

Tasks:

1. Add planner mission template.
2. Add Codex implementation mission template.
3. Add Claude review mission template.
4. Add worker health smoke tests.
5. Add output capture paths.
6. Add review-loop state.
7. Add final PM verification gate.

### Phase 3: Environment hygiene

Tasks:

1. Add resource ledger files.
2. Add helpers for tracked process/service/container creation.
3. Add cleanup/inventory card.
4. Add safe cleanup allowlist.
5. Add ambiguous resource report.
6. Add cleanup closeout gate.

### Phase 4: Observability/status

Tasks:

1. Add FTD JSON status output.
2. Surface active sprint/runner/log/watchdog/checkpoint.
3. Link board, handoff, reviews, dogfood reports.
4. Add liveness sentinel cron.
5. Improve Discord notification formatting.

### Phase 5: Self-improvement mode

Tasks:

1. Add explicit `ftd-system` objective mode.
2. Require control-plane tests for any FTD script changes.
3. Require canary start/stop test.
4. Require rollback plan.
5. Disallow mutation of running control-plane scripts without restart/liveness verification.

---

## 19. Open Design Questions

1. Should FTD create missing specialist profiles (`galtcode`, `galtresearch`, `galtops`) or avoid profiles and use direct CLI workers from the PM runner?
2. Should project watchdogs run every 1 minute, 2 minutes, or 5 minutes? Faster means better responsiveness, more scheduler noise.
3. Should branch naming be standardized globally, e.g. `ftd/<project>/<feature>`?
4. Should FTD include a dashboard plugin or rely on existing Kanban dashboard plus CLI status?
5. Should a global liveness sentinel monitor all FTD projects in addition to project-scoped watchdogs?
6. Should Claude planning be mandatory for every sprint, or skipped for trivial continuation/cleanup sprints?
7. What exact threshold defines “substantial feature bundle” for checkpoint notification? Config can express rough limits, but PM judgment remains necessary.

---

## 20. Recommended Immediate Next Step

Do not start autonomous feature development yet.

First build and test the FTD control plane:

1. `ftd_watchdog.py`
2. `ftd_liveness_check.py`
3. status/watchdog visibility
4. split-brain protection tests
5. dead-runner recovery tests
6. generated cron wrapper smoke test

Only after that should we layer the Claude/Codex sprint pipeline and environment cleanup system.

---

## 21. Success Criteria for the Ultimate FTD

The system is ready when all are true:

1. A repo can be started explicitly and enters `ACTIVE` with exactly one PM runner.
2. A PM runner can complete a sprint, write handoff, set `READY_NEXT_SPRINT`, and exit.
3. Watchdog starts the next sprint silently with fresh context.
4. Checkpoint state pauses and notifies Benjamin once, not repeatedly.
5. Dead runner is detected and reported/recovered.
6. Generic Kanban dispatcher cannot claim PM control cards.
7. Claude planner output is captured.
8. Codex implementation output is captured.
9. Claude review output is captured.
10. PM verification gates are recorded.
11. Environment cleanup only removes FTD-owned resources.
12. Commits/pushes happen according to config.
13. Handoff is compact and sufficient for a fresh PM context.
14. Benjamin is notified only for substantial checkpoint/block/error/approval/complete states.
15. FTD can safely modify its own control plane only with tests, canary, and rollback.
