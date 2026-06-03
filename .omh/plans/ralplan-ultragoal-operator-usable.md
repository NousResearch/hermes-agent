---
title: RALPLAN — Ultragoal Operator-Usable Hermes Direct Goal Lane
status: pending_approval
created_utc: 2026-06-03
planning_artifact: /home/ubuntu/.hermes/hermes-agent/.worktrees/plan-20260603-ultragoal-operator-usable/.omh/plans/ralplan-ultragoal-operator-usable.md
intended_canonical_path: .omh/plans/ralplan-ultragoal-operator-usable.md
target_repo: /home/ubuntu/.hermes/hermes-agent
plan_worktree: /home/ubuntu/.hermes/hermes-agent/.worktrees/plan-20260603-ultragoal-operator-usable
source_request: "Chris: ultragoal 실사용까지 내가 할수있게 하기 위한 ralplan을 작성해"
admission_state: not_admitted
execution_authority: not_admitted
execution_approved: false
approval_boundary: "Planning only. No implementation, worker dispatch, commit/push/PR, gateway restart/reload, deploy/live apply, env/secret/provider/customer mutation."
ralplan_consensus_gate:
  complete: true
  architect_verdict: APPROVE
  critic_verdict: APPROVE
---

# RALPLAN — Ultragoal Operator-Usable Hermes Direct Goal Lane

## 0. One-sentence goal

Make Ultragoal usable by Chris as a separate lane from Autopilot: **Hermes Agent directly executes a Kanban-authorized goal through a durable `/goal` loop without using the Kanban dispatcher as the worker substrate**, while Kanban remains the authority, Done Criteria SSOT, and audit ledger.

This includes two operator modes:

1. **Single-card mode** — run one Kanban task/card to PR-ready or a terminal blocker.
2. **Parent-task mode** — run a Kanban parent task as the top-level Ultragoal objective, using the parent + hierarchy children as the durable goal/subgoal authority set, still without using the Kanban dispatcher as executor.

## 1. Current-state context package

### User intent / lane boundary

Chris's intended lane split:

- **Autopilot**: parent-scoped selector/controller over the existing Kanban dispatcher.
- **Ultragoal**: Hermes Agent direct execution lane using durable `/goal` / subgoal / checkpoint artifacts; **does not rely on Kanban dispatcher worker spawn as its executor**. It must support both single-card execution and Kanban parent-task execution.

This RALPLAN is only for the second lane. Parent-task support here means **Ultragoal parent mode**, not Autopilot parent continuation. The defining difference is that child cards may be used as scoped subgoal/evidence units, but they are not handed to `kanban dispatch`.

### Current code anchors inspected

- `hermes_cli/kanban_ultragoal.py`
  - Current durable Kanban-Ultragoal controller.
  - Provides authority snapshot, pilot-check, run root, ledger, state machine, evidence gates.
  - Does **not** yet implement a direct Hermes `/goal` execution loop.
- `hermes_cli/ultragoal.py`
  - Current source-compatible Ultragoal artifact layer.
  - Provides durable `goals.json`, `ledger.jsonl`, quality-gate reconciliation concepts.
  - Not yet fully integrated with `kanban_ultragoal.py` as the execution engine.
- `tests/hermes_cli/test_kanban_ultragoal.py`
  - Tests authority gates, stale snapshot blocking, transition gates, PR/CI/review_ready artifact requirements.
- `tests/hermes_cli/test_ultragoal.py`
  - Tests current durable Ultragoal artifact behavior.
- `gateway/run.py:_handle_autopilot_command`
  - Confirms Autopilot is intentionally a deterministic wrapper over `kanban dispatch --parent`.
- `hermes_cli/kanban_db.py`
  - Confirms existing dispatcher/ready-gate substrate belongs to Autopilot, not this Ultragoal lane.

### Current proof baseline

Focused current tests passed before this plan:

```text
python -m pytest tests/hermes_cli/test_kanban_ultragoal.py tests/hermes_cli/test_ultragoal.py -q
32 passed in 3.15s

python -m pytest <autopilot selected tests> tests/hermes_cli/test_kanban_ultragoal.py tests/hermes_cli/test_ultragoal.py -q
39 passed in 5.30s
```

### Current capability accounting

Current Ultragoal is approximately:

- Authority / pilot gate: medium-high.
- Evidence state machine: medium-high.
- Direct Hermes `/goal` execution loop: low.
- Failure-to-repair subgoal loop: low-medium.
- Operator usable `run/tick/resume`: low-medium.
- Parent-task Ultragoal mode: not yet covered by current implementation.
- Upstream OMX Ultragoal parity: incomplete.

This plan therefore treats the current code as **durable controller prototype**, not finished Ultragoal.

## 2. Principles

1. **Lane separation first** — Ultragoal must not become a renamed Autopilot or a wrapper around `kanban dispatch`.
2. **Port, do not imitate** — original OMX Ultragoal command/state/artifact semantics must be read directly and converted into a parity matrix before implementation claims.
3. **Kanban authority, not Kanban dispatcher** — Kanban provides task authority, Done Criteria, approval, and audit; Hermes direct `/goal` loop performs execution.
4. **Parent scope is explicit** — if the target is a Kanban parent, the parent + hierarchy child set forms the allowed objective boundary; dependency edges are scheduling constraints, not scope membership.
5. **Evidence gates are executable, not prose** — PR/review_ready may happen only after per-criterion evidence and independent verifier pass.
6. **Durable resume is part of the product** — tool-budget or context-limit pauses must leave a machine-readable checkpoint and continue on the next tick.

## 3. Decision drivers

1. Chris needs a command he can actually use without child-by-child babysitting.
2. Ultragoal must survive long work, context loss, tool limits, and reviewer/CI failure without fake completion.
3. The implementation must preserve SSOT truth: Kanban is authority; Ultragoal artifacts are execution journal; PR is review artifact; merge/live apply remain separate approval gates.

## 4. Viable options

### Option A — Extend current `kanban_ultragoal.py` directly into an execution engine

Pros:
- Fastest path from BO-196 smoke.
- Existing gates and tests already present.
- Fewer files initially.

Cons:
- Risks burying direct `/goal` execution in a state-machine file.
- Higher chance of local imitation instead of upstream Ultragoal parity.
- Harder to keep artifact compatibility clean.

### Option B — Create a separate `hermes_cli/ultragoal_runtime.py` engine and keep `kanban_ultragoal.py` as authority/gate adapter

Pros:
- Clean lane boundary: runtime brain separate from Kanban authority gate.
- Easier to port upstream OMX Ultragoal nearly verbatim into a runtime core.
- Easier tests: upstream parity tests, Hermes adapter tests, Kanban authority tests.

Cons:
- More files and integration seams.
- Requires careful handoff between runtime and current state machine.

### Option C — Reuse Kanban dispatcher and call it Ultragoal

Pros:
- Fastest apparent “working” behavior.
- Existing dispatcher already spawns workers and reviewers.

Cons:
- Violates Chris's lane definition.
- Collapses Ultragoal into Autopilot.
- Does not prove Hermes direct `/goal` execution.

### Option D — Implement only a skill/prompt wrapper

Pros:
- Very fast.

Cons:
- Not progress by Chris's standard.
- No durable execution, no gate, no recovery.
- Repeats the “SKILL.md is not enough” failure.

## 5. Proposed decision

Choose **Option B**.

Build an explicit runtime core:

```text
hermes_cli/ultragoal_runtime.py        # direct Hermes goal-loop runtime core
hermes_cli/ultragoal_adapters.py       # Kanban/GitHub/SessionDB adapters, narrow and testable
hermes_cli/kanban_ultragoal.py         # authority snapshot + transition gate + CLI operator shell
hermes_cli/ultragoal.py                # upstream/source-compatible artifact model, preserved/extended
```

`kanban_ultragoal.py` remains the operator-facing Kanban authority command, but it delegates execution decisions to `ultragoal_runtime.py` rather than the Kanban dispatcher.

## 6. ADR

### Decision

Implement Ultragoal as a Hermes direct `/goal` durable runtime that uses Kanban for authority and audit but does not use the Kanban dispatcher as the worker substrate.

### Drivers

- Separate Ultragoal from Autopilot cleanly.
- Preserve original OMX Ultragoal behavior instead of building a Hermes-shaped imitation.
- Give Chris a usable command with truthful terminal reports only.
- Support both `run <single-card>` and `run <parent-card>` while keeping parent mode distinct from Autopilot.

### Alternatives rejected

- **Reuse Kanban dispatcher** rejected because that is Autopilot's lane.
- **Prompt/skill wrapper only** rejected because it cannot enforce completion truth.
- **Single monolithic `kanban_ultragoal.py`** rejected as too likely to blur authority gate and runtime brain.

### Consequences

- More initial architecture work.
- Better long-term correctness and debuggability.
- Requires explicit upstream parity matrix as the first implementation slice.
- Requires new tests for direct `/goal` execution and resume behavior.

### Follow-ups not in v1

- Gateway live `/ultragoal` slash UX.
- Full parent-level Autopilot integration.
- Merge/live runtime application automation.
- Production/customer-facing execution.

## 7. Target operator UX

Chris should be able to say or run:

```text
BO-203을 ultragoal로 실행해.
Kanban dispatcher는 쓰지 말고, Hermes /goal loop가 직접 수행해.
Kanban card와 Done criteria는 authority로 삼고,
진행상황은 .hermes/goal-runs/<id>/에 durable하게 남겨.
작업은 subgoal로 쪼개고, 실패/리뷰 지적은 다음 subgoal로 흡수해.
PR 생성이나 review_ready는 verifier가 Done criteria별 evidence를 확인한 뒤에만 해.
보고는 pr_ready / blocked_needs_user_decision / approval_required / fatal_infra_blocker 중 하나일 때만 해.
```

Equivalent CLI shape:

```bash
hermes kanban-ultragoal pilot-check BO-203
hermes kanban-ultragoal run BO-203 --terminal-only
hermes kanban-ultragoal status BO-203
hermes kanban-ultragoal resume BO-203
```

`run` may internally perform bounded ticks, but each tick must be checkpointable.

Parent-task UX must also be supported:

```text
BO-195 parent를 ultragoal로 실행해.
Kanban dispatcher는 쓰지 말고, Hermes /goal loop가 parent objective를 직접 수행해.
parent와 hierarchy children을 scope로 삼고, child들은 subgoal/evidence 단위로 사용해.
raw/unready child를 dispatcher에 넘기지 말고, parent Done Criteria와 child evidence를 기준으로 진행/차단해.
범위 밖 child나 stale hierarchy가 발견되면 blocked_needs_user_decision으로 멈춰.
보고는 pr_ready / blocked_needs_user_decision / approval_required / fatal_infra_blocker 중 하나일 때만 해.
```

Parent CLI shape:

```bash
hermes kanban-ultragoal pilot-check BO-195 --mode parent
hermes kanban-ultragoal run BO-195 --mode parent --terminal-only
hermes kanban-ultragoal status BO-195
hermes kanban-ultragoal resume BO-195
```

## 8. Required state model

### Run root

```text
.hermes/goal-runs/<run_id>/
  run.json
  authority.json
  ledger.jsonl
  goals.json
  brief.md
  checkpoint.json
  current_goal.json
  evidence/
    worker/<goal_id>.json
    verifier/<goal_id>.json
    ci.json
  reviews/
    cycles/<n>.json
    final.json
  pr.json
  terminal_report.json
```

### `run.json` required fields

```json
{
  "version": 2,
  "runId": "t_xxx",
  "publicId": "BO-203",
  "lane": "ultragoal",
  "targetMode": "single|parent",
  "executor": "hermes-direct-goal-loop",
  "dispatcherUsed": false,
  "state": "admitted|running|checkpointed|verification_failed|review_failed|ci_failed|review_ready|blocked|fatal",
  "tick": 0,
  "authority": {},
  "scope": {
    "parentTaskId": null,
    "childTaskIds": [],
    "childSnapshotHashes": {}
  },
  "currentGoalId": null,
  "budgets": {
    "maxWorkerAttemptsPerGoal": 3,
    "maxReviewCycles": 5,
    "maxCiFixCycles": 3,
    "maxWallClockMinutes": 120,
    "maxToolCallsPerTick": 40
  },
  "terminalReport": null
}
```

### Terminal states

Only these states should report to Chris by default:

- `pr_ready`
- `blocked_needs_user_decision`
- `approval_required`
- `fatal_infra_blocker`

## 9. Implementation slices

### Slice 1 — Upstream parity matrix and source-compatible contract

Objective: prevent imitation drift before coding.

Tasks:
1. Locate and read original OMX/oh-my-codex Ultragoal implementation, not just SKILL.md.
2. Create `.omh/plans/ultragoal-upstream-parity-matrix.md` or a section in this plan's follow-up branch.
3. Matrix rows must include:
   - upstream file/function/section
   - upstream behavior
   - current Hermes behavior
   - status: `ported`, `partial_adaptation`, `gap`, `intentional_divergence`
   - required test
4. Add a divergence ledger. Any unapproved divergence that changes behavior blocks implementation.

Acceptance criteria:
- Matrix covers command surface, artifact schema, story/subgoal parsing, steering, checkpoints, quality gate, resume, terminal reporting.
- Chris can see which parts are copied/adapted versus invented.

Verification:

```bash
python -m pytest tests/hermes_cli/test_ultragoal.py tests/hermes_cli/test_kanban_ultragoal.py -q
```

### Slice 2 — Runtime schema v2 and artifact compatibility

Objective: make current artifacts capable of running a direct execution loop.

Files:
- Modify: `hermes_cli/ultragoal.py`
- Modify: `hermes_cli/kanban_ultragoal.py`
- Create: `tests/hermes_cli/test_ultragoal_runtime_schema.py`

Required behavior:
- Preserve current v1 artifact read compatibility.
- Add v2 fields: `lane`, `executor`, `dispatcherUsed=false`, `budgets`, `terminalReport`, `checkpoint`.
- Add helpers to append goals and repair goals to `goals.json`.
- Add ledger event types:
  - `goal_selected`
  - `goal_attempt_started`
  - `goal_attempt_completed`
  - `repair_goal_created`
  - `checkpoint_written`
  - `terminal_report_written`

Acceptance criteria:
- Loading old BO-196-style run roots still works.
- New run roots include explicit `dispatcherUsed=false`.
- Tests fail if dispatcher mode is silently enabled.

### Slice 3 — Direct Hermes `/goal` adapter

Objective: establish the direct execution substrate without Kanban dispatcher.

Files:
- Create: `hermes_cli/ultragoal_adapters.py`
- Create: `tests/hermes_cli/test_ultragoal_goal_adapter.py`

Adapter interface:

```python
class GoalLoopAdapter(Protocol):
    def set_goal(self, run_id: str, objective: str, constraints: dict) -> dict: ...
    def run_goal_turn(self, run_id: str, prompt: str, *, tool_budget: int) -> dict: ...
    def checkpoint(self, run_id: str, summary: str) -> dict: ...
```

Hermes implementation must:
- Use Hermes session/oneshot or agent runtime directly.
- Not call `kanban dispatch` or `dispatch_once`.
- Persist session id / transcript anchors in the run root.
- Respect tool budget caps.

Acceptance criteria:
- Unit test monkeypatches `kanban_db.dispatch_once` to raise if called; Ultragoal tick still passes without it.
- Adapter writes a checkpoint when the simulated tool budget is exhausted.

### Slice 4 — `run/tick/resume` controller

Objective: turn state machine into an operator-usable controller.

Files:
- Modify: `hermes_cli/kanban_ultragoal.py`
- Create: `hermes_cli/ultragoal_runtime.py`
- Create: `tests/hermes_cli/test_kanban_ultragoal_runtime.py`

CLI:

```bash
hermes kanban-ultragoal run <task_ref> [--terminal-only]
hermes kanban-ultragoal tick <run_id> --authority-json <path>
hermes kanban-ultragoal resume <run_id>
hermes kanban-ultragoal status <run_id>
```

Tick sequence:
1. Read live Kanban authority snapshot.
2. Verify `executionApproved=true`, done criteria hash, snapshot freshness.
3. Load run root and goals.
4. Select or create next subgoal.
5. Set Hermes `/goal` objective.
6. Run one bounded direct Hermes turn/attempt.
7. Store worker evidence.
8. Return checkpoint if not terminal.

Acceptance criteria:
- `run` creates a run root and performs at most bounded work.
- `resume` refuses if authority hash drifted.
- `resume` does not ask Chris for child-level continuation unless terminal blocker requires it.

### Slice 5 — Worker evidence contract and per-criterion verifier

Objective: make worker self-report insufficient and verifier mandatory.

Files:
- Create: `hermes_cli/ultragoal_verifier.py`
- Create: `tests/hermes_cli/test_ultragoal_verifier.py`

Worker evidence schema:

```json
{
  "goalId": "g1",
  "attempt": 1,
  "filesChanged": [],
  "commandsRun": [],
  "tests": [],
  "criteriaEvidence": [
    {"criterionId": "DC-1", "evidenceRef": "pytest output / file path", "status": "claimed"}
  ],
  "sideEffects": {
    "merge": false,
    "gatewayRestart": false,
    "deploy": false,
    "envSecretMutation": false,
    "customerMutation": false
  }
}
```

Verifier behavior:
- Load Done Criteria Ledger from Kanban authority.
- Require each criterion to have concrete evidence.
- Reject claimed evidence that does not exist or cannot be read.
- Reject forbidden side effects.
- On failure, create repair subgoal with missing criteria and next required action.

Acceptance criteria:
- PR creation is impossible unless verifier passes each criterion.
- Failure creates a concrete repair goal in `goals.json`, not just `current_goal_id` text.

### Slice 6 — Reviewer and CI repair loop

Objective: make review/CI failure become bounded work rather than manual drift.

Files:
- Create: `hermes_cli/ultragoal_reviewer.py`
- Modify: `hermes_cli/kanban_ultragoal.py`
- Create: `tests/hermes_cli/test_ultragoal_review_ci_repair.py`

Behavior:
- Reviewer result must include explicit `securityConcerns=[]`, `logicErrors=[]`, `scopeDrift=[]`.
- REQUEST_CHANGES creates repair goal.
- CI failure creates `ci-repair-*` goal with failing check summary and head SHA.
- Repair attempts are capped by configured budgets.
- Exceeding cap results in `blocked_needs_user_decision` with evidence.

Acceptance criteria:
- Reviewer failure cannot be marked review_ready.
- CI head SHA mismatch remains fail-closed.
- CI failure becomes a repair goal and can reach `ci_passed` after rerun.

### Slice 7 — PR creation and review_ready proof package

Objective: give Chris a final review package, not a stream of partial reports.

Files:
- Create: `hermes_cli/ultragoal_pr.py`
- Modify: `hermes_cli/kanban_ultragoal.py`
- Create: `tests/hermes_cli/test_ultragoal_pr_ready_package.py`

Behavior:
- PR creation occurs only after worker evidence + verifier pass + reviewer approve.
- PR body includes:
  - source Kanban card
  - Done Criteria Ledger
  - per-criterion evidence table
  - verification commands/results
  - reviewer verdict
  - explicit non-actions
- review_ready requires PR tuple: `url`, `number`, `headSha`, latest CI success for same head.

Acceptance criteria:
- Stale PR head is rejected.
- Missing evidence file is rejected.
- Final `terminal_report.json` contains only safe summary fields.

### Slice 7A — Cleanup / residue gate before review_ready

Objective: make cleanup part of Ultragoal lifecycle closeout, not an afterthought, cron job, or policy-only document. `review_ready` is forbidden until residue is either cleaned, preserved as evidence, or retained with a reason and TTL.

Files:
- Create: `hermes_cli/ultragoal_cleanup.py`
- Modify: `hermes_cli/kanban_ultragoal.py`
- Modify: `hermes_cli/ultragoal_runtime.py`
- Create: `tests/hermes_cli/test_ultragoal_cleanup_gate.py`

Required behavior:
1. Each Ultragoal run records a workspace/artifact registry in the run root: implementation worktree path, branch, run artifacts, evidence files, cache/build/test residue candidates, linked PR tuple when present, and parent/child evidence roots for parent mode.
2. Cleanup policy is explicit and machine-readable:
   - **Can delete:** generated caches, temporary build/test outputs, disposable logs already summarized into evidence, failed throwaway artifacts with no audit value.
   - **Must preserve:** `run.json`, `goals.json`, ledger events, authority snapshot hashes, verifier results, terminal reports, PR/review evidence, blocker records, and any artifact referenced by a Done Criteria evidence row.
   - **Must never touch:** canonical checkout, active process CWD, dirty implementation worktree, unpushed user-owned branch, unrelated worktrees, secrets/env files, production/customer/provider state, and Kanban authority except through approved status/evidence updates.
3. Candidate cleanup is gated by live checks before deletion: registered worktree status, dirty/untracked state, active process CWD/lsof where available, branch/remote preservation, PR/open review dependency, file path allowlist, and artifact-reference reachability.
4. `review_ready` requires a cleanup proof package for **cleanup/residue candidates**: cleaned candidates, preserved artifacts with hashes/paths, retained residue with reason + TTL/revisit gate, skipped deletion reason codes, and explicit non-actions. This gate must not require deletion of the active implementation worktree needed for PR/review iteration; that worktree must instead be classified as preserved/retained with evidence, reason, and revisit condition.
5. Parent-task mode includes a child cleanup matrix in the parent terminal report: child/subgoal id, evidence root, cleanup result, retained residue reason/TTL, and blocker if cleanup prevents review readiness.
6. Cleanup failure blocks `review_ready` or records `blocked_needs_user_decision`; it must not be silently downgraded to a warning. The state transition is based on a recorded, read-only cleanup proof artifact in the run root, not on recomputing mutable local filesystem state at decision time.
7. Post-merge/closed cleanup is a separate lifecycle step: after merge/close confirmation, remove only clean, inactive implementation worktrees via `git worktree remove`, prune, verify absence, and record cleanup evidence before final `closed`/`final_done`. It must never target the active run root, evidence-bearing worktree/artifacts, canonical checkout, or any workspace still needed for PR/review iteration.

Acceptance criteria:
- `test_ultragoal_review_ready_blocks_without_cleanup_proof` proves PR/review_ready cannot proceed without cleanup evidence.
- `test_ultragoal_records_retained_worktree_reason_and_ttl` proves retained residue is explicit and revisitable.
- `test_ultragoal_does_not_remove_dirty_or_active_worktree` proves dirty or active worktrees are never deleted.
- `test_ultragoal_cleanup_policy_classifies_delete_preserve_never_touch` proves cleanup categories are executable, not prose-only.
- `test_ultragoal_parent_terminal_report_includes_child_cleanup_matrix` proves parent mode reports child cleanup state.
- `test_ultragoal_post_merge_cleanup_removes_clean_inactive_worktree` proves closed/final_done cleanup removes safe registered worktrees and prunes them.

### Slice 8 — Operator UX and live pilot

Objective: make Chris able to use it safely.

Files:
- Modify: `hermes_cli/commands.py` if needed for help/command registry.
- Modify: docs or CLI help for `kanban-ultragoal`.
- Create: `tests/hermes_cli/test_kanban_ultragoal_operator_cli.py`.

Behavior:
- `pilot-check` remains read-only.
- `run --dry-run` shows exact planned first tick without mutation.
- `run` requires current approval and side-effect boundaries.
- Live pilot uses one explicit test card only.

Acceptance criteria:
- Chris can run one command after approving a card.
- It either returns terminal report or checkpoint/resume pointer.
- It does not use Kanban dispatcher.

### Slice 9 — Kanban parent-task Ultragoal mode

Objective: allow Chris to specify a Kanban parent task as the Ultragoal target while preserving the Ultragoal lane boundary: Hermes direct `/goal` execution, no Kanban dispatcher worker handoff.

Files:
- Modify: `hermes_cli/kanban_ultragoal.py`
- Modify: `hermes_cli/ultragoal_runtime.py`
- Modify: `hermes_cli/ultragoal_adapters.py`
- Create: `tests/hermes_cli/test_kanban_ultragoal_parent_mode.py`

Required behavior:
1. `pilot-check <task_ref> --mode parent` identifies a parent/umbrella task and returns parent identity, parent Done Criteria/hash, hierarchy children, child statuses, child Done Criteria hashes, and dependency edges as ordering/blocking information rather than scope membership.
2. `run <parent> --mode parent` writes `targetMode=parent`, `scope.parentTaskId`, `scope.childTaskIds`, and `scope.childSnapshotHashes` into `run.json`.
3. Parent hierarchy children are projected into `goals.json` as subgoal candidates, but **not** claimed/spawned through `kanban dispatch`.
4. Hermes direct `/goal` loop selects the next child/subgoal to work on, performs the work in the active Ultragoal run, and stores child-scoped evidence under the parent run root.
5. If a child is raw/unready, out-of-scope, stale, missing Done Criteria, blocked by dependency, or requires side-effect approval, parent mode produces a terminal or checkpointed blocker instead of silently dispatching it.
6. Parent terminal report summarizes parent Done Criteria evidence, child/subgoal evidence status, skipped/deferred children with reason codes, PR/review package or blocker pointer, and explicit non-actions.

Parent-mode distinction from Autopilot:

```text
Autopilot parent mode:
  parent -> select eligible child -> kanban dispatch/spawn worker -> verifier/review flow

Ultragoal parent mode:
  parent -> build durable goal/subgoal ledger -> Hermes direct /goal execution -> parent evidence package
```

Acceptance criteria:
- `test_parent_ultragoal_never_calls_kanban_dispatcher` monkeypatches dispatcher calls to raise and parent mode still runs through the direct adapter.
- `test_parent_ultragoal_uses_hierarchy_children_as_subgoals` proves hierarchy children appear in `goals.json` and ledger events.
- `test_parent_ultragoal_blocks_when_child_scope_drift_detected` proves changed child snapshot/hash blocks resume.
- `test_parent_ultragoal_dependency_edges_are_not_scope_membership` proves dependency-only linked tasks do not enter the child scope.
- `test_parent_ultragoal_terminal_report_summarizes_child_evidence` proves terminal report is parent-level and child-aware.

Out of scope for this slice:
- Autopilot-style automatic promotion of raw children to ready.
- Kanban dispatcher child handoff.
- Global queue draining.
- Merge/restart/deploy/live apply.

## 10. Verification plan

### Unit tests

```bash
python -m pytest \
  tests/hermes_cli/test_ultragoal.py \
  tests/hermes_cli/test_kanban_ultragoal.py \
  tests/hermes_cli/test_ultragoal_runtime_schema.py \
  tests/hermes_cli/test_ultragoal_goal_adapter.py \
  tests/hermes_cli/test_kanban_ultragoal_runtime.py \
  tests/hermes_cli/test_ultragoal_verifier.py \
  tests/hermes_cli/test_ultragoal_review_ci_repair.py \
  tests/hermes_cli/test_ultragoal_pr_ready_package.py \
  tests/hermes_cli/test_ultragoal_cleanup_gate.py \
  tests/hermes_cli/test_kanban_ultragoal_parent_mode.py -q
```

### Regression tests that must exist

- `test_ultragoal_never_calls_kanban_dispatcher_for_direct_run`
- `test_resume_refuses_stale_authority_snapshot`
- `test_budget_exhaustion_writes_checkpoint_and_terminal_report_is_absent`
- `test_verifier_requires_each_done_criterion_evidence`
- `test_reviewer_failure_creates_repair_subgoal`
- `test_ci_head_sha_must_match_pr_head_before_review_ready`
- `test_terminal_only_mode_suppresses_non_terminal_reports`
- `test_ultragoal_review_ready_blocks_without_cleanup_proof`
- `test_ultragoal_records_retained_worktree_reason_and_ttl`
- `test_ultragoal_does_not_remove_dirty_or_active_worktree`
- `test_ultragoal_cleanup_policy_classifies_delete_preserve_never_touch`
- `test_ultragoal_parent_terminal_report_includes_child_cleanup_matrix`
- `test_ultragoal_post_merge_cleanup_removes_clean_inactive_worktree`
- `test_parent_ultragoal_never_calls_kanban_dispatcher`
- `test_parent_ultragoal_uses_hierarchy_children_as_subgoals`
- `test_parent_ultragoal_blocks_when_child_scope_drift_detected`
- `test_parent_ultragoal_dependency_edges_are_not_scope_membership`
- `test_parent_ultragoal_terminal_report_summarizes_child_evidence`
- `test_upstream_parity_matrix_has_no_unapproved_gap_for_v1_required_behaviors`

### Live smoke ladder

1. Read-only `pilot-check` on real card.
2. Disposable direct-run card: no PR, simulated worker/verifier/repair.
3. Real low-risk child card: direct Hermes run to PR-ready.
4. Resume smoke: force checkpoint, then resume and complete.
5. Failure smoke: verifier fails one Done criterion, repair goal is created, rerun completes.
6. Cleanup-gate smoke: residue is classified, safe cache is removed or retained with TTL, dirty/active worktree deletion is blocked, and `review_ready` waits for cleanup proof.
7. Parent-mode dry-run smoke: real parent card scope read, hierarchy children projected, no dispatcher call.
8. Parent-mode disposable smoke: parent + two disposable children, one success and one verifier repair, parent terminal report with child cleanup matrix produced.

No merge, gateway restart/reload, deploy/live apply, env/secret/provider/customer mutation in any smoke unless separately approved.

## 11. Risk matrix

| Risk | Likelihood | Impact | Mitigation |
|---|---:|---:|---|
| Ultragoal silently uses Kanban dispatcher and becomes Autopilot | Medium | High | Add tests monkeypatching dispatcher to fail; run state records `dispatcherUsed=false` |
| Local imitation diverges from original OMX Ultragoal | High | High | Upstream parity matrix first; unapproved gaps block |
| `/goal` remains guidance not enforcement | High | High | Controller enforces transitions and evidence gates; `/goal` only execution objective |
| Worker self-report is accepted as proof | Medium | High | Independent verifier checks each Done criterion against concrete evidence refs |
| Long run loses state | Medium | High | Checkpoint/resume required every tick; terminal reports only after package complete |
| Tool budget exhaustion causes fake completion | Medium | High | Budget exhaustion writes `checkpointed`, never `review_ready` |
| Side effects happen without approval | Low-Medium | High | Side-effect ledger + forbidden flags + verifier rejection + prompt guard |
| Parent mode becomes Autopilot by dispatching children | Medium | High | Parent-mode dispatcher monkeypatch regression, `targetMode=parent`, child projection into goals ledger only |
| Parent scope drifts after resume | Medium | High | Store child snapshot hashes and block resume on parent/child scope mismatch |
| Cleanup is treated as policy-only documentation | High | High | `review_ready` blocks without executable cleanup proof and retained-residue TTL |
| Cleanup deletes live or user-owned work | Medium | High | Dirty/active worktree, branch preservation, allowlist, and never-touch checks before deletion |

## 12. Pre-mortem

1. **Looks complete but is just BO-196-style manual recording.**
   - Prevention: direct runtime tests require adapter-driven execution and forbid manual-only recorder success.
2. **Ultragoal becomes a wrapper over dispatcher.**
   - Prevention: dispatcher monkeypatch regression and lane metadata.
3. **Verifier passes vague evidence.**
   - Prevention: evidence refs must be readable or command output attached; per-criterion mapping required.
4. **Parent task execution silently turns into Autopilot.**
   - Prevention: parent-mode tests forbid dispatcher calls and require hierarchy children to be projected as Ultragoal subgoals.

## 13. Rollback / cleanup plan

- All implementation occurs on a feature branch; no gateway restart/reload during implementation.
- If runtime loop is unstable, keep existing `kanban-ultragoal` state-machine commands and disable new `run/resume` path behind CLI flag/config.
- Run roots are local artifacts under `.hermes/goal-runs/<id>` and can be archived per run id without touching Kanban authority.
- Cleanup is lifecycle-gated, not cron-primary: review_ready requires recorded read-only cleanup proof for cleanup/residue candidates; closed/final_done requires post-merge cleanup proof where a merge/close happened.
- Cleanup deletion is allowlist-only and must never touch canonical checkout, active process CWD, active run root, evidence-bearing worktree/artifacts, dirty worktrees, unpushed user-owned branches, unrelated worktrees, secrets/env files, or prod/customer/provider state.
- Retained residue must have reason + TTL/revisit gate; otherwise it blocks final closeout.
- No migration should rewrite historical BO-196 run roots destructively; v1 compatibility reader required.
- Plan worktree must be promoted or explicitly retained/removed after planning closeout.

## 14. Acceptance criteria for “Chris can 실사용”

Chris can consider Ultragoal operator-usable only when all are true:

- `hermes kanban-ultragoal pilot-check <card>` is read-only and accurate.
- `hermes kanban-ultragoal run <card>` uses Hermes direct `/goal` loop, not Kanban dispatcher.
- `hermes kanban-ultragoal run <parent> --mode parent` uses Hermes direct `/goal` loop, not Kanban dispatcher, and projects hierarchy children into the durable subgoal ledger.
- Run root records `dispatcherUsed=false` and session/checkpoint anchors.
- Parent-mode run root records `targetMode=parent`, parent task id, hierarchy child ids, and child snapshot hashes.
- Done Criteria are loaded from Kanban authority and hashed.
- Worker attempts produce structured evidence.
- Independent verifier checks each criterion.
- Review/CI failures create bounded repair goals.
- Tool-budget exhaustion creates resumable checkpoint.
- Cleanup/residue gate blocks PR/review_ready until recorded read-only cleanup proof exists for cleanup candidates, or retained residue has reason + TTL/revisit gate.
- Dirty or active worktrees are never deleted; clean inactive implementation worktrees are removed only at the correct closed/final_done lifecycle step.
- Parent terminal report includes child cleanup matrix.
- Final report appears only in allowed terminal states.
- One disposable smoke and one real low-risk task-card smoke reach PR-ready without merge/live side effects.
- Parent-mode dry-run and disposable parent smoke prove parent scope, child evidence summary, stale-child blocker, and no dispatcher calls.

## 15. Explicit non-goals

- Do not implement Autopilot parent-child continuation here. Parent-task Ultragoal mode is allowed only as Hermes direct `/goal` execution over parent + hierarchy child scope, with no dispatcher handoff.
- Do not use Kanban dispatcher as executor.
- Do not merge PRs.
- Do not restart/reload gateway.
- Do not deploy/live apply.
- Do not mutate prod/customer/env/secret/provider state.
- Do not claim full upstream parity until the parity matrix and tests prove it.

## 16. Architect review

Verdict: **APPROVE**

Architect assessment:

- The plan correctly separates Autopilot and Ultragoal by executor substrate.
- Option B is architecturally sound because it isolates the runtime brain from the Kanban authority gate.
- The strongest tradeoff is added file/module complexity, but this is justified because a monolithic controller would blur SSOT and lane boundaries.
- The test plan explicitly prevents the main likely regression: accidentally calling the Kanban dispatcher.
- The plan preserves current BO-196 smoke assets while acknowledging they are not enough.
- Revision note: parent-task mode is now explicitly included as a separate Ultragoal target mode, not as Autopilot continuation.

Required revisions folded into final plan:

- Cleanup re-review revisions folded: cleanup proof scope is limited to cleanup/residue candidates; active PR/review worktrees are preserved/retained rather than forced-deleted; cleanup proof is recorded read-only evidence in the run root; post-merge cleanup targets only clean inactive implementation worktrees and never active run roots/evidence/canonical checkout.
- Added explicit `dispatcherUsed=false` artifact field.
- Added regression name `test_ultragoal_never_calls_kanban_dispatcher_for_direct_run`.
- Added `v1 compatibility reader required` rollback line.
- Added parent-task mode scope model, tests, and smoke ladder after Chris clarified parent execution must be covered.

Final re-review after cleanup wording revisions: **APPROVE**

- Cleanup gate now targets cleanup/residue candidates rather than forcing deletion of active PR/review worktrees.
- `review_ready` state transition uses a recorded read-only cleanup proof artifact in the run root.
- Post-merge cleanup is limited to clean inactive implementation worktrees and excludes active run roots, evidence-bearing artifacts/worktrees, canonical checkout, and review-iteration workspaces.
- Lane boundary and Kanban SSOT remain intact.

## 17. Critic review

Verdict: **APPROVE**

Critic assessment:

- The plan does not overclaim current completion.
- It rejects prompt-only enforcement and dispatcher reuse, which are the two biggest slop paths.
- Acceptance criteria are testable and tied to concrete files/commands.
- The live smoke ladder is appropriately staged and avoids merge/restart/live side effects.
- The only remaining ambiguity is the exact source path of original OMX Ultragoal, but the plan correctly makes locating and reading that source Slice 1 rather than assuming memory.
- Revision note: parent-mode coverage now blocks the earlier single-card-only gap; the plan still correctly rejects dispatcher reuse.

Required revisions folded into final plan:

- Cleanup re-review revisions folded: narrowed cleanup-gate scope, made cleanup proof a recorded read-only artifact, and hardened post-merge deletion boundaries.
- Added source-location task to Slice 1.
- Added unapproved divergence blocker language.
- Added terminal-only reporting regression.

Final re-review after cleanup wording revisions: **APPROVE**

- Prior Critic REVISE items are folded: cleanup scope narrowed, proof made a recorded read-only artifact, and deletion boundaries hardened.
- No remaining cleanup overreach or accidental-deletion ambiguity blocks consensus.
- Consensus can remain complete; execution is still pending approval/admission only.
- Added parent-mode dispatcher-forbidden regressions and child-scope drift blocker requirements.

## 18. Consensus / admission record

```yaml
ralplan_consensus_gate:
  complete: true
  final_planner_plan: this document
  architect_review:
    verdict: APPROVE
    revisions_folded: true
  critic_review:
    verdict: APPROVE
    after_architect: true
    revisions_folded: true
admission_state: not_admitted
execution_authority: not_admitted
execution_approved: false
approval_boundary:
  allowed_now:
    - read and discuss this plan
  forbidden_without_new_approval:
    - implementation code edits
    - worker dispatch
    - commit/push/PR
    - gateway restart/reload
    - deploy/live apply
    - env/secret/provider/customer mutation
```

## 19. Recommended next Kanban admission shape

If Chris approves this plan, create or update a parent Kanban card:

```text
Title: BO-203 — Ultragoal operator-usable direct Hermes goal lane
Routing: direct-kanban
Execution approval: false at admission; promote only after ready contract exists
Done when:
- Upstream Ultragoal parity matrix exists with no unapproved v1 gaps.
- Direct Hermes /goal adapter runs without Kanban dispatcher calls.
- run/tick/resume/status CLI works with durable checkpoint/resume.
- Per-criterion verifier blocks PR/review_ready until evidence passes.
- Review/CI failures create bounded repair subgoals.
- Disposable and real low-risk live smokes reach PR-ready with no forbidden side effects.
- Parent-mode dry-run and disposable parent smoke prove hierarchy child projection, stale-scope blocking, and parent terminal report.
Forbidden:
- Autopilot dispatcher execution path as Ultragoal executor.
- merge/restart/deploy/live apply/prod/env/secret/provider/customer mutation.
```

Suggested children:

1. Upstream parity matrix and divergence ledger.
2. Runtime schema v2 and artifact compatibility.
3. Direct Hermes `/goal` adapter.
4. `run/tick/resume` controller.
5. Worker evidence + per-criterion verifier.
6. Reviewer/CI repair loop.
7. PR/review_ready proof package.
8. Cleanup/residue gate before review_ready and post-merge closed cleanup.
9. Operator CLI/live smoke.
10. Parent-task Ultragoal mode.

Execution remains unapproved until Chris explicitly approves the parent/card and side-effect boundaries in a later turn.
