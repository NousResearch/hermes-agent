# Dev Project Goals — Feature Spec

Status: **Implemented (v1)** · Owner: Dev (Hermes dev profile) · Created: 2026-05-30

A project-level goal system for the Dev workspace: a durable, hierarchical
spine of **vision → goal → milestone → subgoal** that gives every
clarification → plan → execution cycle a stable "why," and periodically
re-evaluates progress using the same judge pattern that powers session goals.

**Related artifacts**

- OpenSpec change: `openspec/changes/add-dev-project-goals/` (proposal, design,
  requirement scenarios, tasks)
- Code: `gateway/dev_control/project_goals.py`,
  `gateway/dev_control/project_goal_eval.py`

---

## 1. Motivation

Hermes already has a **session goal** feature (`hermes_cli/goals.py`): a
free-form objective that persists across turns, with an auxiliary "judge"
model deciding after each turn whether the goal is satisfied. It is excellent
for keeping a single session on task, but it is ephemeral and session-scoped.

The Dev workspace needs the same idea at a **project** scope:

- A product/project **vision** that outlives any one session.
- **Goals** for short-term feature themes.
- **Milestones** for shippable increments.
- **Subgoals** that map to concrete deliverables (the plans/tasks we already track).

Today `gateway/dev_control/` tracks the *execution* spine
(`clarification → plan_artifact → execution plan → tasks → verification/CI`)
but had no long-lived layer above it expressing intent and tracking progress
toward it. This feature adds that layer.

## 2. Non-goals

- **Not** a replacement for `plan_artifacts`, `dev_execution`, or kanban. Goals
  sit *above* plans and link down to them; they do not duplicate task tracking.
- **Not** a general-purpose project management tool (no Gantt, no assignees, no
  time tracking). Scope is goal hierarchy + progress rollup + re-evaluation.
- **Not** a new orchestration/execution engine. Execution stays on the existing
  loops (cron, lab_loop, kanban dispatcher).

## 3. Decision: extend `dev_control`, do not adopt Temporal

A vision → goal → milestone → subgoal system is fundamentally a **data model +
status rollup + periodic re-evaluation** problem, not a durable-workflow-engine
problem.

- The persistence substrate already exists: SQLite via `hermes_state`
  (`DEFAULT_DB_PATH`, `apply_wal_with_fallback`), and a proven store pattern in
  `gateway/dev_control/plan_artifacts.py`.
- The "keep advancing / re-judge progress" loop already has homegrown durable
  hosts: `cron/scheduler.py`, `gateway/dev_control/lab_loop.py`, and the kanban
  dispatcher.
- The judgment primitive already exists: `judge_goal()` in `hermes_cli/goals.py`.

**Alternatives considered.** Temporal (durable workflow engine) was evaluated
and rejected for v1: it requires running a Temporal server/cluster (or Temporal
Cloud) plus worker processes and the SDK — a heavyweight, stateful infra
component that conflicts with the repo's self-hostable, dependency-minimal
posture (see `AGENTS.md` dependency-pinning / supply-chain policy). Temporal
solves crash-durable distributed *execution code*, which the goal **data model**
does not need. See §12 for the narrow future scenario where it could earn its
place.

## 4. Concepts & hierarchy

```
vision      project-lifetime intent
  └─ goal          a quarter / feature theme
       └─ milestone     a shippable increment
            └─ subgoal       a concrete deliverable → links to execution
```

- A **subgoal** is the join point to execution: set `plan_artifact_id` and/or
  `payload.plan_id` (resolved via latest `dev_plan_artifact_builds` row when only
  the artifact id is set).
- **Parent kinds are enforced:** goal → vision, milestone → goal, subgoal →
  milestone.
- **Leaves are evaluated; parents are rolled up.** Only leaf subgoals are
  re-evaluated. Milestones/goals/vision derive status deterministically from
  children. Abandoned children are excluded from rollup.

## 5. Data model

Module `gateway/dev_control/project_goals.py` — `DevProjectGoalStore` on
`DEFAULT_DB_PATH` with WAL via `apply_wal_with_fallback`.

Table `dev_project_goals` (typed columns + JSON `payload` for judge history,
`weight`, and links). Key fields: `kind`, `status`, `acceptance_criteria`,
`plan_artifact_id`, computed `progress`.

Design notes:

- **`project_id`** uses `resolve_project_id()` from `project_scope.py`
  (default `"OrynWorkspace"`).
- **`acceptance_criteria`** reuses the v2 schema from `acceptance_criteria.py`.
- **`progress`** is computed only (rollup or `achieved` → 1.0); callers cannot
  author it via `update()`.
- **Never hard-delete.** Use `abandon` → `status = abandoned`.

Audit trail: each re-evaluation appends to `payload.judge_history` (last 20).

## 6. Status rollup

`recompute_rollup()` in `project_goals.py`:

- **Parent:** weighted mean of child progress (`payload.weight`, default `1.0`);
  child `achieved` counts as progress 1.0 regardless of cached field.
- **Status:** all children `achieved` → parent `achieved`; any child `blocked` →
  parent `blocked`.
- Propagates upward after create, status change, abandon, and subgoal achievement.

## 7. Evidence assembly

Module `gateway/dev_control/project_goal_eval.py` — `assemble_evidence()` for
**v1** pulls from:

- `DevVerificationStore.list_runs(plan_id=…)` — latest verification results
- `DevExecutionStore.get_plan()` — plan + task statuses
- `fetch_ci_status()` — when repo/branch found on plan or task payloads

Not wired in v1: `production_signals`, `reliability` (reserved for a later pass).

`check_machine_criteria()` matches machine-checkable criteria to verification
result rows before any LLM call.

## 8. Judge adaptation

`judge_project_goal()` in `project_goal_eval.py`:

- Same auxiliary route as session goals: `call_llm("goal_judge", …)`,
  `temperature=0`, fail-open `(verdict, reason, parse_failed)`.
- Input is an **evidence digest** (JSON), not chat `last_response`.
- Judge runs only for **manual** criteria after the machine gate passes.
- Subgoals with **only** machine-checkable criteria auto-`achieved` when
  verification passes — no LLM call.

## 9. Re-evaluation loop

- **`reevaluate_project_goal()`** — manual/API path for one subgoal.
- **`goals_tick()`** — all active subgoals for a project (or all projects with
  active subgoals when `project_id` omitted).
- **`maybe_run_goals_tick()`** — called at end of `run_lab_loop_pass()` when
  enabled.

Gate: **`HERMES_DEV_PROJECT_GOALS_TICK=1`** (default off). Idempotent over
already-`achieved` subgoals. Fail-open on judge errors.

## 10. API & CLI surface

### HTTP (`gateway/dev_control/routes.py`)

| Method | Path | Purpose |
|--------|------|---------|
| `GET/POST` | `/v1/dev/goals` | List / create |
| `GET` | `/v1/dev/goals/tree?project_id=…` | Nested hierarchy + rollup |
| `POST` | `/v1/dev/goals/{goal_id}/reevaluate` | Manual re-evaluation |
| `POST` | `/v1/dev/goals/{goal_id}/abandon` | Abandon (no hard delete) |

Auth: same bearer key as other `/v1/dev/*` routes.

### CLI

```bash
hermes dev goals create vision "North star"
hermes dev goals create goal "Q2 theme" --parent-goal-id <vision-id>
hermes dev goals create milestone "Ship v1" --parent-goal-id <goal-id>
hermes dev goals create subgoal "Goals API" --parent-goal-id <milestone-id> \
  --plan-artifact-id <artifact-id> --status active
hermes dev goals list --project-id OrynWorkspace
hermes dev goals tree
hermes dev goals abandon <goal-id>
```

Add `--json` on subcommands for machine-readable output.

### Lab loop

```bash
HERMES_DEV_PROJECT_GOALS_TICK=1 \
  scripts/run_dev_lab_loop.py --max-passes 1
```

Each pass may attach `project_goals_tick` to the lab pass report.

## 11. Phasing (complete)

| Phase | Scope | Status |
|-------|--------|--------|
| 1 | Store, schema, rollup | Done |
| 2 | Routes + CLI | Done |
| 3 | Evidence + machine criteria | Done |
| 4 | Judge + `goals_tick` + lab loop gate | Done |
| 5 | Docs + OpenSpec finalize | Done |

## 12. When Temporal could earn its place (future)

Revisit Temporal only if **milestone execution itself** becomes a complex,
multi-agent, multi-machine orchestration that must be crash-durable across long
horizons with human-in-the-loop approval gates (signals), hard deadlines
(timers), and exactly-once progress — i.e. when cron + kanban + lab_loop are
genuinely outgrown. Even then, the first move is "harden the existing loops"
vs. "adopt Temporal as the execution backend," and that is an execution-layer
decision kept separate from this goal **data model**.

## 13. Testing

```bash
scripts/run_tests.sh \
  tests/gateway/dev_control/test_project_goals.py \
  tests/gateway/dev_control/test_project_goal_eval.py \
  tests/gateway/dev_control/test_project_goals_api.py
```

Covers: CRUD, hierarchy validation, rollup, abandon, persistence, evidence
assembly, machine gate, judge fail-open, tick idempotency, API create/tree.

## 14. Risks & mitigations

- **Judge drift on fuzzy goals.** Lean on `machine_checkable` criteria; manual
  criteria require judge confirmation (or human review of audit trail).
- **Restructuring / orphaning.** `parent_goal_id` reattachable; `abandoned`
  instead of delete; rollup on every structural edit.
- **Cost.** Judge only after machine gate; only leaf subgoals; loop off by default.

## 15. v1 decisions (formerly open questions)

- **Multiple visions:** allowed — no hard "one vision per project" constraint in
  v1; operators manage tree shape by convention.
- **Subgoal ↔ plan linking:** manual via `plan_artifact_id` and/or
  `payload.plan_id` at create time; auto-link from approved artifacts deferred.
- **Dashboard:** API/CLI only in v1; no dashboard widget yet.
- **Weighting:** `payload.weight` supported; default equal weight (1.0).

## 16. Follow-ups (post-v1)

- Auto-create subgoals from approved `plan_artifact` builds.
- Fold `production_signals` / `reliability` into evidence digest.
- Slash commands (`/vision`, `/goal`, …) for gateway parity with session `/goal`.
- Dashboard / project-dashboard read model inclusion.
