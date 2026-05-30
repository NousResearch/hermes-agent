## Context

`gateway/dev_control/` is a durable, SQLite-backed Dev orchestration layer with
an established store pattern (`DevPlanArtifactStore` in `plan_artifacts.py`) and
an execution spine: `clarification → plan_artifact → execution plan → tasks →
verification/CI`. Evidence is already persisted across `acceptance_verification`,
`ci_status`, `dev_execution`, `production_signals`, and `reliability`. The
session goal-judge (`hermes_cli/goals.py::judge_goal`) is a proven primitive:
`call_llm("goal_judge", …)`, `temperature=0`, returns `(verdict, reason,
parse_failed)`, and is deliberately fail-open with a parse-failure auto-pause
backstop. What is missing is a long-lived intent/progress layer above the
execution spine.

## Goals / Non-Goals

**Goals:**
- A durable, project-scoped goal hierarchy: vision → goal → milestone → subgoal.
- Verifiable acceptance criteria per node, reusing the existing v2 schema.
- Deterministic bottom-up status/progress rollup.
- Judge-based re-evaluation of **leaf subgoals only**, against an evidence
  digest built from existing stores.
- Zero new infrastructure or runtime dependencies.

**Non-Goals:**
- Replacing `plan_artifacts`, `dev_execution`, or kanban (goals link down to
  plans, they do not duplicate task tracking).
- A general PM tool (no Gantt, assignees, or time tracking).
- A new orchestration/execution engine. Execution stays on cron / lab_loop /
  kanban.

## Decisions

### D1: Extend `dev_control`; do not adopt Temporal (v1)

A vision → goal → milestone → subgoal system is fundamentally a **data model +
status rollup + periodic re-evaluation** problem. The persistence substrate
(SQLite via `hermes_state`), the store pattern (`DevPlanArtifactStore`), the
durable re-evaluation hosts (`cron`, `lab_loop`, kanban dispatcher), and the
judgment primitive (`judge_goal`) all already exist.

Temporal solves crash-durable distributed *execution code* — not a goal data
model. Adopting it would require running a Temporal server/cluster (or Temporal
Cloud) plus worker processes and the SDK: a heavyweight, stateful infra
component that conflicts with the repo's self-hostable, dependency-minimal
posture (`AGENTS.md` supply-chain / pinning policy). Rejected for v1.

**Revisit criteria (future):** only if *milestone execution itself* becomes a
complex, multi-agent, multi-machine orchestration that must be crash-durable
across long horizons with human-in-the-loop signal gates, timers, and
exactly-once progress — i.e. when cron + lab_loop + kanban are genuinely
outgrown. Even then, "harden existing loops" is evaluated before "adopt
Temporal," and that is an execution-layer decision kept separate from this goal
data model.

### D2: Leaves are evaluated, parents are rolled up

Only leaf subgoals carry machine-checkable acceptance criteria and a linked
plan; only they are judged. Milestones/goals/vision derive status
deterministically from children. This keeps LLM judgment at the layer with
concrete evidence and keeps the upper tree predictable.

### D3: Evidence digest replaces chat response in the judge

`judge_project_goal()` keeps the `judge_goal()` contract verbatim but swaps the
`last_response` input for an evidence digest assembled by `assemble_evidence()`
from `acceptance_verification`, `ci_status`, `dev_execution` (and optionally
`production_signals` / `reliability`). Machine-checkable criteria are settled
deterministically by `check_machine_criteria()` **before** any LLM call; the
judge only adjudicates soft / `manual` criteria.

### D4: `progress` is computed, never authored

`progress` (0..1) is a cached output of rollup, stored only for cheap reads/UI.
Authors set status/criteria; the system derives progress.

### D5: Re-evaluation lives in `lab_loop`, gated off by default

`goals_tick()` folds into the existing observe-loop (which already pulls
verification/CI/reliability on a tick). Cron is reserved for an optional nightly
digest (cron sessions run `skip_memory=True` with a 3-minute hard interrupt —
wrong for tight orchestration). The tick is idempotent and gated behind
`HERMES_DEV_PROJECT_GOALS_TICK=1` (default off). See
[docs/dev-project-goals-spec.md](../../../docs/dev-project-goals-spec.md) for
operator commands.

## Risks / Trade-offs

- **Judge drift on fuzzy goals.** Mitigation: lean on `machine_checkable`
  criteria; treat `manual` criteria as advisory (surface for human
  confirmation, never auto-flip to `achieved`); carry over the
  parse-failure auto-pause backstop from `goals.py`.
- **Restructuring / orphaning.** Goals get re-parented and abandoned in
  practice. Mitigation: `parent_goal_id` nullable + reattachable; an
  `abandoned` status instead of hard delete (mirrors the curator's
  archive-never-delete instinct); rollup recomputed on every structural edit.
- **Cost.** Judging only leaf subgoals, only after machine criteria pass, bounds
  LLM calls; the loop is off by default.
- **Trade-off vs Temporal.** We accept hand-rolled loop durability (process
  restart re-reads SQLite state on the next tick) instead of engine-guaranteed
  durable execution. Acceptable because goal state is data, not in-flight
  workflow state — there is no long-running orchestration to lose.
