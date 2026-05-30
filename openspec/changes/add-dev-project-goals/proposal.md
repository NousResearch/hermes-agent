## Why

Hermes has a session-scoped goal feature (`hermes_cli/goals.py`): a standing
objective with an auxiliary "judge" model deciding after each turn whether it is
satisfied. It keeps a single session on task but is ephemeral. The Dev workspace
needs the same idea at a **project** scope — a durable spine of vision → goals →
milestones → subgoals that gives every clarification → plan → execution cycle a
stable "why" and tracks progress toward it over time. The execution spine
already exists in `gateway/dev_control/`; the intent/progress layer above it does
not.

## What Changes

- Add a new `dev-project-goals` capability: a durable, hierarchical goal tree
  (vision → goal → milestone → subgoal) persisted in the existing
  `hermes_state` SQLite database.
- Add a `DevProjectGoalStore` (modeled on `DevPlanArtifactStore`) with a new
  `dev_project_goals` table, CRUD + tree reads, and bottom-up status/progress
  rollup.
- Reuse the existing acceptance-criteria v2 schema (`acceptance_criteria.py`) for
  each node's definition of done.
- Adapt the session goal-judge into `judge_project_goal()` — same `call_llm`
  plumbing and fail-open contract, but evaluating an **evidence digest**
  assembled from existing verification/CI/execution stores instead of a chat
  response.
- Add a `goals_tick()` re-evaluation step folded into the existing `lab_loop`
  observe-loop, gated behind a config flag (default off).
- Expose `/v1/dev/goals` routes (create, list/filter, tree, reevaluate) and a
  `hermes dev goals …` CLI surface.
- No new infrastructure or runtime dependencies. **Temporal was evaluated and
  rejected for v1** (see `design.md`).

## Capabilities

### New Capabilities
- `dev-project-goals`: durable project-level goal hierarchy with verifiable
  acceptance criteria, deterministic status rollup, and judge-based
  re-evaluation of leaf subgoals.

### Modified Capabilities
<!-- None. dev_control has no existing OpenSpec specs; this is the first capability captured. -->

## Impact

- **New code:** `gateway/dev_control/project_goals.py` (store + rollup),
  `gateway/dev_control/project_goal_eval.py` (evidence, machine criteria,
  judge, tick), route handlers in `gateway/dev_control/routes.py`, CLI in
  `hermes_cli/dev_goals.py`, tests under `tests/gateway/dev_control/`.
- **Reused, unchanged:** `acceptance_criteria.py`,
  `acceptance_verification.py`, `ci_status.py`, `dev_execution.py`,
  `production_signals.py`, `reliability.py`, `project_scope.py`, `lab_loop.py`
  (one tick step added behind a flag), and the `goal_judge` auxiliary route.
- **Data:** one new SQLite table (`dev_project_goals`) in the existing
  `state.db`; no migration of existing tables.
- **Dependencies:** none added.
- **Config:** `HERMES_DEV_PROJECT_GOALS_TICK` env var gates the automated
  re-evaluation loop (default off).
- **Companion doc:** [docs/dev-project-goals-spec.md](../../../docs/dev-project-goals-spec.md)
  (long-form design narrative, operator reference, v1 decisions).
- **OpenSpec:** `openspec/changes/add-dev-project-goals/` (proposal, design,
  spec scenarios, tasks).
