## 1. Store, schema, and rollup (deterministic, no LLM)

- [x] 1.1 Create `gateway/dev_control/project_goals.py` with the
  `dev_project_goals` table schema and `DevProjectGoalStore` dataclass
  (`DEFAULT_DB_PATH`, `apply_wal_with_fallback`, `executescript` in
  `__post_init__`), mirroring `DevPlanArtifactStore`.
- [x] 1.2 Implement `create`, `update`, `get`, `list` (filter by `project_id` /
  `kind` / `status` / `parent_goal_id`), and `tree(project_id)`.
- [x] 1.3 Enforce parent rules: `vision` has null parent; all other kinds
  require a non-null `parent_goal_id`.
- [x] 1.4 Normalize/store acceptance criteria via the existing
  `acceptance_criteria.py` v2 helpers.
- [x] 1.5 Implement `recompute_rollup(store, goal_id)`: weighted-mean progress,
  all-achieved → `achieved`, any-blocked → `blocked`, propagate to parent;
  treat `progress` as computed-only.
- [x] 1.6 Implement `abandon(goal_id)` (status + `abandoned_at`, no hard delete)
  and trigger parent rollup.
- [x] 1.7 Unit tests under `tests/gateway/dev_control/test_project_goals.py`
  (CRUD, filtering, tree, rollup leaf/parent/all-achieved/blocked/partial,
  abandon, restart persistence) against a temp DB; run via
  `scripts/run_tests.sh`.

## 2. Control API and CLI

- [x] 2.1 Wire `DevProjectGoalStore` into `gateway/dev_control/routes.py`.
- [x] 2.2 Add routes: `POST /v1/dev/goals`, `GET /v1/dev/goals` (filter),
  `GET /v1/dev/goals/tree`, `POST /v1/dev/goals/{id}/reevaluate`.
- [x] 2.3 Add `hermes dev goals …` CLI subcommands (create/list/tree/abandon).
- [x] 2.4 Route/CLI tests (create → tree shows rolled-up progress end-to-end).

## 3. Evidence assembly and machine-criteria check (deterministic)

- [x] 3.1 Implement `assemble_evidence(subgoal)` pulling from
  `acceptance_verification`, `ci_status.fetch_ci_status`, and `dev_execution`
  task statuses (optionally `production_signals` / `reliability`).
- [x] 3.2 Implement `check_machine_criteria(subgoal, evidence)` evaluating
  `test` / `command` criteria deterministically (reuse the allowlist validator).
- [x] 3.3 Tests against fabricated verification/CI/task rows — no live network.

## 4. Judge and re-evaluation loop (gated, default off)

- [x] 4.1 Implement `judge_project_goal(goal, evidence)` reusing the
  `judge_goal` contract (`call_llm("goal_judge", …)`, `temperature=0`,
  fail-open, parse-failure handling); evidence replaces `last_response`.
- [x] 4.2 Append each verdict + reason to the node's `payload` audit trail.
- [x] 4.3 Implement `goals_tick(store, project_id)`: gate → judge soft criteria →
  on `done` set `achieved` + rollup; idempotent over already-achieved nodes.
- [x] 4.4 Fold `goals_tick` into `gateway/dev_control/lab_loop.py` behind a new
  config flag (default off).
- [x] 4.5 Judge/tick tests with `call_llm` mocked: fail-open, gate-before-judge,
  idempotency, manual-criteria-stay-advisory.

## 5. Docs and finalize

- [x] 5.1 Cross-link `docs/dev-project-goals-spec.md` from the proposal as the
  long-form narrative; reconcile any drift.
- [x] 5.2 Run `openspec validate add-dev-project-goals --strict` clean.
- [x] 5.3 Update relevant `AGENTS.md` / dev_control notes if new conventions land.
