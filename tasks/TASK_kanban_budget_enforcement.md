# Kanban Task Budget Enforcement (V2.1 - Slice 2/3)

## Mode

`guarded`

## Goal

- Add per-task/run budget enforcement to the existing Kanban dispatcher.
- Keep the feature inert for existing boards unless a task has an explicit
  budget or a new `kanban.task_budget.default_usd` config default applies at
  create time.
- Record worker-run usage/cost from Hermes' existing usage accounting
  (`agent/usage_pricing.py` via `run_conversation()` result fields), persist it
  on `task_runs`, accumulate it on the task, and stop future claims once the
  task budget is exhausted.
- Do not build a new orchestrator, account-budget service, billing API, or
  live provider hook.

## Scope

Allowed:
- Add additive Kanban DB columns for per-task budget cap/spend and per-run
  usage/cost accounting.
- Add config keys under `kanban.task_budget.*` in `hermes_cli/config.py` and
  `cli-config.yaml.example`.
- Add CLI/tool/dashboard create/update surfaces for per-task budget caps.
- Wire dispatcher-spawned workers to write a per-run JSON usage report, using
  existing Hermes session usage fields and cost estimation.
- Add a dispatcher-side usage-report ingestion pass that updates `task_runs`
  and task spend idempotently.
- Add budget gates in the existing Kanban lifecycle so exhausted tasks cannot
  be claimed or respawned.
- Add focused tests for migration, accounting, claim/dispatch gating,
  idempotency, unknown-cost policy, and CLI/tool/dashboard surfaces.

Not allowed:
- Board/day/account/provider-wide budgets.
- Live provider calls, billing API calls, gateway/cron/browser launches, or
  network validation.
- New model tools or a new orchestration engine.
- OpenClaw, new worktree/lock layers, auto-commit, push, or deploy.
- Live per-token/API-call interruption inside `AIAgent.run_conversation()`.
  This slice enforces at Kanban task/run boundaries. A single worker run may
  exceed the cap before the usage report is available; the next claim/spawn is
  blocked.
- Slice 3 skill/auditor orchestration work, including the actual
  `record_plan_audit_verdict()` caller.

## Source Of Truth

- User request: baseline V2.1 after architecture audit: keep Kanban as the
  orchestrator, slice 1 is plan-audit gate, slice 2 is budget enforcement at
  Kanban task/run level using existing usage pricing, slice 3 is the thin skill
  and real auditor caller.
- Repo docs: `AGENTS.md` (Kanban, Footprint Ladder, prompt-cache and config
  rules), `PROJECT_RULES.md` (smallest safe change, validation, handoff).
- Current state verified before writing this runsheet:
  - `agent/usage_pricing.py` already provides `CanonicalUsage`,
    `CostResult`, `normalize_usage()`, and `estimate_usage_cost()`.
  - `agent/turn_finalizer.py` already returns `estimated_cost_usd`,
    `cost_status`, `cost_source`, token counts, model/provider, and session id
    from `run_conversation()`.
  - `hermes_cli/oneshot.py::_write_usage_file()` already writes a JSON usage
    report for `-z --usage-file`.
  - `hermes_cli/_parser.py` currently documents `--usage-file` as one-shot
    only; dispatcher workers currently spawn `hermes -p <profile> chat -q ...`
    from `_default_spawn()`, not `-z`.
  - `claim_task()` is the single enforcement point for `ready -> running`.
  - Worker attempts are tracked in `task_runs`, closed via `_end_run()` on
    `complete_task()`, `block_task()`, timeout/crash/reclaim, and spawn
    failure paths.

## Read Order

1. `AGENTS.md`
2. `PROJECT_RULES.md`
3. `HANDOFF.md`
4. `tasks/TASK_kanban_plan_audit_gate.md`
5. `agent/usage_pricing.py`
6. `agent/turn_finalizer.py`
7. `hermes_cli/oneshot.py`
8. `hermes_cli/_parser.py`
9. `tests/hermes_cli/test_oneshot_usage_file.py`
10. `cli.py` quiet single-query path and `_run_kanban_goal_loop_q()`
11. `hermes_cli/kanban_db.py`: `Task`, `Run`, schema/migrations,
    `create_task()`, `_end_run()`, `claim_task()`, `complete_task()`,
    `block_task()`, `_record_task_failure()`, `_default_spawn()`,
    `_dispatch_once_locked()`, `list_runs()`
12. `hermes_cli/kanban.py` create/show/runs/complete/block CLI paths
13. `tools/kanban_tools.py` create/complete/block schemas and handlers
14. `plugins/kanban/dashboard/plugin_api.py` task create/update/detail payloads
15. Existing tests:
    `tests/hermes_cli/test_kanban_plan_audit_gate.py`,
    `tests/hermes_cli/test_kanban_goal_mode.py`,
    `tests/hermes_cli/test_kanban_db.py`,
    `tests/hermes_cli/test_kanban_cli.py`

## Planned Files

- `hermes_cli/kanban_db.py`
  - Add additive columns on `tasks`, likely:
    `budget_usd REAL`, `budget_spent_usd REAL NOT NULL DEFAULT 0`,
    `budget_unknown_cost_runs INTEGER NOT NULL DEFAULT 0`.
  - Add additive columns on `task_runs`, likely:
    `usage_report_path TEXT`, `usage_report_ingested_at INTEGER`,
    `estimated_cost_usd REAL`, `cost_status TEXT`, `cost_source TEXT`,
    `input_tokens INTEGER`, `output_tokens INTEGER`,
    `cache_read_tokens INTEGER`, `cache_write_tokens INTEGER`,
    `reasoning_tokens INTEGER`, `total_tokens INTEGER`, `api_calls INTEGER`,
    `model TEXT`, `provider TEXT`, `usage_session_id TEXT`.
  - Extend `Task` and `Run` dataclasses/from-row parsing.
  - Extend `create_task()` with `budget_usd`.
  - Add helpers such as `_effective_task_budget_usd()`,
    `_budget_spent_usd()`, `_record_run_usage_report()`,
    `_ingest_worker_usage_reports()`, and `_block_task_budget_exhausted()`.
  - Add a budget gate inside `claim_task()` after parent/plan-audit gates and
    before the CAS that transitions to `running`.
  - Call usage-report ingestion at the start of `_dispatch_once_locked()` after
    `reap_worker_zombies()` and before ready-row selection.
  - Include budget data in `list_runs()`/`latest_run()` via `Run`.
  - Add a `DispatchResult` bucket if useful, e.g. `budget_exhausted`.
- `cli.py`
  - Treat as high-risk/god-file surface. Investigate only after proving the
    lower-footprint `_default_spawn()` + `-z --usage-file` path cannot preserve
    worker behavior.
  - If `cli.py` must be touched, implement the smallest internal usage-report
    path: `_default_spawn()` sets an env path (for example
    `HERMES_KANBAN_USAGE_FILE`) and quiet `chat -q` writes the report after
    the entire worker flow, including `_run_kanban_goal_loop_q()`.
  - Do not write the usage report before goal-mode continuation turns finish.
- `hermes_cli/oneshot.py`
  - Reuse or extract `_write_usage_file()` rather than duplicating JSON report
    formatting in `cli.py`. If importing a private helper becomes awkward,
    move the helper to a small shared module instead of adding new agent-core
    surface.
- `hermes_cli/kanban.py`
  - Add `kanban create --budget-usd USD` and, if config default budgets are
    supported, `--no-budget`.
  - Show budget cap/spend/remaining in `kanban show` and include budget fields
    in JSON.
  - Include cost fields in `kanban runs` JSON/text output.
  - Do not hide the existing plan-audit warning.
- `tools/kanban_tools.py`
  - Add `budget_usd` to `kanban_create` schema/handler.
  - Tool description must warn this is a per-task cap and that exhausted tasks
    stop being claimed; it is not a board/day/account budget.
  - Optionally include budget fields in `kanban_show` output.
- `plugins/kanban/dashboard/plugin_api.py`
  - Add budget fields to task dicts and create/update payloads.
  - Include run cost fields in `_run_dict()`.
- `hermes_cli/config.py`
  - Add nested config, likely:
    `kanban.task_budget.default_usd: None`
    `kanban.task_budget.unknown_cost_policy: "allow"`
  - Keep `.env` out of this; this is behavioral config.
- `cli-config.yaml.example`
  - Document `kanban.task_budget.*`.
- `tests/hermes_cli/test_kanban_budget_enforcement.py` (new)
- `tests/hermes_cli/test_oneshot_usage_file.py` when `_write_usage_file()` is
  reused, moved, or otherwise touched.
- Possibly `tests/hermes_cli/test_kanban_goal_mode.py` and
  `tests/hermes_cli/test_kanban_cli.py` for focused regression coverage.

## Mini-Plan

- Files to inspect/edit: planned files above.
- Commands to run:
  - targeted `rg`/file reads for every planned symbol before editing
  - `python -m py_compile` on changed Python files
  - targeted pytest for the new budget tests and nearby Kanban tests
- Live provider/gateway/cron/browser state touched: no.
- Stop conditions:
  - First investigate the lower-footprint route: switch worker spawn to
    top-level `-z --usage-file` from `_default_spawn()`. If that preserves
    Kanban worker behavior, prefer it because it avoids editing `cli.py`.
    If it breaks goal mode, rate-limit sentinel exit codes, image extraction
    from task bodies, session id handling, or toolset/profile resolution, keep
    `chat -q` and consider the internal env usage-report path instead.
  - If quiet `chat -q` cannot produce a complete usage report after goal-mode
    continuation turns without invasive `AIAgent` loop changes, stop and report
    before implementing a partial budget feature.
  - If accounting cannot be made idempotent across dispatcher ticks and process
    restarts, stop before writing claim/spawn gates.
  - If switching worker spawn to `-z` breaks Kanban-specific quiet path
    behavior (goal mode, rate-limit sentinel exit code, image extraction from
    task body, session id handling, toolset/profile resolution), keep `chat -q`
    and implement the internal usage-report path instead.
  - If the same validation failure repeats twice with the same signature,
    follow the Repeated-Blocker Rule.
- Validation plan:
  - Unit/DB tests for additive migrations, budget defaults, explicit budget
    create, idempotent report ingestion, and exhausted-budget claim rejection.
  - Dispatcher tests showing an exhausted budgeted ready task is not spawned.
  - Worker-spawn test showing the usage-report path is pinned to the current
    run id and board, and that no live provider call is required.
  - Goal-mode regression: usage report is written after `_run_kanban_goal_loop_q`
    so multi-turn goal workers are not undercounted.
  - CLI/tool/dashboard surface tests for create/show payloads.

## Safety Audit

- No destructive command without approval: yes.
- No live external/provider side effect without approval: yes.
- No secrets printed or committed: yes.
- No generated/local/runtime files staged: usage reports created by tests must
  live under temp dirs only and must not be staged.
- No unrelated dirty state hidden: run `git status --short --branch` before and
  after. Known unrelated dirty state at runsheet creation time:
  `.gitignore`, `apps/desktop/src/app/shell/model-menu-panel.tsx`, and local
  docs/tasks/reports files. Do not revert or fold unrelated changes into this
  slice.

## Validation Commands

- `python -m py_compile hermes_cli/kanban_db.py hermes_cli/kanban.py tools/kanban_tools.py plugins/kanban/dashboard/plugin_api.py hermes_cli/config.py cli.py hermes_cli/oneshot.py tests/hermes_cli/test_kanban_budget_enforcement.py`
- Preferred CI-parity command when a POSIX venv exists:
  `bash scripts/run_tests.sh tests/hermes_cli/test_kanban_budget_enforcement.py -q`
- Current checkout fallback, matching prior slice validation caveat:
  `uv run --with pytest python -m pytest tests/hermes_cli/test_kanban_budget_enforcement.py -q`
- Focused regression:
  `uv run --with pytest python -m pytest tests/hermes_cli/test_kanban_budget_enforcement.py tests/hermes_cli/test_oneshot_usage_file.py tests/hermes_cli/test_kanban_plan_audit_gate.py tests/hermes_cli/test_kanban_goal_mode.py tests/hermes_cli/test_kanban_cli.py -q`
- Before `DONE` if environment is fixed:
  `bash scripts/run_tests.sh tests/hermes_cli/test_kanban_budget_enforcement.py tests/hermes_cli/test_oneshot_usage_file.py tests/hermes_cli/test_kanban_plan_audit_gate.py tests/hermes_cli/test_kanban_goal_mode.py tests/hermes_cli/test_kanban_cli.py -q`
- `git diff --check -- hermes_cli/kanban_db.py hermes_cli/kanban.py tools/kanban_tools.py plugins/kanban/dashboard/plugin_api.py hermes_cli/config.py cli-config.yaml.example cli.py hermes_cli/oneshot.py tests/hermes_cli/test_kanban_budget_enforcement.py tests/hermes_cli/test_oneshot_usage_file.py tasks/TASK_kanban_budget_enforcement.md HANDOFF.md`

## Attempt Ledger

Attempt 1:
- Goal: implement slice 2 task/run budget enforcement without changing the
  Kanban orchestration shape.
- Change or action:
  - Investigated the lower-footprint `-z --usage-file` route and rejected it
    for dispatcher workers because `oneshot.py` does not preserve the existing
    Kanban worker behavior (`chat -q` task-body/image handling, goal-loop mode,
    rate-limit sentinel exit handling, session/output semantics, and
    profile/toolset resolution).
  - Kept `_default_spawn()` on `hermes -p <profile> chat -q ...`, added an
    internal `HERMES_KANBAN_USAGE_FILE` env path, and wrote the usage report
    from the quiet `chat -q` path after any Kanban goal-loop continuation
    turns finish.
  - Added additive `tasks` budget columns and `task_runs` usage/cost columns,
    idempotent usage-report ingestion, a dispatcher ingestion pass, and budget
    gates for both ready-task claims and review-task claims.
  - Added CLI/tool/dashboard create/read surfaces for per-task budget caps and
    run cost fields.
  - Added `tests/hermes_cli/test_kanban_budget_enforcement.py`.
- Validation command:
  - `python -m py_compile hermes_cli/kanban_db.py hermes_cli/kanban.py tools/kanban_tools.py plugins/kanban/dashboard/plugin_api.py hermes_cli/config.py cli.py hermes_cli/oneshot.py tests/hermes_cli/test_kanban_budget_enforcement.py tests/hermes_cli/test_oneshot_usage_file.py`
  - `py -3.12 -m pytest tests/hermes_cli/test_kanban_budget_enforcement.py -q`
  - `py -3.12 -m pytest tests/hermes_cli/test_kanban_budget_enforcement.py tests/hermes_cli/test_oneshot_usage_file.py tests/hermes_cli/test_kanban_plan_audit_gate.py tests/hermes_cli/test_kanban_goal_mode.py -q`
  - `python -c "import tools.kanban_tools as kt; assert 'budget_usd' in kt.KANBAN_CREATE_SCHEMA['parameters']['properties']; print('tools kanban schema ok')"`
  - `python -c "import plugins.kanban.dashboard.plugin_api as api; print(api.CreateTaskBody.__name__)"`
  - `python` assertion for `_kanban_usage_result_from_agent()`
  - `git diff --check -- ...`
- Result:
  - py_compile: pass.
  - New budget tests: 12 passed, 1 skipped. The skipped test imports
    `cli.py` under Python 3.12, which lacks `yaml`; the same helper was
    separately asserted with the managed Hermes `python` that has `yaml`.
  - Focused regression: 38 passed, 1 skipped.
  - Tool schema import/check: pass.
  - Dashboard plugin import/check: pass.
  - `git diff --check`: pass, with only CRLF warnings from Windows.
  - `bash scripts/run_tests.sh tests/hermes_cli/test_kanban_budget_enforcement.py -q`:
    failed with `error: no virtualenv found in /mnt/d/OneDrive/Hermes/.venv or
    /mnt/d/OneDrive/Hermes/venv`.
- Post-audit follow-up:
  - External audit approved the slice with one actionable minor: negative
    `estimated_cost_usd` in a malformed usage report could reduce
    `budget_spent_usd`.
  - Fixed by clamping negative ingested run cost to `0.0` before persisting the
    run cost or adding task spend.
  - Added coverage for negative-cost clamping and config-driven
    `unknown_cost_policy` resolution without relying on a PyYAML import.
  - Re-validation:
    `py -3.12 -m pytest tests/hermes_cli/test_kanban_budget_enforcement.py -q`
    -> 14 passed, 1 skipped.
    `py -3.12 -m pytest tests/hermes_cli/test_kanban_budget_enforcement.py tests/hermes_cli/test_oneshot_usage_file.py tests/hermes_cli/test_kanban_plan_audit_gate.py tests/hermes_cli/test_kanban_goal_mode.py -q`
    -> 40 passed, 1 skipped.
    `python -m py_compile hermes_cli/kanban_db.py tests/hermes_cli/test_kanban_budget_enforcement.py`
    -> pass.
    `git diff --check -- hermes_cli/kanban_db.py tests/hermes_cli/test_kanban_budget_enforcement.py tasks/TASK_kanban_budget_enforcement.md HANDOFF.md`
    -> pass, with only CRLF warnings from Windows.
- Artifact/log: none beyond the test output in this session.
- Error signature:
  - `python -m pytest ...` cannot run in the managed Hermes venv because it has
    `yaml` but no `pytest`.
  - `py -3.12` has `pytest` but lacks `yaml`, `prompt_toolkit`, and some app
    dependencies, so `tests/hermes_cli/test_kanban_cli.py` still fails in this
    environment for dependency reasons.
  - The broad `tests/hermes_cli/test_kanban_db.py` run produced 209 passed /
    14 failed, matching existing Windows/platform-test drift (raw wait-status
    classification, `os.waitpid` reaper no-op on Windows, Git worktree slash
    formatting, and managed shim resolution), not this budget slice.
- Hypothesis after result: slice 2 behavior is implemented and validated in
  the focused DB/dispatcher/usage-report paths. Full CI-parity still needs a
  clean dev environment with pytest plus project extras installed together.

Attempt 2:
- Goal:
- Change or action:
- Validation command:
- Result:
- Artifact/log:
- Error signature:
- Hypothesis after result:

## Execution Log

- Runsheet created for slice 2 only.
- Slice 2 executed:
  - Budget fields on `Task` and run usage/cost fields on `Run`.
  - Additive schema/migration support for fresh and legacy boards.
  - `create_task()` explicit/default/no-budget behavior.
  - Dispatcher usage-report path wiring and idempotent ingestion.
  - Budget gate at `claim_task()` for `ready -> running`.
  - Budget gate at `claim_review_task()` for `review -> running`, so review
    agents cannot continue spending a task/run chain after the cap is reached.
  - CLI/tool/dashboard budget and run-cost surfaces.
- Slice 3 was not implemented.

## Final Report Checklist

- Status reported
- Changed files listed
- Commands and validation results listed
- Artifacts listed, or `none`
- Handoff update noted
- Remaining risk noted

---

## Scope Notes

This is **slice 2/3** of the V2.1 Kanban-first baseline:

1. Slice 1: plan-audit gate in `claim_task()` - implemented and approved.
2. Slice 2: this runsheet - task/run budget enforcement.
3. Slice 3: "Kanban Orchestrated Coding" skill plus the real auditor caller
   that records `record_plan_audit_verdict()` events.

Do not merge slice 2 with slice 3. Budget accounting touches dispatcher spawn,
worker exit, and task-run persistence; that is already enough blast radius for
one guarded slice.
