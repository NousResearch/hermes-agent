# Hermes Local Handoff

Canonical local-state handoff for `D:\OneDrive\Hermes`.

## Workspace

- Path: `D:\OneDrive\Hermes`
- Upstream: `https://github.com/NousResearch/hermes-agent`
- Branch: `main`
- Cloned on: 2026-07-09
- Current user goal: build Hermes into a persistent, cost-aware AI
  orchestration system that turns high-level goals into verified outcomes by
  coordinating replaceable AI workers, long-running processes, isolated Git
  workflows, bounded retries, evidence-based verification, and selective
  escalation to expensive expert models.

## Current Status

- Repo has been cloned from upstream into `D:\OneDrive\Hermes`.
- Upstream `AGENTS.md` already exists and is the main project development guide.
- Local north star is now explicit: Hermes is the boss/orchestrator above
  replaceable worker AIs, not a coding-agent replacement hard-coded around
  Claude, Codex, DeepSeek, OpenCode, or any single provider.
- Local operator workflow files added:
  - `CLAUDE.md` local startup index, ignored by git
  - `PROJECT_RULES.md`
  - `PROJECT_STRUCTURE.md`
  - `HANDOFF.md`
  - `docs/ai/workflows/guarded-task-loop.md`
  - `tasks/TASK_RUNSHEET_TEMPLATE.md`
  - `reports/FINAL_REPORT_TEMPLATE.md`
- `.gitignore` updated to ignore `CLAUDE.md` and `.ai-runs/`.

## Setup Findings

- Hermes supports native Windows and WSL2.
- Native Windows installer:
  `iex (irm https://hermes-agent.nousresearch.com/install.ps1)`
- Source/dev requirements:
  - Python `>=3.11,<3.14`
  - Node `>=20`
  - `uv`
  - `rg` recommended
- Python tests should be run through:
  `scripts/run_tests.sh`

## Safety Notes

- Do not commit `.env`, API keys, OAuth state, bot tokens, `~/.hermes` runtime state, logs, local sessions, `.ai-runs/`, venvs, `node_modules`, or generated dist folders.
- Do not run live messaging gateway, browser automation, cron jobs, or provider/API calls unless explicitly requested.
- Do not replace upstream `AGENTS.md` with local Gmail automation rules; keep Hermes-specific upstream guidance authoritative.

## Build Direction

- Implementation path remains Kanban-first V2.1: use Hermes Kanban as the
  workflow engine, not a new orchestrator database or workflow daemon.
- Cheap models should do routine worker labor; expensive models should be used
  selectively for expertise, audit, and escalation.
- Prioritize persistent state, long-running process supervision, model routing,
  bounded retry, Git worktree/branch isolation, verification evidence,
  commit/merge control, audit, cost accounting, and final reporting.
- Next build priority: clean up and audit the current Kanban/live-E2E diff,
  then continue only toward remaining persistent-orchestration gaps.

## Recommended Next Plan

1. Decide install target:
   - Native Windows managed install for daily use.
   - WSL2/Linux install if you prefer Linux shell tooling.
   - Source-dev venv for contributing to Hermes code.
2. Run dependency/bootstrap checks:
   - `git status --short --branch`
   - check `python --version`, `uv --version`, `node --version`, `npm --version`, `rg --version`
3. Install/configure Hermes:
   - native Windows: run official PowerShell installer
   - source dev: create venv and install `.[all,dev]`
4. Configure model/tool provider via `hermes setup`, `hermes model`, `hermes tools`.
5. Validate with `hermes doctor` and a small CLI smoke test.

## Open Questions

- Which runtime path should be primary: native Windows, WSL2, Docker, or source-dev only?
- Which provider should be configured first: Nous Portal, OpenRouter, OpenAI-compatible endpoint, local/Ollama, or another provider?
- Which Hermes surfaces matter first: CLI/TUI only, gateway messaging, cron, browser tools, skills, or desktop?

## Active Work: Orchestration Architecture (V2.1)

- Context: Codex proposed a new "personal AI orchestrator" layer for Hermes
  (planner/auditor/executor roles, task state machine, model router, SQLite
  task store). Audited in-session as an independent architecture review.
- Outcome: **v1 REJECTED** — it duplicated an existing, shipped, tested
  subsystem (Kanban: `plugins/kanban/`, `hermes_cli/kanban*.py`,
  `docs/hermes-kanban-v1-spec.pdf`) plus `delegate_task`, `auxiliary.*` model
  routing, and `tools/approval.py`, none of which the v1 plan referenced.
- **v2 (Kanban-first) → APPROVED WITH CHANGES**, then **v2.1 → chốt làm
  baseline**. Direction: no new orchestrator/DB/engine. Extend Kanban with
  exactly two new pieces — (1) a plan-audit gate before a task can claim into
  `running`, (2) budget enforcement at the Kanban task/run level — plus a thin
  skill/config layer tying phases together. Full reasoning is in the chat
  transcript of this session, not duplicated here.
- Artifact created: [tasks/TASK_kanban_plan_audit_gate.md](tasks/TASK_kanban_plan_audit_gate.md)
  — detailed runsheet for slice 1/3 (plan-audit gate in `claim_task()`,
  `hermes_cli/kanban_db.py:3372`). **Not yet executed.**
- Next actions (not started):
  1. Execute slice 1 runsheet above, or write it out further if gaps appear
     during implementation.
  2. Write slice 2 runsheet — budget enforcement (`kanban.task_budget.*`,
     built on `agent/usage_pricing.py`).
  3. Write slice 3 runsheet — "Kanban Orchestrated Coding" skill, reusing
     `hermes kanban specify` / `auxiliary.triage_specifier` for intake/spec.
- Constraint carried into all three slices: no OpenClaw in-tree, no new
  worktree/lock mechanism (reuse `resolve_workspace()`,
  `hermes_cli/kanban_db.py:5505`), no auto-commit/push/deploy, per-task budget
  only (no board/day/account budget) in this baseline.

## Active Work Update: Slice 1 Executed

- Slice 1 implemented: plan-audit gate now lives in `claim_task()` via per-task
  `plan_audit_required` / `plan_audit_max_rounds`, event verdicts
  `plan_audit_requested|approved|rejected|exhausted`, and helper
  `record_plan_audit_verdict()`. CLI/tool/dashboard create paths can opt a task
  in. `auxiliary.plan_auditor` config is registered, but the actual LLM caller
  remains for slice 3/skill work so no provider call happens under the DB lock.
- CLI help/show output and `kanban_create` tool schema now warn that gated tasks
  remain ready/unclaimed until an auditor caller records verdict events.
- Artifact updated: [tasks/TASK_kanban_plan_audit_gate.md](tasks/TASK_kanban_plan_audit_gate.md)
  records implementation details and validation.
- Target validation:
  `uv run --with pytest python -m pytest tests/hermes_cli/test_kanban_plan_audit_gate.py -q`
  -> 8 passed.
- Focused regression:
  `uv run --with pytest python -m pytest tests/hermes_cli/test_kanban_plan_audit_gate.py tests/hermes_cli/test_kanban_goal_mode.py tests/hermes_cli/test_kanban_promote.py tests/hermes_cli/test_kanban_cli.py -q`
  -> 83 passed.
- Validation caveat: `bash scripts/run_tests.sh ...` cannot see the Windows
  `.venv` from WSL (`no virtualenv found`). Broader native-Windows `uv run`
  regressions still show environment/platform failures around `os.waitpid`,
  the POSIX `true` command, uv shim resolution, Git worktree slash formatting,
  and cached profile-home assumptions; these are not tied to the plan-audit gate.
- Next actions: execute the slice 2 budget-enforcement runsheet, or write
  the slice 3 runsheet if budget implementation should wait. Slice 3 owns the
  "Kanban Orchestrated Coding" skill and the caller that uses
  `auxiliary.plan_auditor` outside `claim_task()` before recording verdicts.

## Active Work Update: Slice 2 Runsheet Written

- Artifact created: [tasks/TASK_kanban_budget_enforcement.md](tasks/TASK_kanban_budget_enforcement.md).
- Status: superseded by execution update below.
- Important implementation decision captured in the runsheet: dispatcher
  workers currently spawn as `hermes -p <profile> chat -q ...`, while
  `--usage-file` is currently documented and wired for one-shot `-z` only.
  Investigate switching `_default_spawn()` to `-z --usage-file` first because
  it may avoid editing high-risk `cli.py`; keep `chat -q` plus an internal
  env usage-report path only if `-z` breaks Kanban worker behavior, rate-limit
  sentinel exit codes, image extraction from task bodies, toolset/profile
  resolution, session id handling, or goal-mode behavior.
- Validation note: include `tests/hermes_cli/test_oneshot_usage_file.py` in
  slice 2 regressions if `_write_usage_file()` is reused, moved, or changed.
- Planned enforcement shape: persist budget cap/spend on `tasks`, persist
  usage/cost on `task_runs`, ingest per-run usage reports idempotently at
  dispatcher/task-run boundaries, and block future `ready -> running` claims
  once a task budget is exhausted.
- Current next actions: see slice 2 execution update below.

## Active Work Update: Slice 2 Executed

- Slice 2 implemented: Kanban now has per-task budget caps/spend on `tasks`,
  per-run usage/cost fields on `task_runs`, idempotent usage-report ingestion,
  and task/run-boundary budget gates. Existing boards/tasks remain inert unless
  a task has `budget_usd` or newly-created tasks inherit
  `kanban.task_budget.default_usd`.
- Implementation decision: the attempted lower-footprint `-z --usage-file`
  route was rejected because one-shot mode does not preserve Kanban worker
  behavior. Dispatcher workers stay on `hermes -p <profile> chat -q ...`;
  `_default_spawn()` sets `HERMES_KANBAN_USAGE_FILE`, and the quiet chat path
  writes the report after any Kanban goal-loop turns finish.
- Enforcement points:
  - `claim_task()` blocks `ready -> running` when spend reaches the task budget
    or when `unknown_cost_policy=block` and an unknown-cost run was recorded.
  - `claim_review_task()` now applies the same budget gate to `review ->
    running`, so review agents cannot continue a capped task/run chain.
  - `_dispatch_once_locked()` ingests pending worker usage reports before ready
    recomputation and spawn selection.
- Surfaces updated: `kanban create --budget-usd/--no-budget`, `kanban show`,
  `kanban runs`, `kanban_create` tool schema/handler/show output, dashboard
  task create payload, dashboard run dict, `hermes_cli/config.py`, and
  `cli-config.yaml.example`.
- Tests added: [tests/hermes_cli/test_kanban_budget_enforcement.py](tests/hermes_cli/test_kanban_budget_enforcement.py).
- Validation:
  - `python -m py_compile hermes_cli/kanban_db.py hermes_cli/kanban.py tools/kanban_tools.py plugins/kanban/dashboard/plugin_api.py hermes_cli/config.py cli.py hermes_cli/oneshot.py tests/hermes_cli/test_kanban_budget_enforcement.py tests/hermes_cli/test_oneshot_usage_file.py` -> pass.
  - `py -3.12 -m pytest tests/hermes_cli/test_kanban_budget_enforcement.py -q` -> 12 passed, 1 skipped. Skip is the `cli.py` helper import under Python312, which lacks `yaml`; the helper was separately asserted with managed Hermes `python`.
  - `py -3.12 -m pytest tests/hermes_cli/test_kanban_budget_enforcement.py tests/hermes_cli/test_oneshot_usage_file.py tests/hermes_cli/test_kanban_plan_audit_gate.py tests/hermes_cli/test_kanban_goal_mode.py -q` -> 38 passed, 1 skipped.
  - Managed `python` import checks for `tools.kanban_tools`, `plugins.kanban.dashboard.plugin_api`, and `_kanban_usage_result_from_agent()` -> pass.
  - `git diff --check -- ...` -> pass, only Windows CRLF warnings.
  - `bash scripts/run_tests.sh tests/hermes_cli/test_kanban_budget_enforcement.py -q` -> failed with `error: no virtualenv found in /mnt/d/OneDrive/Hermes/.venv or /mnt/d/OneDrive/Hermes/venv`.
- Post-audit follow-up:
  - External audit approved slice 2 and flagged one non-blocking hardening
    issue: negative `estimated_cost_usd` in a malformed usage report could
    reduce `budget_spent_usd`.
  - Fixed by clamping negative ingested run cost to `0.0`; added tests for
    that and for config-driven `unknown_cost_policy` without depending on the
    local PyYAML availability.
  - Re-validation after the fix:
    `py -3.12 -m pytest tests/hermes_cli/test_kanban_budget_enforcement.py -q`
    -> 14 passed, 1 skipped.
    Focused regression (`budget`, `oneshot_usage_file`, `plan_audit_gate`,
    `goal_mode`) -> 40 passed, 1 skipped.
- Broader validation caveat:
  - Managed Hermes `python` has `yaml` but no `pytest`; Python312 has
    `pytest` but lacks `yaml`, `prompt_toolkit`, and some app deps. As a
    result, `test_kanban_cli.py` still fails in this checkout for dependency
    reasons.
  - A broad `test_kanban_db.py` run produced 209 passed / 14 failed from
    existing Windows/platform-test drift (raw wait-status classification,
    `os.waitpid` reaper no-op on Windows, Git worktree slash formatting, and
    managed shim resolution), not from the budget slice.
- Remaining V2.1 work: Slice 3 is still not started. It owns the "Kanban
  Orchestrated Coding" skill and the real caller that uses
  `auxiliary.plan_auditor` outside `claim_task()` before recording
  `record_plan_audit_verdict()` events.
- Environment debt before slice 3 or final CI confidence: create a source-dev
  venv with pytest plus project extras (`.[all,dev]`) or restore `uv` on PATH
  so validation can match CI instead of splitting dependency coverage across
  two interpreters.

## Active Work Update: Architecture Plan Reset

- User clarified the original goal again: this work started as an architecture
  design request for "Hermes as a personal AI Chief of Staff"; the original
  request explicitly said to plan architecture only and not write code yet.
- Artifact created:
  [docs/ai/hermes-personal-orchestration-architecture.md](docs/ai/hermes-personal-orchestration-architecture.md).
- The document resets the architecture baseline to Kanban-first V2.1:
  no new orchestrator, no second workflow DB, no new workflow engine. Hermes
  Kanban is the workflow engine; profiles/skills/config provide the thin
  orchestration layer; OpenClaw and external CLIs are optional adapters/lanes.
- It separates existing Hermes mechanisms, config/skill work, small code
  patches, and later/non-MVP work. It also captures feasibility, pushback on
  ambiguous requirements, component/data flow, workflow/router/audit-loop
  design, storage/memory, safety, MVP, roadmap, acceptance criteria, tests,
  risks, and user decisions required before more code.
- Treat slice 1/2 implementation work above as separate local product patches,
  not as the architecture deliverable itself. The next implementation work, if
  approved, should be slice 3 only: "Kanban Orchestrated Coding" skill plus the
  real plan-auditor caller that records verdicts outside the dispatcher claim
  lock.

## Active Work Update: Architecture Audit Follow-up

- Claude audited
  [docs/ai/hermes-personal-orchestration-architecture.md](docs/ai/hermes-personal-orchestration-architecture.md)
  and approved the Kanban-first direction, but flagged blockers in the slice 3
  design: missing loop actuator, verdict vocabulary exceeding the current
  binary `record_plan_audit_verdict()` primitive, happy-path-only tests,
  missing `review`/`scheduled` states in the diagram, workflow-level cost
  exposure, config namespace ambiguity, and detached-worker HITL risk.
- The architecture doc was revised to close those findings:
  - MVP actuator is now explicitly a worker-embedded plan-auditor task, not a
    dispatcher hook or polling daemon.
  - MVP verdicts stay binary (`approved`/`rejected`); `needs_user_decision`
    and `changes_requested` are represented as rejected-plus-comment/block.
  - Rejection now creates the next planner-revision task or blocks at
    `max_rounds`.
  - State machine now documents existing `review` and `scheduled` statuses.
  - Budget section now requires a visible workflow cost ceiling even if
    subtree roll-up enforcement is not yet implemented.
  - Config section now separates role-to-profile mapping
    (`kanban_orchestration.roles.*`) from secondary model-call binding
    (`auxiliary.plan_auditor`).
  - HITL now says detached dispatcher workers must block/comment/notify
    asynchronously, not wait on interactive `tools.approval.py` prompts.
  - Test strategy now requires reject -> revise -> approve, max-rounds block,
    idempotent graph creation, budget-ceiling, and detached-HITL tests.
- Next slice 3 implementation should follow the revised doc, especially the
  worker-embedded plan-auditor actuator and binary-verdict constraint.

## Active Work Update: Architecture Audit Round 2 Follow-up

- Claude audit round 2 confirmed the prior 10 findings were genuinely closed.
  It flagged one remaining slice-3 blocker and three small cleanup items:
  - M5: clarify that `record_plan_audit_verdict()` must target the task that
    carries `plan_audit_required` (the gated executor task), not the auditor
    task.
  - L4: fix the architecture flowchart so detached workers do not appear to
    use interactive `tools.approval.py` directly.
  - L5: require round-scoped idempotency keys for planner/auditor revision
    tasks created after a rejected plan audit.
  - L6: give plan-audit rejections structured `rejected.kind` metadata so the
    actuator can distinguish `revise_plan` from `needs_user_decision`.
- The architecture doc was patched accordingly:
  - §7.5 now states the plan-audit flag and verdict event live on the same
    gated executor task, with `record_plan_audit_verdict(executor_task_id, ...)`.
  - §6 flowchart now routes worker HITL through Kanban blocked/comment/notify
    and labels `tools.approval.py` as interactive/gateway-surface only.
  - §7.5 and §17 now require idempotent revision creation per root workflow,
    executor task, and round.
  - §7.5 now allows plan-audit rejection metadata such as
    `rejected.kind = "revise_plan"` or `"needs_user_decision"`.
  - §20 now tells slice 3 implementers to record verdicts on the executor task
    id and use round-scoped idempotency keys for revision tasks.
- With those doc fixes, the architecture spec is ready to use as the slice 3
  planning baseline, pending user approval to move from documentation into code.

## Active Work Update: Slice 3 Runsheet Written

- Artifact created:
  [tasks/TASK_kanban_orchestrated_coding_slice3.md](tasks/TASK_kanban_orchestrated_coding_slice3.md).
- Status: planning/runsheet only; no runtime code has been implemented for
  slice 3 in this step.
- Scope: "Kanban Orchestrated Coding" skill plus the worker-embedded
  plan-auditor actuator. The runsheet keeps Kanban as the workflow engine and
  explicitly forbids a dispatcher hook, polling daemon, second workflow DB, new
  core model tool, OpenClaw/Codex/Claude external lane implementation, or live
  provider calls in tests.
- Critical implementation constraints captured:
  - `record_plan_audit_verdict()` must target the gated executor task id, not
    the auditor task id.
  - A rejected audit creates planner/auditor revision tasks with round-scoped
    idempotency keys.
  - `max_rounds` must block with `needs_input`, not loop.
  - Detached worker HITL must use Kanban comment/block/notify, not interactive
    `tools.approval.py`.
  - If a worker needs a verdict-recording surface, prefer the narrowest
    existing-Kanban-toolset addition; do not add a new core tool.
- Next action, if user approves implementation: execute this runsheet, starting
  with tests for the three acceptance gates called out at the bottom of the
  runsheet.

## Active Work Update: Slice 3 Runsheet Audit Follow-up

- External audit verified the slice 3 runsheet's code-grounding claims:
  `link_tasks()` really demotes `ready -> todo` when adding an unfinished
  parent; `create_task()` and `kanban_create` expose the expected idempotency,
  plan-audit, budget, and status fields; all Read Order files exist.
- Audit found one real correctness gap in the runsheet's replay story:
  `record_plan_audit_verdict()` currently appends verdict events
  unconditionally, while `_plan_audit_rejections_since_approval()` counts
  rejected events. If the actuator records a rejected verdict and crashes before
  creating revision tasks, replaying the same round could append a second
  rejected event and double-count toward `max_rounds`.
- Runsheet was patched to require:
  - DB-contract tests before choosing the worker-facing verdict surface.
  - Idempotent verdict recording by `(executor_task_id, round)`, not just
    idempotent revision task creation.
  - Replay assertions that rejected-round count and executor status stay
    correct, not only that revision task count is stable.
  - A promotion-race test: executor may be `ready` before approval but still
    must not claim.
  - Auditor actuator task must complete itself after recording verdict and
    applying graph mutation.
- Before implementing slice 3, start with those DB-contract tests. Do not write
  a worker-facing verdict tool first.

## Active Work Update: Slice 3 Executed

- Status: slice 3 implementation complete for the Kanban-first V2.1 MVP; ready
  for external audit before commit.
- Implemented runtime pieces:
  - `hermes_cli/kanban_db.py`
    - `record_plan_audit_verdict()` is now idempotent by
      `(task_id, verdict kind, metadata.round)`.
    - `_plan_audit_rejections_since_approval()` counts unique keyed rejected
      rounds and preserves raw counting for unkeyed legacy events.
    - Added `apply_plan_audit_actuation()` as the worker-embedded
      plan-auditor actuator primitive. It records verdicts on the gated
      executor task id, creates round-scoped revision planner/auditor cards,
      blocks `needs_input` / max-round cases, and completes the current
      auditor task.
    - Post-audit polish: plan-audit blocking preserves the existing
      `block_recurrences` loop-breaker signal and replayed actuator comments
      are idempotent by task/author/body.
  - `tools/kanban_tools.py`
    - Added `kanban_apply_plan_audit_actuation` under the existing gated
      Kanban toolset.
    - Kept lower-level `kanban_record_plan_audit_verdict` for direct verdict
      recording, with explicit executor-task-id guardrails.
  - `toolsets.py` and `agent/transports/hermes_tools_mcp_server.py`
    - Exposed the new actuator tool only through the existing Kanban
      check_fn-gated surface / Codex-runtime MCP worker callback.
  - `skills/software-development/kanban-orchestrated-coding/SKILL.md`
    - Added MVP workflow contract and updated it to prefer
      `kanban_apply_plan_audit_actuation(...)`.
- Tests added/updated:
  - `tests/hermes_cli/test_kanban_orchestrated_coding.py`
  - `tests/tools/test_kanban_tools.py`
  - `tests/agent/transports/test_hermes_tools_mcp_server.py`
- Validation run:
  - `py -3.12 -m pytest tests/hermes_cli/test_kanban_orchestrated_coding.py -q`
    -> 7 passed.
  - `python -m pytest tests/hermes_cli/test_kanban_orchestrated_coding.py tests/hermes_cli/test_kanban_plan_audit_gate.py tests/hermes_cli/test_kanban_budget_enforcement.py tests/hermes_cli/test_kanban_goal_mode.py -q`
    -> 44 passed.
  - `python -m pytest tests/agent/transports/test_hermes_tools_mcp_server.py tests/tools/test_kanban_tools.py::test_record_plan_audit_verdict_opens_executor_gate tests/tools/test_kanban_tools.py::test_record_plan_audit_verdict_reject_replay_is_idempotent tests/tools/test_kanban_tools.py::test_apply_plan_audit_actuation_approved_opens_gate_and_completes_auditor tests/tools/test_kanban_tools.py::test_apply_plan_audit_actuation_reject_creates_revision_and_completes_auditor tests/tools/test_kanban_tools.py::test_apply_plan_audit_actuation_needs_user_blocks_executor -q`
    -> 14 passed.
  - `python -m py_compile hermes_cli/kanban_db.py tools/kanban_tools.py toolsets.py agent/transports/hermes_tools_mcp_server.py tests/hermes_cli/test_kanban_orchestrated_coding.py tests/tools/test_kanban_tools.py tests/agent/transports/test_hermes_tools_mcp_server.py`
    -> passed.
- Environment caveat remains: Claude installed `pyyaml`/`python-dotenv` into
  `py -3.12`, which unskipped one budget test there, but that interpreter
  still lacks `rich`; the focused DB bundle is therefore validated with managed
  `python` for dependency completeness. `scripts/run_tests.sh` still cannot be
  used in this checkout until a proper POSIX/uv-backed venv is available.

## Active Work Update: Live E2E Smoke Passed

- Live Kanban Orchestrated Coding smoke test was added at
  `tests/e2e/test_kanban_orchestrated_coding_smoke.py`.
- The test exercises a real temp Git repo, worktree/branch, pytest process,
  Kanban task graph, plan-audit gate, worker model call, patch application,
  verification, commit metadata, local merge, and final audit report.
- Local deterministic harness:
  `python -m pytest tests/e2e/test_kanban_orchestrated_coding_smoke.py -q`
  -> 4 passed, 1 deselected.
- Live integration command:
  `python -m pytest tests/e2e/test_kanban_orchestrated_coding_smoke.py -q -m integration -s`
  -> 1 passed, 4 deselected.
- Provider note: the available live key was an OpenCode key, so the smoke test
  routes the DeepSeek-family worker lane through `opencode-go` with
  `deepseek-v4-flash`; direct DeepSeek and OpenRouter remain supported route
  options. Do not treat a specific provider as the architecture center.
- Next action remains cleanup/audit of the current diff before expanding the
  orchestration surface.
