# Kanban Orchestrated Coding (V2.1 - Slice 3/3)

## Mode

`guarded`

## Goal

- Add the thin "Kanban Orchestrated Coding" layer promised by the V2.1
  architecture: a skill/profile contract plus the real worker-embedded
  plan-auditor actuator that turns a user coding goal into a durable Kanban
  planner -> plan-auditor -> executor -> code-auditor -> summarizer workflow.
- Keep Hermes Kanban as the workflow engine. Do not add a second orchestrator,
  workflow DB, dispatcher hook, polling daemon, or new core model tool.
- Implement the plan-audit actuator outside the dispatcher claim lock. The
  actuator records binary verdicts with the existing Kanban primitive and
  creates planner-revision tasks idempotently on rejection.
- Make the critical task-id rule impossible to miss: `plan_audit_required` and
  `record_plan_audit_verdict()` target the gated executor task id, not the
  auditor task id.

## Scope

Allowed:
- Add an in-repo skill, likely
  `skills/software-development/kanban-orchestrated-coding/SKILL.md`, that
  instructs orchestrator/planner/auditor/executor/reviewer profiles how to use
  the existing Kanban tools for the coding MVP.
- Add a very small plan-audit verdict surface if needed so a Kanban worker can
  call the already-implemented DB helper without shelling out:
  either `tools/kanban_tools.py` exposes a Kanban-gated tool such as
  `kanban_record_plan_audit_verdict`, or a focused Python helper/CLI command is
  added under `hermes_cli/kanban*.py` and used only by a Hermes-owned worker.
  Prefer the smallest surface that preserves the worker contract.
- Add a plan-auditor actuator helper that:
  - reads the planner output/comment/run metadata,
  - calls the configured auditor outside `claim_task()`,
  - records `approved`/`rejected` on the executor task id idempotently for the
    current audit round,
  - writes structured comments with `rejected.kind`,
  - creates the next planner/auditor revision tasks with round-scoped
    `idempotency_key`s, or blocks the workflow at `max_rounds`.
  - completes its own auditor task after verdict recording and graph mutation,
    so actuator tasks do not remain hanging.
- Add focused tests for:
  - verdict recorded on auditor task id does not open the executor gate;
  - verdict recorded on executor task id opens the gate;
  - reject -> revise -> approve;
  - executor can be promoted to `ready` before approval but still cannot claim;
  - `max_rounds` -> `blocked`/`needs_input`, no infinite loop;
  - replay after crash/retry does not duplicate revision tasks or double-count
    rejected rounds;
  - detached-worker HITL degrades to Kanban comment/block rather than
    interactive approval.
- Update user-facing docs only if the feature is exposed beyond the skill
  itself.

Not allowed:
- No new dispatcher hook, background daemon, workflow database, or broad
  workflow-template engine.
- No new core model tools. A Kanban tool addition is only allowed if it is the
  smallest way for an existing Kanban worker to record the existing verdict
  primitive.
- No OpenClaw integration, Codex/Claude CLI lane implementation, PR creation,
  auto-commit, push, deploy, or destructive cleanup.
- No live provider/API calls in tests.
- No account/day/provider-wide budget enforcement; slice 2 remains per-task/run
  plus visible workflow ceiling.
- No changes to prompt-cache semantics or system-prompt mutation outside normal
  skill loading.

## Source Of Truth

- User request: original Hermes "AI Chief of Staff" architecture request,
  followed by three Claude audit rounds. Final audit verdict: spec is ready for
  slice 3 after closing task-id keying, async HITL, round idempotency, and
  reject taxonomy.
- Architecture artifact:
  `docs/ai/hermes-personal-orchestration-architecture.md`, especially §7.2,
  §7.5, §16, §17, and §20.
- Prior slices:
  - `tasks/TASK_kanban_plan_audit_gate.md`: implemented
    `plan_audit_required`, `plan_audit_max_rounds`, verdict events, and
    `record_plan_audit_verdict()`.
  - `tasks/TASK_kanban_budget_enforcement.md`: implemented task/run budget
    accounting and claim gates.
- Current code verified before this runsheet:
  - `hermes_cli/kanban_db.py:create_task()` accepts `idempotency_key`,
    `skills`, `plan_audit_required`, `plan_audit_max_rounds`, `budget_usd`,
    and `initial_status`.
  - `hermes_cli/kanban_db.py:link_tasks()` demotes a ready child to `todo`
    when linking an unfinished parent, so a rejected audit can add revision
    parents in front of the executor safely.
  - `hermes_cli/kanban_db.py:record_plan_audit_verdict()` records binary
    approved/rejected events with optional metadata.
  - `record_plan_audit_verdict()` currently appends events unconditionally,
    while `_plan_audit_rejections_since_approval()` counts rejected events; the
    slice 3 actuator must not replay the same rejected round as a second event.
  - `hermes_cli/kanban_db.py:claim_task()` gates only the task id being
    claimed; it checks `_latest_plan_audit_event(conn, task_id)`.
  - `tools/kanban_tools.py:kanban_create` already exposes
    `idempotency_key`, `plan_audit_required`, `plan_audit_max_rounds`,
    `budget_usd`, `skills`, and parent links.
  - `tools/kanban_tools.py:kanban_comment` is available to task workers for
    durable structured notes.
  - Checkout currently does **not** contain
    `skills/autonomous-ai-agents/kanban-codex-lane/SKILL.md` even though the
    generated website docs mention it; do not depend on that file unless it is
    restored or created as part of a separate task.

## Read Order

1. `AGENTS.md`
2. `PROJECT_RULES.md`
3. `HANDOFF.md`
4. `docs/ai/hermes-personal-orchestration-architecture.md`
5. `tasks/TASK_kanban_plan_audit_gate.md`
6. `tasks/TASK_kanban_budget_enforcement.md`
7. `skills/software-development/plan/SKILL.md`
8. `skills/software-development/requesting-code-review/SKILL.md`
9. `skills/autonomous-ai-agents/codex/SKILL.md`
10. `skills/autonomous-ai-agents/claude-code/SKILL.md`
11. `skills/software-development/hermes-agent-skill-authoring/SKILL.md`
12. `hermes_cli/kanban_db.py`: `create_task()`, `link_tasks()`,
    `add_comment()`, `record_plan_audit_verdict()`, `claim_task()`,
    `block_task()`, `complete_task()`
13. `tools/kanban_tools.py`: `kanban_create`, `kanban_comment`,
    `kanban_block`, `kanban_complete`, schema/registration patterns
14. `hermes_cli/kanban.py`: CLI parser/verbs if a human CLI verdict command is
    needed
15. Tests:
    `tests/hermes_cli/test_kanban_plan_audit_gate.py`,
    `tests/hermes_cli/test_kanban_budget_enforcement.py`,
    `tests/hermes_cli/test_kanban_goal_mode.py`,
    `tests/tools/test_skill_usage.py` or nearby skill metadata tests if the
    new skill needs catalog coverage

## Planned Files

- `skills/software-development/kanban-orchestrated-coding/SKILL.md` (new)
  - MVP procedural contract for orchestrator, planner, plan-auditor,
    executor, code-auditor/reviewer, and summarizer roles.
  - Must explicitly state that `record_plan_audit_verdict()` targets the gated
    executor task id.
  - Must instruct detached workers to use Kanban `blocked` + comment for HITL,
    not interactive approval prompts.
  - Must include idempotency key shape for planner/auditor revisions, e.g.
    `koc:<root_task_id>:<executor_task_id>:plan-round:<n>:planner` and
    `...:auditor`.
  - Must state that the plan-auditor task calls `kanban_complete` after it has
    recorded the verdict and either created revision tasks, opened the executor,
    or blocked for human input.
- `tools/kanban_tools.py`
  - Investigate adding a Kanban-gated verdict-recording tool. Candidate:
    `kanban_record_plan_audit_verdict` with params:
    `task_id` (executor/gated task id), `approved`, `reason`,
    `metadata`, `board`.
  - Stop and reconsider if this would become a broad ungated surface. Follow
    the existing Kanban tool pattern in this checkout: the tool name may be
    present in platform/core bundles for discovery parity, but the actual model
    schema is gated by `tools/kanban_tools.py` `check_fn` and only appears for
    dispatcher workers or profiles explicitly enabling the `kanban` toolset.
- `hermes_cli/kanban_db.py`
  - Prefer no schema changes.
  - May add small helper(s) if idempotent verdict recording, revision creation,
    or metadata assembly would otherwise duplicate fragile SQL in tools/CLI/tests.
  - Important correctness risk: rejected-round replay must be idempotent by
    `(executor_task_id, round)`. A replay after `record_plan_audit_verdict()`
    but before revision creation must not append another rejected event that
    causes `_plan_audit_rejections_since_approval()` to double-count and block
    `max_rounds` early.
  - Do not change `claim_task()` unless tests reveal a bug in the already
    implemented gate.
- `hermes_cli/kanban.py`
  - Only if a human/debug CLI command is needed, e.g.
    `hermes kanban plan-audit-verdict <executor-task-id> --approved/--rejected`.
  - Prefer tool/API path first because dispatcher workers use tools, not shell
    CLI, per Kanban docs.
- `hermes_cli/config.py` and `cli-config.yaml.example`
  - Only if the actuator needs a new behavior key beyond existing
    `auxiliary.plan_auditor`, `kanban.plan_audit_*`, and
    `kanban.task_budget.*`.
  - Do not create another parallel config namespace unless absolutely needed;
    the architecture doc treats `kanban_orchestration.roles.*` as a future
    profile mapping, not required for MVP code if a skill can encode it.
- `tests/hermes_cli/test_kanban_orchestrated_coding.py` (new)
  - DB/tool-level orchestration tests without live providers.
- `tests/tools/test_kanban_tools.py` or equivalent existing tool tests
  - Only if adding a new Kanban verdict tool.
- `website/docs/user-guide/skills/...`
  - Do not hand-edit generated docs. If skill docs are generated in this repo,
    update via the documented generator only after confirming expected process.

## Mini-Plan

- Files to inspect/edit:
  - Inspect all Read Order files.
  - Edit only the planned files required by the smallest implementable slice.
- TDD sequence:
  1. Write DB-contract tests first, independent of worker-facing surface:
     wrong-id/right-id gate behavior, ready-but-unapproved no-claim,
     reject replay does not double-count rounds, reject -> revise -> approve,
     and max-rounds block.
  2. Run those tests and confirm they fail for the missing actuator/idempotency
     behavior.
  3. Do a short surface spike: choose the smallest worker-facing way to record
     verdicts (`kanban` toolset preferred, CLI/debug path only if justified).
  4. Add surface tests only after the surface is chosen.
  5. Implement the smallest code to pass DB-contract tests, then the chosen
     surface tests, then the skill contract.
- Commands to run:
  - `rg` and targeted `Get-Content`/`sed` for code references before editing.
  - `python -m py_compile` on changed Python files.
  - focused pytest for new orchestration tests plus slice 1/2 regressions.
- Live provider/gateway/cron/browser state touched: no.
- Stop conditions:
  - If recording verdict from a worker requires adding a broad core model tool,
    stop. The allowed surface is only the existing `kanban` toolset or a narrow
    CLI/debug path.
  - If an LLM/auditor call would run inside `claim_task()` or any SQLite write
    transaction, stop. The actuator must run outside claim locks.
  - If the executor task id cannot be carried unambiguously through planner and
    auditor task bodies/comments, stop and redesign the task metadata shape
    before coding.
  - If idempotent revision creation cannot be made deterministic using existing
    `idempotency_key`, stop before implementing retry-prone graph mutation.
  - If idempotent verdict recording cannot be keyed to a stable audit round
    without ambiguous metadata, stop before writing the worker-facing surface.
  - If detached HITL would depend on `tools.approval.py` without a listener,
    stop and route through `kanban_block(kind="needs_input")` + comment.
  - If the same validation failure repeats twice with the same signature,
    follow the Repeated-Blocker Rule.
- Validation plan:
  - Test the current shipped gate contract first: wrong task id verdict does
    not open executor; executor task id verdict does.
  - Test promotion-race window: executor is `ready` before plan approval but
    `claim_task()` still refuses it until approved.
  - Test reject -> planner r2/auditor r2 -> approve -> executor claim path.
  - Test `max_rounds` block behavior and no infinite revision creation.
  - Test replay/idempotency around the crash window after reject before
    revision creation: replay must not duplicate revision tasks, must not
    append a second rejected event for the same round, and must not block the
    executor early via a double-counted rejection.
  - Test plan-auditor actuator completes/terminates its own task after
    successful actuation.
  - Test skill metadata/loading if adding a new skill.

## Safety Audit

- No destructive command without approval: yes.
- No live external/provider side effect without approval: yes.
- No secrets printed or committed: yes.
- No generated/local/runtime files staged: yes; tests must use temp
  `HERMES_HOME`/temp DB.
- No unrelated dirty state hidden: run `git status --short --branch` before and
  after. Known unrelated/pending state exists in this checkout; do not stage
  broadly or fold unrelated desktop/setup/docs changes into this slice.

## Validation Commands

- `python -m py_compile hermes_cli/kanban_db.py tools/kanban_tools.py hermes_cli/kanban.py tests/hermes_cli/test_kanban_orchestrated_coding.py`
- Preferred CI-parity when venv is fixed:
  `bash scripts/run_tests.sh tests/hermes_cli/test_kanban_orchestrated_coding.py -q`
- Current Windows fallback used by prior slices:
  `py -3.12 -m pytest tests/hermes_cli/test_kanban_orchestrated_coding.py -q`
- Focused regression:
  `py -3.12 -m pytest tests/hermes_cli/test_kanban_orchestrated_coding.py tests/hermes_cli/test_kanban_plan_audit_gate.py tests/hermes_cli/test_kanban_budget_enforcement.py tests/hermes_cli/test_kanban_goal_mode.py -q`
- If a new Kanban tool is added:
  `py -3.12 -m pytest tests/hermes_cli/test_kanban_orchestrated_coding.py tests/tools/test_kanban_tools.py -q`
- Skill/catalog smoke, exact test target to confirm during implementation:
  `py -3.12 -m pytest tests/tools/test_skill_usage.py -q`
- `git diff --check -- skills/software-development/kanban-orchestrated-coding/SKILL.md tools/kanban_tools.py hermes_cli/kanban_db.py hermes_cli/kanban.py hermes_cli/config.py cli-config.yaml.example tests/hermes_cli/test_kanban_orchestrated_coding.py tasks/TASK_kanban_orchestrated_coding_slice3.md HANDOFF.md`

## Attempt Ledger

_(để trống khi lập runsheet; điền khi thực thi)_

Attempt 1:
- Goal: lock DB-level slice 3 contracts before exposing worker surface.
- Change or action: added `tests/hermes_cli/test_kanban_orchestrated_coding.py` for wrong-id gating, ready-but-unapproved claim refusal, rejected-round replay, legacy duplicate round counting, max-round blocking, reject-revise-approve, and async HITL block/comment.
- Validation command: `py -3.12 -m pytest tests/hermes_cli/test_kanban_orchestrated_coding.py -q`
- Result: red-first exposed missing idempotent rejected-round handling; after DB fixes, 7 passed.
- Artifact/log: `tests/hermes_cli/test_kanban_orchestrated_coding.py`.
- Error signature: duplicate `plan_audit_rejected` events for the same round could double-count toward `max_rounds`.
- Hypothesis after result: verdict recording needed round-key dedup and rejection counting needed unique keyed rounds.

Attempt 2:
- Goal: provide the worker-facing plan-auditor actuator surface without adding dispatcher hooks or a new core model tool.
- Change or action: added idempotent `apply_plan_audit_actuation()` DB helper plus `kanban_apply_plan_audit_actuation` under the existing gated Kanban toolset. The helper records verdicts on the executor task id, creates round-keyed revision planner/auditor tasks, blocks `needs_input`/max-round cases, and completes the current auditor task.
- Validation command: `python -m pytest tests/tools/test_kanban_tools.py::test_kanban_tools_visible_with_env_var tests/tools/test_kanban_tools.py::test_worker_with_kanban_toolset_still_hides_board_routing tests/tools/test_kanban_tools.py::test_kanban_tools_visible_with_toolset_config tests/tools/test_kanban_tools.py::test_apply_plan_audit_actuation_reject_creates_revision_and_completes_auditor tests/tools/test_kanban_tools.py::test_apply_plan_audit_actuation_needs_user_blocks_executor tests/agent/transports/test_hermes_tools_mcp_server.py::TestModuleSurface::test_kanban_worker_tools_exposed -q`
- Result: 6 passed.
- Artifact/log: `tools/kanban_tools.py`, `toolsets.py`, `agent/transports/hermes_tools_mcp_server.py`, and focused tool/MCP tests.
- Error signature: none after implementation.
- Hypothesis after result: the surface is narrow enough because it is only in the existing Kanban toolset/check_fn path and codex-runtime MCP worker callback.

## Execution Log

- Runsheet created for slice 3 only.
- Implemented DB idempotency for `record_plan_audit_verdict()` by `(task_id, kind, metadata.round)` and made `_plan_audit_rejections_since_approval()` count unique keyed rounds while preserving fallback raw counts for unkeyed legacy events.
- Implemented `apply_plan_audit_actuation()` in `hermes_cli/kanban_db.py` for the worker-embedded actuator: verdict on executor id, revision planner/auditor cards by deterministic `koc:<root>:<executor>:plan-round:<n>:...` keys, `needs_input` block/comment, max-round block, and auditor self-completion.
- Post-audit polish: `_block_plan_audit_executor_for_input()` now preserves the block recurrence signal and routes to `triage` when the existing loop-breaker threshold is reached; replayed plan-audit comments are idempotent by task/author/body.
- Added `kanban_apply_plan_audit_actuation` to `tools/kanban_tools.py`, the `kanban` toolset, and the Codex-runtime MCP callback, alongside the lower-level `kanban_record_plan_audit_verdict` primitive.
- Added `skills/software-development/kanban-orchestrated-coding/SKILL.md` and updated it to prefer the actuator tool for plan-auditor workers.
- Validation:
  - `py -3.12 -m pytest tests/hermes_cli/test_kanban_orchestrated_coding.py -q` -> 7 passed.
  - `python -m pytest tests/hermes_cli/test_kanban_orchestrated_coding.py tests/hermes_cli/test_kanban_plan_audit_gate.py tests/hermes_cli/test_kanban_budget_enforcement.py tests/hermes_cli/test_kanban_goal_mode.py -q` -> 44 passed.
  - `python -m pytest tests/agent/transports/test_hermes_tools_mcp_server.py tests/tools/test_kanban_tools.py::test_record_plan_audit_verdict_opens_executor_gate tests/tools/test_kanban_tools.py::test_record_plan_audit_verdict_reject_replay_is_idempotent tests/tools/test_kanban_tools.py::test_apply_plan_audit_actuation_approved_opens_gate_and_completes_auditor tests/tools/test_kanban_tools.py::test_apply_plan_audit_actuation_reject_creates_revision_and_completes_auditor tests/tools/test_kanban_tools.py::test_apply_plan_audit_actuation_needs_user_blocks_executor -q` -> 14 passed.
  - `python -m py_compile hermes_cli/kanban_db.py tools/kanban_tools.py toolsets.py agent/transports/hermes_tools_mcp_server.py tests/hermes_cli/test_kanban_orchestrated_coding.py tests/tools/test_kanban_tools.py tests/agent/transports/test_hermes_tools_mcp_server.py` -> passed.

## Final Report Checklist

- Status reported
- Changed files listed
- Commands and validation results listed
- Artifacts listed, or `none`
- Handoff update noted
- Remaining risk noted

---

## Scope Notes

This is **slice 3/3** of the V2.1 Kanban-first baseline:

1. Slice 1: plan-audit gate in `claim_task()` - implemented and approved.
2. Slice 2: task/run budget enforcement - implemented and approved.
3. Slice 3: this runsheet - skill plus worker-embedded plan-auditor actuator.

The most important acceptance gates before calling slice 3 `DONE`:

1. Executor cannot claim until `plan_audit_approved` is recorded on that exact
   executor task id.
2. `max_rounds` routes to `blocked`/`needs_input`, not an infinite loop.
3. Replaying the actuator after a reject/crash window does not duplicate
   planner/auditor revision tasks or double-count rejected rounds.
4. The plan-auditor task completes itself after recording verdict and applying
   graph mutation.

Do not start by implementing external Codex/Claude/OpenClaw lanes. The MVP can
use normal Hermes profiles and existing Kanban tools. External lanes are later
optional adapters.
