# Claude Code Task Packet — Phase 7 Runtime Orchestration Context + Minimal Status Commands

IMPORTANT: This is a narrow Phase 7 runtime/context + read-only status-command task. Build the smallest safe bridge between the Phase 6 status formatter and future `/tasks` / `/agents` / natural-language status UX. Do **not** implement gateway natural-language routing, append/correction attachment, worker dispatch, cancel UX, durable DB, Ralph runtime, LLM classifier, or broad CLI/gateway/TUI refactors.

## Worktree

```text
/tmp/hermes-orchestrator-phase-7
```

## Baseline

```text
eddabc597 feat(agent): add orchestration status observatory
```

## Source plans

```text
docs/plans/2026-05-12-hermes-orchestrator-roadmap-phase7-plus.md
docs/plans/2026-05-12-hermes-orchestrator-first-update-plan.md
docs/plans/2026-05-12-phase6-orchestration-observatory-notes.md
docs/plans/2026-05-12-claude-phase6-task-packet.md
```

## Product intent

Woo compared the desired Hermes behavior to a smooth Manus-like coordinator and Claude's recent agent view: the user can keep talking naturally, while a visible coordinator knows which task/agent is doing what.

Phase 6 created the read-only formatter:

```text
TaskRegistry + WorkerLaneRegistry
→ OrchestrationStatusFormatter
```

Phase 7 should create the smallest runtime context that lets CLI/gateway code later access a live `TaskRegistry` + `WorkerLaneRegistry`, and should wire minimal read-only status commands **only if** they can safely read that runtime context without fake state.

This is the bridge from formatter-only to actual observability UX.

## Current substrate

Already exists:

```text
agent/pending_turn_queue.py       # PendingTurnItem / queue
agent/task_registry.py            # TaskRegistry / FocusedTask
agent/worker_lanes.py             # WorkerLaneRegistry / ThreadWorkerLane
agent/followup_router.py          # conservative router
agent/orchestration_status.py     # format_tasks / format_agents / format_overview
```

## Required scope

Prefer a small new module:

```text
agent/orchestration_runtime.py
```

Suggested minimal API:

```python
@dataclass
class OrchestrationRuntime:
    task_registry: TaskRegistry
    worker_registry: WorkerLaneRegistry

    @classmethod
    def create(cls) -> "OrchestrationRuntime": ...

    def snapshot(self, *, session_key: str | None = None) -> OrchestrationSnapshot: ...
    def format_tasks(self, *, session_key: str | None = None, compact: bool = True) -> str: ...
    def format_agents(self, *, compact: bool = True) -> str: ...
    def format_overview(self, *, session_key: str | None = None, compact: bool = True) -> str: ...
```

Also consider helper functions that make integration safe and duck-typed:

```python
get_or_create_orchestration_runtime(owner) -> OrchestrationRuntime
get_orchestration_runtime(owner) -> OrchestrationRuntime | None
set_orchestration_runtime(owner, runtime) -> OrchestrationRuntime
format_runtime_tasks(owner, ...)
format_runtime_agents(owner, ...)
```

The `owner` may later be a `HermesCLI`, gateway runner, session object, or test dummy. Keep it simple: store on a private attribute such as `_orchestration_runtime`. Do not create a global singleton.

## Minimal command wiring

If the central command registry and handler path are straightforward, add read-only commands:

```text
/tasks
/agents
```

Expected UX:

```text
/tasks
→ No active tasks are currently registered.
```

```text
/agents
→ No active workers are currently registered.
```

With injected runtime in tests:

```text
/tasks
→ Active tasks:
  - task_ab12 [running] Phase 7 runtime wiring
    worker: worker_cd34 (claude_code) [running]
```

```text
/agents
→ Workers:
  - worker_cd34 [running] lane=thread task=task_ab12 goal="Phase 7 runtime wiring"
```

If command wiring would require broad `cli.py` / `gateway/run.py` refactor, stop at the runtime helper module and document why command wiring is deferred. But first inspect the existing command registry/handler pattern; adding read-only command handlers may be small.

## Explicit non-goals

Do **not** implement:

- Telegram/gateway natural-language status auto-routing.
- append/correction attachment from live Telegram messages.
- worker dispatch or new task creation.
- cancel / stop task behavior.
- force kill.
- durable routing DB or SQLite schema.
- public `delegate_task(background=True)` API.
- Ralph runtime.
- LLM classifier.
- global singleton runtime.
- broad TUI/gateway refactor.
- worker result synthesis pipeline.

## Allowed files

Prefer:

```text
agent/orchestration_runtime.py
tests/agent/test_orchestration_runtime.py
docs/plans/2026-05-12-phase7-runtime-status-notes.md
```

Allowed if small and well-tested:

```text
hermes_cli/commands.py
cli.py
gateway/run.py
tests/cli/test_orchestration_status_commands.py
tests/gateway/test_orchestration_status_commands.py
```

Also include the roadmap doc in this phase artifact if it remains uncommitted:

```text
docs/plans/2026-05-12-hermes-orchestrator-roadmap-phase7-plus.md
```

## Acceptance criteria

- `OrchestrationRuntime.create()` creates a fresh in-memory `TaskRegistry` and `WorkerLaneRegistry`.
- Runtime formatting delegates to Phase 6 `format_tasks`, `format_agents`, `format_overview`.
- Helper functions can attach runtime to a CLI/gateway-like owner object without global singleton behavior.
- Empty state is graceful.
- Injected runtime with one task/worker formats correctly.
- Runtime snapshots remain JSON-safe.
- If `/tasks` and `/agents` are wired:
  - central command registry/help recognizes them;
  - CLI handler returns formatter output;
  - gateway handler, if touched, returns read-only formatter output;
  - missing runtime creates safe empty runtime or returns empty state;
  - no normal chat/gateway behavior changes.
- No `PendingTurnItem.raw` traversal/serialization/deepcopy.
- Existing Phase 1-6 targeted suites remain green.

## Required tests

Add agent tests:

```text
tests/agent/test_orchestration_runtime.py
```

Cover:

- create runtime
- explicit registries injection
- get/set/get-or-create helpers on dummy owner
- no global singleton leakage between owners
- format empty tasks/agents/overview
- format injected task/worker state
- JSON-safe snapshot
- `PendingTurnItem.raw` safety if a follow-up exists on a task

If command wiring is implemented, add tests for command registry and handler behavior. Possible files:

```text
tests/cli/test_orchestration_status_commands.py
tests/gateway/test_orchestration_status_commands.py
```

At minimum run:

```bash
/Users/wookim/.hermes/hermes-agent/venv/bin/python -m pytest \
  tests/agent/test_orchestration_runtime.py \
  tests/agent/test_orchestration_status.py \
  tests/agent/test_followup_router.py \
  tests/agent/test_worker_lanes.py \
  tests/agent/test_task_registry.py \
  tests/agent/test_pending_turn_queue.py -q

/Users/wookim/.hermes/hermes-agent/venv/bin/python -m pytest \
  tests/cli/test_busy_queue_coalescing.py \
  tests/cli/test_busy_input_mode_command.py \
  tests/gateway/test_restart_drain.py \
  tests/gateway/test_session_race_guard.py -q

/Users/wookim/.hermes/hermes-agent/venv/bin/python -m compileall -q \
  agent/orchestration_runtime.py agent/orchestration_status.py agent/followup_router.py \
  agent/worker_lanes.py agent/task_registry.py agent/pending_turn_queue.py cli.py gateway/run.py

git diff --check
```

If slash command files are changed, include their targeted tests too.

## Required notes

Create:

```text
docs/plans/2026-05-12-phase7-runtime-status-notes.md
```

Include:

```text
## Summary
## PURPOSE-FIT DESIGN RATIONALE
## WHAT YOU INTENTIONALLY DID NOT BUILD
## RALPH/FUTURE FOCUSED-AGENT NOTES
## Validation
## Risks / Follow-up
```

## Claude Code instructions

Use Claude Opus-class model and max effort.

Implement exactly this phase. Do not commit. Do not push. Stop once there is a coherent, testable, reviewable runtime/status bridge. Return:

```text
Summary
Changed files
Tests run + results
Purpose-fit rationale
Intentional non-goals
Ralph/future focused-agent notes
Known risks/questions
```
