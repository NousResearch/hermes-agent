# Claude Code Task Packet — Phase 6 Orchestration Observatory

IMPORTANT: This is a narrow Phase 6 visibility/status UX task. The user explicitly compared Manus-like smooth orchestration and Claude's recent agent-view style visibility. Build the first read-only observability layer for tasks/workers and minimal `/tasks` / `/agents` command surfaces where appropriate. Do **not** implement Ralph runtime, LLM classifier, automatic Telegram routing, background delegate API, force kill, durable routing DB, or broad gateway refactor.

## Worktree

```text
/tmp/hermes-orchestrator-phase-6
```

## Baseline

```text
d875d6c15 feat(agent): add conservative follow-up router
```

## Product intent

The desired UX is not command-heavy. Woo should be able to speak naturally while Hermes coordinates work behind the scenes.

However, once Hermes has tasks and workers, the user needs a lightweight observatory:

```text
What tasks are active?
Which workers/agents are running?
What follow-ups are queued?
Is anything blocked or waiting for me?
```

This should feel like a front-desk/concierge status board, not a developer-only dump.

## Current substrate

Existing phases already provide:

- Phase 2: `agent/pending_turn_queue.py` — structured input/follow-up units.
- Phase 3: `agent/task_registry.py` — focused task identity/status/followups/artifacts/notes/worker linkage.
- Phase 4: `agent/worker_lanes.py` — worker specs/handles/results/cancel/followups/registry.
- Phase 5: `agent/followup_router.py` — conservative follow-up routing decisions.

Phase 6 should add a **read-only presentation/status layer** over those pieces.

## Required scope

Prefer a new module:

```text
agent/orchestration_status.py
```

Suggested concepts:

```python
@dataclass
class OrchestrationSnapshot:
    tasks: list[dict]
    workers: list[dict]
    counts: dict
    warnings: list[str]

class OrchestrationStatusFormatter:
    def snapshot(task_registry=None, worker_registry=None, *, session_key=None) -> OrchestrationSnapshot: ...
    def format_tasks(snapshot_or_registry, *, compact=True, session_key=None) -> str: ...
    def format_agents(snapshot_or_registry, *, compact=True) -> str: ...
    def format_overview(...) -> str: ...
```

Also add minimal command registry entries if the command system supports it cleanly:

```text
/tasks
/agents
```

But keep the handler integration conservative. If full runtime registry wiring is not yet available, command handlers may return a graceful message such as:

```text
No focused tasks are currently registered in this session.
```

or use duck-typed registries if the CLI/gateway later injects them. Do not invent a global durable registry just to make the command non-empty.

## UX expectations

`/tasks` should answer:

```text
Active tasks:
- task_ab12 [running] Hermes Phase 6 observability
  worker: worker_cd34 (claude_code) [running]
  follow-ups: 2
  notes: 1
```

`/agents` should answer:

```text
Workers:
- worker_cd34 [running] lane=thread task=task_ab12 goal="Implement Phase 6"
- worker_ef56 [done] lane=review task=task_ab12
```

When empty:

```text
No active tasks/workers are currently registered.
```

The output should be concise and Telegram-friendly: bullets and labels, no markdown tables.

## Optional natural-language status helper

Add deterministic helper functions only if they fit cleanly:

```python
looks_like_orchestration_status_query(text) -> bool
```

Examples:

- `지금 뭐 하고 있어?`
- `돌고 있는 작업 있어?`
- `에이전트 뭐 돌아가?`
- `what are you working on?`
- `active tasks?`
- `any agents running?`

This helper should not route messages by itself; it only lets later gateway wiring call the formatter.

## Explicit non-goals

Do **not** implement:

- Ralph runtime.
- LLM/model classifier.
- automatic Telegram/gateway routing of natural language.
- real worker process dashboard with live polling beyond existing `WorkerLaneRegistry` snapshots.
- force kill / force cancel.
- public `delegate_task(background=True)` API.
- durable routing DB or SQLite schema.
- long-lived global singleton registry unless an existing runtime already has one.
- broad gateway/CLI/TUI refactor.
- worker result delivery/synthesis pipeline.

## Allowed files

Prefer:

```text
agent/orchestration_status.py
tests/agent/test_orchestration_status.py
docs/plans/2026-05-12-phase6-orchestration-observatory-notes.md
```

Allowed if needed and small:

```text
hermes_cli/commands.py
cli.py
gateway/run.py
tests/cli/test_*commands*.py
tests/gateway/test_*commands*.py
```

But avoid editing `cli.py` / `gateway/run.py` unless the command integration path is obvious, small, and well-tested. A pure formatter/status module is acceptable if runtime wiring is too broad for this phase.

## Acceptance criteria

- Can build a snapshot from `TaskRegistry` with zero, one, and multiple tasks.
- Can build a snapshot from `WorkerLaneRegistry` with zero, running, done, error, and cancelled workers.
- Task formatting includes id, status, compact goal, worker linkage, follow-up count, note count, and blocked/error indicators.
- Agent/worker formatting includes worker id, status, lane, task id, compact goal, cancel-requested marker, and error/result summary when available.
- Output is concise, deterministic, and Telegram-friendly.
- Empty state is graceful.
- No `PendingTurnItem.raw` traversal/serialization/deepcopy.
- Snapshot/formatter outputs are JSON-safe where advertised, using strict JSON behavior for metadata if needed.
- `/tasks` and `/agents` command registry entries exist only if they can be wired without broad refactor; otherwise document why they are deferred.
- No production gateway behavior changes unless covered by targeted tests.

## Required tests

Add targeted tests for:

```text
tests/agent/test_orchestration_status.py
```

Cover:

- empty snapshot/formatting
- one running task
- multiple active tasks ordered predictably
- task with worker linkage/followups/notes/artifacts
- blocked/error/cancelled/done task formatting
- worker registry running/done/error/cancelled handles/results
- cancel-requested marker
- compact truncation
- JSON-safe snapshot if `to_dict()` is provided
- no `PendingTurnItem.raw` touch with raw object that raises on deepcopy/access
- Korean/English natural-language status query helper if implemented

If slash commands are integrated, add command tests for `/tasks` and `/agents` help/registry/handler behavior.

Run at minimum:

```bash
/Users/wookim/.hermes/hermes-agent/venv/bin/python -m pytest \
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
  agent/orchestration_status.py agent/followup_router.py agent/worker_lanes.py agent/task_registry.py agent/pending_turn_queue.py cli.py gateway/run.py

git diff --check
```

## Required final notes

Create:

```text
docs/plans/2026-05-12-phase6-orchestration-observatory-notes.md
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

Implement exactly this phase. Do not commit. Do not push. Stop once there is a coherent, testable, reviewable observability/status foundation. Return:

```text
Summary
Changed files
Tests run + results
Purpose-fit rationale
Intentional non-goals
Ralph/future focused-agent notes
Known risks/questions
```
