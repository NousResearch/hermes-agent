---
name: kanban-orchestrated-coding
description: Use when a coding goal should run through Hermes Kanban with planner, plan-auditor, executor, reviewer, and summarizer tasks. Builds a durable workflow using existing Kanban tools, plan-audit verdicts, budgets, comments, and blocks without adding a new orchestrator.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [kanban, orchestration, coding, plan-audit, multi-agent]
    related_skills: [plan, test-driven-development, requesting-code-review]
---

# Kanban Orchestrated Coding

## Overview

Use the existing Hermes Kanban board as the orchestration spine for coding work. Do not invent a second workflow database, dispatcher hook, daemon, or core model tool. The workflow is a small graph of normal Kanban tasks whose workers communicate through task bodies, comments, run summaries, task links, and plan-audit verdict events.

The plan-auditor task is an actuator: it reviews the planner output, records the verdict on the gated executor task, mutates the graph when revision is needed, and then completes itself. The dispatcher remains a claim engine only.

## When to Use

Use this skill when:

- A user asks Hermes to implement a coding change and wants planner/auditor/executor separation.
- The executor should not start until a plan has been approved.
- Rejection should create a durable revision round instead of relying on chat memory.
- A detached worker needs human input and must block through Kanban rather than wait on an interactive prompt.

Do not use for:

- Small one-shot local edits that do not need multi-agent separation.
- Live provider experiments, external CLI lanes, PR creation, commit, push, deploy, or destructive cleanup unless the task explicitly asks for that.
- Adding broad workflow-template infrastructure. This skill uses the Kanban primitives already present.

## Core Invariants

- The executor task carries `plan_audit_required=true`.
- The plan-audit verdict is recorded on the executor task id, not the auditor task id.
- The preferred actuator call is `kanban_apply_plan_audit_actuation(executor_task_id=<executor_task_id>, ...)`; the lower-level verdict primitive is `kanban_record_plan_audit_verdict(task_id=<executor_task_id>, ...)`.
- `metadata.round` is required on every verdict so actuator replay is idempotent.
- Rejected verdicts use `metadata.kind = "revise_plan"` or `"needs_user_decision"`.
- Revision tasks use round-scoped `idempotency_key`s so crash replay cannot duplicate planner/auditor cards.
- Detached human input uses `kanban_comment` plus `kanban_block(kind="needs_input")`; do not wait on interactive approval tools.
- The plan-auditor task calls `kanban_complete` after it records the verdict and applies the graph mutation.

## Task Graph

Create the smallest graph that can prove the plan before execution:

1. Root task: stores the user's coding goal, constraints, budget ceiling, and final acceptance criteria.
2. Planner task: writes a concrete implementation plan with files, tests, risks, and stop conditions.
3. Plan-auditor task: reviews the planner output and acts on the executor gate.
4. Executor task: carries `plan_audit_required=true`; performs the coding work only after approval.
5. Code-auditor/reviewer task: reviews diff, tests, and residual risk after execution.
6. Summarizer task: reports final outcome to the user when upstream tasks are done.

For a minimal MVP, the planner and plan-auditor can be the only upstream parents before the executor. Add reviewer and summarizer only when the coding goal needs post-change verification and final packaging.

## Creating Tasks

Use `kanban_create` for each task. Put enough context in `body` that a restarted worker can continue from the board alone.

Executor task requirements:

```json
{
  "title": "Execute approved plan: <short goal>",
  "assignee": "executor-profile",
  "parents": ["<planner_task_id>", "<plan_auditor_task_id_if_used_as_parent>"],
  "plan_audit_required": true,
  "plan_audit_max_rounds": 2,
  "budget_usd": 1.25,
  "skills": ["kanban-orchestrated-coding"]
}
```

Planner body must include:

- The user goal and non-goals.
- Repository/workspace path and branch/worktree expectations.
- Planned files and exact validation commands.
- A visible per-step or per-task budget expectation when budget is configured.
- The executor task id if the executor already exists; otherwise the planner must name where the orchestrator will insert it.

Plan-auditor body must include:

- The planner task id.
- The executor task id carrying `plan_audit_required`.
- The current audit round.
- The max rounds.
- Required verdict metadata shape.

## Plan-Auditor Actuator

The plan-auditor worker performs this sequence:

1. Read the planner task, executor task, parent summaries, comments, and recent events with `kanban_show`.
2. Decide binary verdict: approved or rejected.
3. Apply the verdict on the executor task id with `kanban_apply_plan_audit_actuation`. This records the verdict, creates revision tasks or blocks for human input when needed, and completes the current auditor task.

```json
{
  "executor_task_id": "<executor_task_id>",
  "root_task_id": "<root_task_id>",
  "approved": true,
  "reason": "Plan names files, tests, rollback, and budget guard.",
  "metadata": {"round": 1}
}
```

For rejection that should produce another planner pass:

```json
{
  "executor_task_id": "<executor_task_id>",
  "root_task_id": "<root_task_id>",
  "approved": false,
  "reason": "Plan lacks migration test and exact files.",
  "metadata": {"round": 1, "kind": "revise_plan"},
  "planner_assignee": "planner",
  "auditor_assignee": "plan-auditor"
}
```

For rejection that needs a human decision:

```json
{
  "executor_task_id": "<executor_task_id>",
  "root_task_id": "<root_task_id>",
  "approved": false,
  "reason": "Needs product decision between API A and API B.",
  "metadata": {"round": 1, "kind": "needs_user_decision"},
  "comment": "PLAN AUDIT NEEDS INPUT: choose between API A and API B."
}
```

4. If approved, the helper leaves the executor ready for the dispatcher and completes the auditor task.
5. If rejected with `revise_plan`, the helper creates planner and auditor revision tasks for the next round using deterministic idempotency keys.
6. If rejected with `needs_user_decision`, the helper comments on the executor and blocks it with `kind="needs_input"`.
7. If max rounds is exhausted, the helper comments and blocks the executor with `kind="needs_input"` rather than creating more revision tasks.
8. The helper completes the auditor task with a summary naming the executor id, round, verdict, and created revision cards.

## Idempotency Keys

Use this shape for revision tasks:

```text
koc:<root_task_id>:<executor_task_id>:plan-round:<round>:planner
koc:<root_task_id>:<executor_task_id>:plan-round:<round>:auditor
```

If the actuator crashes after recording a rejection but before creating revision cards, replay the same round. The verdict tool is idempotent by `(executor_task_id, round)`, and `kanban_create(idempotency_key=...)` must return existing revision cards instead of duplicating them.

On replay, verify all three:

- There is only one rejected event for the same executor id and round.
- The planner/auditor revision task ids are the same as before.
- The executor has not been blocked early by a double-counted rejection.

## Human Input

Detached workers must not use interactive approval prompts. They may not have a listener and can hang forever.

For a user decision:

1. Add a concise comment on the executor task explaining the decision needed.
2. Call `kanban_block(task_id=<executor_task_id>, kind="needs_input", reason=<short reason>)`.
3. Complete the current auditor task with metadata linking `executor_task_id`, `round`, and `kind`.

## Budget And Scope

Use per-task `budget_usd` for executor and reviewer tasks when budget is configured. Make the workflow ceiling visible in the root/planner text as:

```text
workflow ceiling = sum(step_cap * max_rounds)
```

This is a visible planning bound, not an account-wide or provider-wide budget. Unknown-cost and exhausted-budget behavior is enforced by the Kanban task/run budget gates.

## Executor Contract

Before editing, the executor must confirm:

- Its own task has a `plan_audit_approved` event on the executor task id.
- The approved plan is visible in parent summaries or comments.
- Workspace path and branch/worktree expectations match the task body.
- The work is within stated scope and budget.

After editing, the executor must complete with:

- Changed files.
- Validation commands and results.
- Any skipped validation and why.
- Residual risk.
- Created follow-up cards, if any, in `created_cards`.

## Code-Auditor Contract

The code auditor reviews the executor handoff, diff, and validation logs. It records findings as comments or completes with approval. If it needs a human decision, it uses `kanban_comment` plus `kanban_block(kind="needs_input")`.

The MVP does not need a second DB verdict model for code audit. Use comments, run summaries, and task status unless a later slice explicitly adds a typed code-audit primitive.

## Common Pitfalls

1. Recording verdict on the auditor task id. This leaves the executor gated forever. Always target the executor id.
2. Replaying a rejected round without `metadata.round`. This can double-count rejections and block early.
3. Creating revision tasks without `idempotency_key`. Crash replay can duplicate planner/auditor cards.
4. Leaving the auditor task running after actuation. The auditor must complete itself after verdict plus graph mutation.
5. Using interactive approval in a detached worker. Use Kanban block/comment instead.
6. Expanding into a new orchestrator. The board is already the workflow engine.

## Verification Checklist

- [ ] Executor cannot be claimed before `plan_audit_approved` on the executor task id.
- [ ] Recording a verdict on the auditor task id does not open the executor gate.
- [ ] Approved verdict on the executor task id opens the gate.
- [ ] Replayed rejection for the same round does not append a second rejected event.
- [ ] Revision planner/auditor tasks are idempotent by key.
- [ ] `max_rounds` routes to `blocked` plus `needs_input`, not an infinite loop.
- [ ] Plan-auditor task completes after actuation.
- [ ] Human input path uses Kanban comment/block, not interactive approval.
