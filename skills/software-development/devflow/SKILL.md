---
name: devflow
description: "Plan/verify -> develop -> verify workflow for scoped implementation tasks."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [development, planning, verification, workflow, qa]
    related_skills: [plan, verify, ultraqa, tdd, trace, requesting-code-review]
    created_by: agent
---

# Devflow

Use this skill when the user invokes `/devflow ...` or asks for a full development workflow that first plans and verifies the plan, then implements and verifies completion.

## Purpose

Make Hermes own the development loop:

1. understand the task,
2. create a concrete plan,
3. verify the plan against live repo/context,
4. implement only the approved/inferred scope,
5. verify with evidence,
6. update handoff or report clearly.

Other agents and tools may be used, but only as bounded helpers, implementers, or reviewers. Do not hand over overall orchestration to another harness unless the user explicitly asks.

## Workflow

### 1. Scope discovery

- Inspect current repo/config/handoff as needed before designing changes.
- Identify unrelated dirty files and preserve them.
- Infer acceptance criteria from the user's request. If ambiguity would change what files or systems you touch, ask a concise clarifying question. Otherwise proceed with a clearly stated assumption.

### 2. Plan

Create a short, actionable plan. For code changes include:

- goal,
- files likely to change,
- implementation approach,
- tests/validation commands,
- risks and rollback notes.

If the user asked to execute immediately, the plan may be saved as a checkpoint under `.hermes/plans/` and then execution should continue in the same turn.

### 3. Verify the plan before coding

Before editing, check that the plan is feasible:

- the target files/commands exist,
- the existing code has the expected extension points,
- planned tests are available,
- no obvious conflict with repo instructions or user preferences exists.

If the plan is wrong, revise it before coding.

### 4. Develop

- Make the smallest scoped edits needed.
- Prefer targeted patches over broad rewrites.
- Keep behavior in skills/docs when implementing workflow policy; keep code wrappers thin.
- Do not commit, push, deploy, or run destructive cleanup unless explicitly requested.

### 5. Verify implementation

Run fresh checks. Minimum expected evidence:

- targeted unit/integration tests for changed behavior,
- syntax/lint/format checks when available and relevant,
- `git diff --check` for touched files,
- direct inspection that acceptance criteria are represented in code/docs.

If a check fails, diagnose and iterate within scope. If blocked by environment, report BLOCKED with evidence and the exact remaining action.

### 6. Optional independent review gate

For material code changes, security-sensitive changes, or high-risk workflow changes, consider a bounded independent review. For this user, Claude Code Opus review should be constrained and exact when used:

```bash
claude -p 'Review only. Do not edit files.' --model claude-opus-4-6 --allowedTools 'Read' --max-turns 1
```

Use the review as input; Hermes still decides what to apply and verifies again.

### 7. Handoff and final report

When the repo uses `.hermes/handoff/current.md`, update it with:

- what changed,
- files touched,
- tests/checks run and exit codes,
- known caveats,
- next pickup prompt if relevant.

Final response should be concise and evidence-based:

1. Plan checkpoint
2. Changes made
3. Verification results with commands/exit codes
4. Final status: PASS / FAIL / BLOCKED
5. Any restart/reload needed

## Acceptance criteria template

Use or adapt this checklist:

- Plan was created and verified before implementation.
- Implementation changed only scoped files.
- Tests/checks relevant to changed behavior pass.
- `git diff --check` passes for touched files.
- Handoff/final summary records evidence and caveats.

## Pitfalls

- Do not confuse plan-only mode with devflow. Devflow continues after plan verification unless blocked.
- Do not claim broad quality gates passed unless they were actually run.
- Do not let a spawned agent or Claude Code session become the primary orchestrator by accident.
- Do not overwrite unrelated user changes in a dirty repo.
