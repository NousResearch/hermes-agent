---
name: orchestration-roles
description: "Planner/Worker/Reviewer execution loop with worktree-first isolation and hard verification gates."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [orchestration, planner, worker, reviewer, worktree, quality-gates]
    related_skills: [requesting-code-review, test-driven-development, systematic-debugging]
---

# Orchestration Roles

Use this for non-trivial code work, especially Hermes self-improvement.

## Role contract

### Planner

- State the smallest useful deliverable.
- Identify files likely to change.
- Define acceptance criteria before editing.
- Choose the verification commands.

### Worker

- Implement exactly one scoped deliverable.
- Avoid unrelated refactors.
- Touch only intentional files.
- Preserve pre-existing dirty files.

### Reviewer

- Inspect `git diff --stat` and the changed files.
- Run targeted tests.
- Check security, rollback, and scope.
- Return pass/fix/stop.

## Worktree-first rule

Use a git worktree for parallel or risky work:

```bash
git fetch origin
git worktree add -b <type>/<short-name> ../hermes-agent-<short-name> origin/main
```

Direct edits in the current tree are acceptable only for small, already-scoped changes.

## Hard gates

Do not commit until:

1. The diff is scoped to the deliverable.
2. Tests or a justified equivalent command ran.
3. A reviewer pass happened, either via independent subagent or explicit self-review checklist.
4. Unrelated dirty files are excluded from staging.

## Failure handling

- If a fix fails three times, stop and question the architecture.
- If tests cannot run, state the blocker and run the closest deterministic substitute.
- If the repo target is unclear, inspect before asking; ask only if inspection cannot resolve it.
