---
name: opencode-sdd-orchestrator
description: Hermes-only orchestration workflow for SDD implementation via OpenCode in isolated git worktrees, with fail-closed agent checks and PR handoff.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [opencode, sdd, orchestrator, worktree, pull-request]
    related_skills: [opencode, github-pr-workflow, writing-plans]
---

# OpenCode SDD Orchestrator

## When to use

Use this workflow when the user wants Hermes to coordinate delivery while OpenCode performs the code implementation.

Typical triggers:
- "Orchestrate this task with OpenCode"
- "Use SDD orchestrator agent"
- "Create branch/worktree, delegate coding, open PR"

## Non-negotiable rule

**Hermes orchestrates, OpenCode codes.**

- Hermes: validates prerequisites, prepares branch/worktree, writes brief, executes OpenCode, verifies outcomes, opens PR, returns PR URL.
- OpenCode: performs code changes described in the brief and leaves the required commit(s) on the isolated branch for helper handoff.

If prerequisites fail, **stop**. Do not improvise with another agent.

## Required artifacts in this skill

- Runbook: `references/runbook.md`
- Brief template: `templates/task-brief.md`
- PR body template: `templates/pr-body.md`

## Mandatory fail-closed checks

Before any coding run, Hermes must verify all of the following:

1. OpenCode CLI exists and is callable.
2. Git repo is clean enough to branch/worktree safely.
3. GitHub auth is available for PR creation (if PR is requested).
4. Requested OpenCode agent is exactly available.
5. Workflow starts from the main repository checkout, not from an already-linked worktree root.

### Critical alias guard (explicit)

On this machine:
- ✅ valid: `sdd-orchestrator`
- ❌ invalid: `sdd-orquesador`

`sdd-orquesador` does **NOT** exist and can silently fall back to `build` in OpenCode flows.

Therefore:
- If user asks for `sdd-orquesador`, correct to `sdd-orchestrator` and require explicit confirmation if ambiguity remains.
- If runtime output indicates fallback/default agent usage, **abort the workflow** and report failure.

## Workflow (must follow in order)

1. Read `references/runbook.md` and execute it step-by-step.
2. Validate prerequisites first (no coding before passing checks).
3. Validate repository safety gates before any branch/worktree action (clean tree + no in-progress rebase/merge/cherry-pick/revert).
4. Create isolated git worktree + dedicated branch.
5. Provide a **fully populated** brief file based on `templates/task-brief.md` (template is a starter, helper does not auto-fill placeholders).
6. Launch OpenCode with `--agent sdd-orchestrator` (exact string).
7. Verify OpenCode run used requested agent (no fallback).
8. Run the explicit verification commands from the populated brief and collect evidence.
9. Confirm acceptance criteria evidence is present in OpenCode output/report.
10. Ensure OpenCode leaves committed work in the isolated branch; this helper refuses push/PR handoff when no commits exist relative to base.
11. Create PR using a populated file derived from `templates/pr-body.md` (required when PR creation is requested) only after verification passes.
12. Do not silently reuse an existing PR; if `gh pr create` reports a duplicate PR, stop and require explicit user action.
13. Return PR URL to the user.

## Pitfalls

- Running OpenCode in main checkout instead of isolated worktree.
- Starting the helper from a linked worktree root instead of the main checkout.
- Forgetting to pin `--agent sdd-orchestrator`.
- Accepting silent fallback to `build`.
- Creating PR without verifying acceptance criteria/tests from brief.
- Returning "done" without the PR URL.

## Verification checklist

Before final response, Hermes must confirm:

- [ ] Prerequisite checks passed.
- [ ] Worktree and branch were created and used.
- [ ] Brief file exists and is fully populated.
- [ ] OpenCode executed with `sdd-orchestrator` (verified, not assumed).
- [ ] Brief-defined verification commands were executed and passed.
- [ ] OpenCode left commit(s) on the isolated branch before push/PR handoff.
- [ ] Acceptance criteria evidence is present and mapped.
- [ ] PR was created successfully.
- [ ] Final response includes PR URL.
