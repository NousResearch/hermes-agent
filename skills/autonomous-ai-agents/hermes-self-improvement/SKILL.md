---
name: hermes-self-improvement
description: "Run Hermes self-improvement safely: context refresh, planner/worker/reviewer orchestration, worktree-first changes, GitHub review gates, and daily research loops."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Hermes, Self-Improvement, Orchestration, GitHub, Review-Gates, Cron]
    related_skills: [hermes-agent, context-refresh-loop, orchestration-roles, github-flow-automation, daily-improvement-loop, requesting-code-review, systematic-debugging]
---

# Hermes Self-Improvement

Use this skill when improving Hermes itself or turning daily research findings into repo-backed changes. The priority is verified code in the Hermes repo, not chat-only plans.

## Hard boundaries

1. Work only in the Hermes repository unless the user explicitly names another repo.
2. Start with `git status --short`, `git branch --show-current`, and `git remote -v`.
3. If unrelated dirty files exist, do not edit them. Mention them as pre-existing and keep the diff scoped.
4. Prefer a feature branch or worktree for any non-trivial change.
5. Never expose secrets. Keep credentials in `.env` or auth stores, not code or docs.
6. Do not push, merge, delete branches, or restart services unless the user asked for that side effect or it is required and safe.

## Phase loop

For every phase, keep the main chat short and state:

- Goal
- Files touched
- Verification command
- Result
- Next action

Before moving to the next phase, verify the current phase with a real command.

## Context refresh loop

Use this for long sessions or after meaningful milestones:

1. Write a 3–7 bullet state summary in the chat or a plan file.
2. Include current repo, branch, dirty files, completed checks, blockers, and next command.
3. If the conversation is large, use `session_search` to rehydrate only the needed prior context.
4. Keep permanent behavior in repo docs/skills; keep temporary task state out of memory.

Definition of done:

- A fresh agent can continue from the summary without rereading the whole conversation.
- The summary names the exact repo and excludes unrelated projects.

## Planner / worker / reviewer orchestration

Use three roles, even when one Hermes instance performs them:

1. Planner
   - Define the smallest safe change.
   - List acceptance criteria and risks.
   - Identify verification commands before editing.

2. Worker
   - Modify only the files needed for that change.
   - Avoid broad refactors unless the phase is explicitly a refactor.
   - Keep unrelated dirty files untouched.

3. Reviewer
   - Re-read the diff.
   - Run targeted tests.
   - Check scope, safety, and rollback path.
   - Decide: pass, fix, or stop and ask.

Definition of done:

- Every change has a stated acceptance criterion and a test or command that was run.
- Reviewer checks include `git diff --stat` and a targeted test command.

## Worktree-first code changes

For complex or parallel self-improvement:

1. `git fetch origin`
2. `git worktree add -b <type>/<short-name> ../hermes-agent-<short-name> origin/main`
3. Work inside that worktree.
4. Run targeted tests there.
5. Merge or cherry-pick only after review.

Use direct edits in the current tree only for small, urgent, already-scoped changes.

## MCP and tool safety gates

When adding or changing external access:

1. Default to local-only or read-only.
2. Add explicit allowlists for remote hosts or server names.
3. Reject exfiltration-shaped stdio configs before any spawn/probe path.
4. Prefer environment placeholders such as `${MCP_SERVICE_API_KEY}` over literal tokens.
5. Verify both the save path and runtime loading path.

Minimum MCP verification:

- Clean local stdio server remains allowed.
- Shell plus network-exfiltration args are rejected.
- Remote URL host is blocked unless allowlisted.
- Placeholder credentials are allowed; literal token-shaped values are rejected or moved out of config.

## GitHub / repo automation gates

For repo-backed changes:

1. Branch from a clean base.
2. Keep one logical change per PR.
3. Before commit, run:
   - `git diff --stat`
   - targeted tests
   - a scope check for unrelated files
4. Commit with a conventional commit message.
5. Open a PR with summary and test plan.
6. Do not merge until CI and review gates pass.

PR body checklist:

- Summary
- Files changed
- Test plan with real command output
- Risk / rollback
- Unrelated dirty files, if any

## Daily research improvement loop

A daily research job should produce proposals, not silent code changes.

Prompt shape:

1. Inspect current Hermes docs/repo and recent session context.
2. Find at most 3 actionable improvements for Hermes or the WELTBERG AI stack.
3. For each item, include value, risk, files likely touched, and verification path.
4. Ask for approval before implementation.
5. If the same useful workflow repeats, turn it into a repo-backed skill or docs change.

Output must stay short and scannable.

## Verification checklist before reporting done

- Correct repo confirmed.
- Unrelated projects untouched.
- Dirty files explained.
- Tests or equivalent commands run.
- Diff reviewed.
- User-facing summary includes what changed and what remains.
