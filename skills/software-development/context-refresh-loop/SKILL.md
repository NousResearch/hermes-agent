---
name: context-refresh-loop
description: "Keep long Hermes sessions resumable: milestone summaries, rehydration, repo/status checkpoints, and context layering."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [context, compression, session-search, handoff, long-running-tasks]
    related_skills: [hermes-agent, plan, systematic-debugging]
---

# Context Refresh Loop

Use this when a task lasts more than one phase, spans multiple tool calls, or may hit context compression.

## Trigger points

Refresh context after:

1. A phase is completed.
2. A commit is created.
3. A blocker appears.
4. A repo or branch changes.
5. A context compression or handoff happens.

## Refresh format

Write 3-7 bullets containing:

- Active repo and branch.
- Dirty files and whether they are yours or pre-existing.
- Current goal and phase.
- Last verification command and result.
- Blockers or risks.
- Exact next command.

## Rehydrate before acting

When resuming:

1. Use `session_search` for prior decisions if the user references earlier work.
2. Run `git status --short && git branch --show-current` in the target repo.
3. Read the plan or skill that defines the active workflow.
4. Continue only after repo and scope match the user request.

## Boundaries

- Permanent behavior belongs in skills or project docs.
- Temporary task state belongs in chat summaries or plan files.
- Do not store task progress, PR numbers, or stale artifacts in memory.
- Do not switch repos without explicitly confirming the target from tool output.

## Definition of done

A fresh agent can continue from the latest summary without guessing the repo, branch, dirty files, phase, tests, or next action.
