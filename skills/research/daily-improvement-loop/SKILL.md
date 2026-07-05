---
name: daily-improvement-loop
description: "Turn daily Hermes/WELTBERG research into short proposals, approval gates, repo-backed implementation, and reusable skills."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [daily-research, self-improvement, cron, approval, proposals]
    related_skills: [hermes-agent, hermes-self-improvement]
---

# Daily Improvement Loop

Use this for recurring research jobs that improve Hermes or the WELTBERG AI stack.

## Daily job output

Keep output short and scannable:

1. Top 3 findings only.
2. For each: value, risk, likely files, verification path.
3. Recommend one next action.
4. Ask for approval before implementation.

## Implementation gate

Do not silently change code from a research job. Implementation starts only when the user approves or explicitly asks for autonomous execution.

## Durable learning

When the same finding repeats:

- Create or patch a skill if it is a reusable procedure.
- Patch docs if it is product/user knowledge.
- Create a repo issue/PR if it requires code.
- Do not save stale task progress to memory.

## Verification

A good daily loop is working when:

- The cron job is active.
- The output is short enough for WhatsApp.
- Findings become skills/docs/PRs instead of disappearing in chat history.
- Every implementation has tests or a clear verification command.
