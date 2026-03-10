---
name: opencode
description: Delegate coding tasks to OpenCode CLI agent for feature implementation, refactoring, and PR review workflows. Requires the opencode CLI installed and authenticated.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [Coding-Agent, OpenCode, Autonomous, Refactoring, Code-Review]
    related_skills: [claude-code, codex, hermes-agent]
---

# OpenCode CLI

Use OpenCode as an autonomous coding worker orchestrated by Hermes.

## When to use

- User asks to run work through OpenCode specifically
- You want an external coding agent to implement/refactor/review code
- You need long-running coding sessions with progress monitoring

## Prerequisites

- OpenCode installed and on PATH (`opencode`)
- OpenCode authenticated (`opencode auth`)
- A git repository for code tasks
- `pty=true` for interactive sessions

## Quick reference

One-shot execution:

terminal(command="opencode run 'Add retry logic to API calls and update tests'", workdir="~/project", pty=true)

Interactive background session:

terminal(command="opencode", workdir="~/project", background=true, pty=true)
process(action="submit", session_id="<id>", data="Implement OAuth refresh flow and add tests")
process(action="log", session_id="<id>")
process(action="submit", session_id="<id>", data="/exit")

## Procedure

1. Verify OpenCode availability:
   - `terminal(command="opencode --version")`
2. Use one-shot mode (`opencode run`) for bounded tasks.
3. Use interactive mode (`opencode`) for iterative collaboration.
4. For long tasks, run in background and monitor with `process(action="poll"|"log")`.
5. Summarize resulting file changes and test outcomes before reporting completion.

## Parallel work pattern

Use separate workdirs/worktrees to avoid collisions:

terminal(command="opencode run 'Fix issue #101 and commit'", workdir="/tmp/issue-101", background=true, pty=true)
terminal(command="opencode run 'Add parser regression tests and commit'", workdir="/tmp/issue-102", background=true, pty=true)
process(action="list")

## Pitfalls

- `pty=true` is required for interactive `opencode` sessions.
- If command behavior differs between shells, check which binary is being resolved:
  - `terminal(command="which -a opencode")`
- If OpenCode asks follow-up questions, respond with `process(action="submit", ...)`.

## Verification

- Smoke test:
  - `terminal(command="opencode run 'Respond with exactly: OPENCODE_SMOKE_OK'", pty=true)`
- Confirm output contains `OPENCODE_SMOKE_OK`.
- Confirm any requested code changes and tests were applied successfully.
