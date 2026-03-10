---
name: opencode
description: Delegate coding tasks to OpenCode CLI agent for feature implementation, refactoring, PR review, and long-running autonomous sessions. Requires the opencode CLI installed and authenticated.
version: 1.1.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [Coding-Agent, OpenCode, Autonomous, Refactoring, Code-Review]
    related_skills: [claude-code, codex, hermes-agent]
---

# OpenCode CLI

Use OpenCode as an autonomous coding worker orchestrated by Hermes terminal/process tools.

## When to Use

- User explicitly asks to use OpenCode
- You want an external coding agent to implement/refactor/review code
- You need long-running coding sessions with progress checks
- You want parallel task execution in isolated workdirs/worktrees

## Prerequisites

- OpenCode installed (`opencode`)
- Auth configured (`opencode auth list` should show at least one provider)
- Git repository for code tasks (recommended)
- `pty=true` for interactive sessions

## Binary Resolution (Important)

Shell environments may resolve different OpenCode binaries. If behavior differs between your terminal and Hermes, check:

terminal(command="which -a opencode")
terminal(command="opencode --version")

If needed, pin an explicit binary path in commands:

terminal(command="$HOME/.opencode/bin/opencode run '...'", workdir="~/project", pty=true)

## Quick Reference

One-shot task:

terminal(command="opencode run 'Add retry logic to API calls and update tests'", workdir="~/project", pty=true)

Interactive session (background):

terminal(command="opencode", workdir="~/project", background=true, pty=true)
process(action="submit", session_id="<id>", data="Implement OAuth refresh flow and add tests")
process(action="log", session_id="<id>")
process(action="submit", session_id="<id>", data="/exit")

## Common Flags

| Flag | Use |
|------|-----|
| `run 'prompt'` | One-shot execution and exit |
| `--continue` / `-c` | Continue the last OpenCode session |
| `--session <id>` / `-s` | Continue a specific session |
| `--agent <name>` | Choose OpenCode agent/profile |
| `--model provider/model` | Force specific model |
| `--format json` | Machine-readable output/events |

## Procedure

1. Verify tool readiness:
   - `terminal(command="opencode --version")`
   - `terminal(command="opencode auth list")`
2. For bounded tasks, use `opencode run '...'`.
3. For iterative tasks, start `opencode` with `background=true, pty=true`.
4. Monitor long tasks with `process(action="poll"|"log")`.
5. If OpenCode asks for input, respond via `process(action="submit", ...)`.
6. Summarize file changes, test results, and next steps back to user.

## PR Review Workflow

Safe review in temporary clone:

terminal(command="REVIEW=$(mktemp -d) && git clone https://github.com/user/repo.git $REVIEW && cd $REVIEW && gh pr checkout 42 && opencode run 'Review this PR vs main. Report bugs, security risks, test gaps, and style issues.'", pty=true)

## Parallel Work Pattern

Use separate workdirs/worktrees to avoid collisions:

terminal(command="opencode run 'Fix issue #101 and commit'", workdir="/tmp/issue-101", background=true, pty=true)
terminal(command="opencode run 'Add parser regression tests and commit'", workdir="/tmp/issue-102", background=true, pty=true)
process(action="list")

## Pitfalls

- Interactive `opencode` sessions require `pty=true`.
- PATH mismatch can select the wrong OpenCode binary/model config.
- If OpenCode appears stuck, inspect logs before killing:
  - `process(action="log", session_id="<id>")`
- Avoid sharing one working directory across parallel OpenCode sessions.

## Verification

Smoke test:

terminal(command="opencode run 'Respond with exactly: OPENCODE_SMOKE_OK'", pty=true)

Success criteria:
- Output includes `OPENCODE_SMOKE_OK`
- Command exits without provider/model errors
- For code tasks: expected files changed and tests pass

## Rules

1. Prefer `opencode run` for one-shot automation.
2. Use interactive background mode only when iteration is needed.
3. Always scope OpenCode sessions to a single repo/workdir.
4. For long tasks, provide progress updates from `process` logs.
5. Report concrete outcomes (files changed, tests, remaining risks).
