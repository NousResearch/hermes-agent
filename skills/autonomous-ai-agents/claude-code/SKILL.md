---
name: claude-code
description: Delegate coding tasks to Claude Code (Anthropic's CLI agent). Use for building features, refactoring, PR reviews, and iterative coding. Requires the claude CLI installed.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [Coding-Agent, Claude, Anthropic, Code-Review, Refactoring]
    related_skills: [codex, hermes-agent]
---

# Claude Code

Delegate coding tasks to [Claude Code](https://docs.anthropic.com/en/docs/claude-code) via the Hermes terminal. Claude Code is Anthropic's autonomous coding agent CLI.

## Prerequisites

- Claude Code installed: `npm install -g @anthropic-ai/claude-code`
- Authenticated: run `claude` once to log in
- Use `pty=true` in terminal calls — Claude Code is an interactive terminal app

## One-Shot Tasks

```
terminal(command="claude 'Add error handling to the API calls'", workdir="/path/to/project", pty=true)
```

For quick scratch work:
```
terminal(command="cd $(mktemp -d) && git init && claude 'Build a REST API for todos'", pty=true)
```

## Background Mode (Long Tasks)

For tasks that take minutes, use background mode so you can monitor progress:

```
# Start in background with PTY
terminal(command="claude 'Refactor the auth module to use JWT'", workdir="~/project", background=true, pty=true)
# Returns session_id

# Monitor progress
process(action="poll", session_id="<id>")
process(action="log", session_id="<id>")

# Send input if Claude asks a question
process(action="submit", session_id="<id>", data="yes")

# Kill if needed
process(action="kill", session_id="<id>")
```

## PR Reviews

Clone to a temp directory to avoid modifying the working tree:

```
terminal(command="REVIEW=$(mktemp -d) && git clone https://github.com/user/repo.git $REVIEW && cd $REVIEW && gh pr checkout 42 && claude 'Review this PR against main. Check for bugs, security issues, and style.'", pty=true)
```

Or use git worktrees:
```
terminal(command="git worktree add /tmp/pr-42 pr-42-branch", workdir="~/project")
terminal(command="claude 'Review the changes in this branch vs main'", workdir="/tmp/pr-42", pty=true)
```

## Parallel Work

Spawn multiple Claude Code instances for independent tasks:

```
terminal(command="claude 'Fix the login bug'", workdir="/tmp/issue-1", background=true, pty=true)
terminal(command="claude 'Add unit tests for auth'", workdir="/tmp/issue-2", background=true, pty=true)

# Monitor all
process(action="list")
```

## Key Flags

| Flag | Effect |
|------|--------|
| `claude 'prompt'` | One-shot task, exits when done |
| `claude --dangerously-skip-permissions` | Auto-approve all file changes |
| `claude --model <model>` | Use a specific model |

## Rules

1. **Always use `pty=true`** — Claude Code is an interactive terminal app and will hang without a PTY
2. **Use `workdir`** — keep the agent focused on the right directory
3. **Background for long tasks** — use `background=true` and monitor with `process` tool
4. **Don't interfere** — monitor with `poll`/`log`, don't kill sessions because they're slow
5. **Report results** — after completion, check what changed and summarize for the user
