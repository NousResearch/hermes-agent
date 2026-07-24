---
name: pi
description: Delegate coding tasks to Pi Coding Agent CLI. Use for building features, refactoring, PR reviews, and autonomous coding sessions. Requires the pi CLI installed and authenticated.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [Coding-Agent, Pi, AI-Agent, Refactoring, Code-Review]
    related_skills: [claude-code, codex, opencode, hermes-agent]
---

# Pi Coding Agent

Delegate coding tasks to [Pi](https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent) via the Hermes terminal. Pi is an open-source AI coding agent with interactive TUI and CLI modes.

## Prerequisites

- Pi installed: `npm install -g @mariozechner/pi-coding-agent`
- Authenticated: run `pi auth` to configure provider API keys
- Verify: `pi --version` should show the version
- Git repository recommended for code tasks
- Use `pty=true` for interactive TUI sessions

## Installation

```bash
# Install globally
npm install -g @mariozechner/pi-coding-agent

# Or use npx (no install)
npx @mariozechner/pi-coding-agent
```

## One-Shot Tasks

Use `pi` command with a prompt for bounded tasks:

```
terminal(command="pi 'Add input validation to the login form'", workdir="~/project")
```

For scratch work (creates temp git repo if needed):
```
terminal(command="cd $(mktemp -d) && git init && pi 'Build a todo app in React'", pty=true)
```

## Background Mode (Long Tasks)

For iterative work or long-running sessions:

```
# Start in background with PTY
terminal(command="pi", workdir="~/project", background=true, pty=true)
# Returns session_id

# Send a prompt
process(action="submit", session_id="<id>", data="Refactor the database layer to use connection pooling")

# Monitor progress
process(action="poll", session_id="<id>")
process(action="log", session_id="<id>")

# Send follow-up input
process(action="submit", session_id="<id>", data="Also add retry logic for failed connections")

# Exit cleanly — Ctrl+C
process(action="write", session_id="<id>", data="\x03")
# Or kill the process
process(action="kill", session_id="<id>")
```

## PR Reviews

Clone to a temp directory for safe review:

```
terminal(command="REVIEW=$(mktemp -d) && git clone https://github.com/user/repo.git $REVIEW && cd $REVIEW && gh pr checkout 42 && pi 'Review this PR against main. Check for bugs, security issues, and style problems.'", pty=true)
```

Or review in the current repo:
```
terminal(command="pi 'Review the changes in this branch vs main'", workdir="~/project", pty=true)
```

## Parallel Work

Spawn multiple Pi instances for independent tasks:

```
# Create worktrees for isolation
terminal(command="git worktree add -b fix/issue-78 /tmp/issue-78 main", workdir="~/project")
terminal(command="git worktree add -b fix/issue-99 /tmp/issue-99 main", workdir="~/project")

# Launch Pi in each
terminal(command="pi 'Fix issue #78: <description>. Commit when done.'", workdir="/tmp/issue-78", background=true, pty=true)
terminal(command="pi 'Fix issue #99: <description>. Commit when done.'", workdir="/tmp/issue-99", background=true, pty=true)

# Monitor all
process(action="list")
```

## Common Commands

| Command | Description |
|---------|-------------|
| `pi` | Start interactive TUI session |
| `pi 'prompt'` | One-shot task execution |
| `pi --version` | Show version |
| `pi auth` | Configure authentication |

## Key Flags

| Flag | Effect |
|------|--------|
| `--model <model>` | Use specific model |
| `--provider <provider>` | Use specific provider |

## Rules

1. **Always use `pty=true`** — Pi is an interactive terminal app and may hang without a PTY
2. **Git repo recommended** — Initialize with `git init` if working in temp directories
3. **Background for long tasks** — use `background=true` and monitor with `process` tool
4. **Don't interfere** — monitor with `poll`/`log`, be patient with long-running tasks
5. **Exit with Ctrl+C** — Send `\x03` via `process(action="write")` or use `process(action="kill")`
6. **Scope to single repo** — Avoid sharing one working directory across parallel Pi sessions

## Verification

Smoke test:

```
terminal(command="pi --version")
```

Success criteria:
- Command shows Pi version
- For code tasks: expected files changed and tests pass
