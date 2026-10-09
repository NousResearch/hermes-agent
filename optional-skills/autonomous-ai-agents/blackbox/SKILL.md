---
name: blackbox
description: Delegate coding tasks to the Blackbox AI multi-model CLI.
version: 1.0.1
author: Hermes Agent (Nous Research)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Coding-Agent, Blackbox, Multi-Agent, Judge, Multi-Model]
    related_skills: [claude-code, codex, hermes-agent]
---

# Blackbox CLI

Delegate coding tasks to [Blackbox AI](https://www.blackbox.ai/) via the Hermes terminal. Blackbox is a multi-model coding agent CLI that dispatches tasks to multiple LLMs (Claude, Codex, Gemini, Blackbox Pro) and uses a judge to select the best implementation.

The CLI (npm `@blackbox_ai/blackbox-cli`, binary `blackbox`) is a TypeScript coding agent (forked from Gemini CLI) and supports interactive sessions, non-interactive one-shots, checkpointing, MCP, and vision model switching.

## Prerequisites

- Node.js 20+ installed
- Blackbox CLI installed: `npm install -g @blackbox_ai/blackbox-cli` (binary: `blackbox`)
- API key from [app.blackbox.ai/dashboard](https://app.blackbox.ai/dashboard)
- Configured: run `blackbox configure` and enter your API key
- Use `pty=true` in terminal calls — Blackbox CLI is an interactive terminal app

## One-Shot Tasks

```
terminal(command="blackbox --prompt 'Add JWT authentication with refresh tokens to the Express API'", workdir="/path/to/project", pty=true)
```

For quick scratch work:
```
terminal(command="cd $(mktemp -d) && git init && blackbox --prompt 'Build a REST API for todos with SQLite'", pty=true)
```

## Background Mode (Long Tasks)

For tasks that take minutes, use background mode so you can monitor progress:

```
# Start in background with PTY
terminal(command="blackbox --prompt 'Refactor the auth module to use OAuth 2.0'", workdir="~/project", background=true, pty=true)
# Returns session_id

# Monitor progress
process(action="poll", session_id="<id>")
process(action="log", session_id="<id>")

# Send input if Blackbox asks a question
process(action="submit", session_id="<id>", data="yes")

# Kill if needed
process(action="kill", session_id="<id>")
```

## Checkpoints & Resume

Blackbox CLI has built-in checkpoint support for pausing and resuming tasks:

```
# After a task completes, Blackbox shows a checkpoint tag
# Resume with a follow-up task:
terminal(command="blackbox --resume-checkpoint 'task-abc123-2026-03-06' --prompt 'Now add rate limiting to the endpoints'", workdir="~/project", pty=true)
```

## Session Commands

During an interactive session, use these commands:

| Command | Effect |
|---------|--------|
| `/compress` | Shrink conversation history to save tokens |
| `/clear` | Wipe history and start fresh |
| `/stats` | View current token usage |
| `Ctrl+C` | Cancel current operation |

## PR Reviews

Clone to a temp directory to avoid modifying the working tree:

```
terminal(command="REVIEW=$(mktemp -d) && git clone https://github.com/user/repo.git $REVIEW && cd $REVIEW && gh pr checkout 42 && blackbox --prompt 'Review this PR against main. Check for bugs, security issues, and code quality.'", pty=true)
```

## Parallel Work

Spawn multiple Blackbox instances for independent tasks:

```
terminal(command="blackbox --prompt 'Fix the login bug'", workdir="/tmp/issue-1", background=true, pty=true)
terminal(command="blackbox --prompt 'Add unit tests for auth'", workdir="/tmp/issue-2", background=true, pty=true)

# Monitor all
process(action="list")
```

## Multi-Model Mode

Blackbox's unique feature is running the same task through multiple models and judging the results. Configure which models to use via `blackbox configure` — select multiple providers to enable the Chairman/judge workflow where the CLI evaluates outputs from different models and picks the best one.

## Key Flags

| Flag | Effect |
|------|--------|
| `--prompt "task"` (`-p`) | Non-interactive one-shot execution |
| `--resume-checkpoint "tag"` | Resume from a saved checkpoint |
| `--yolo` (`-y`) | Auto-approve all actions and model switches |
| `--vlm-switch-mode <mode>` | Image-handling: `once`, `session`, or `persist` |
| `-c, --checkpointing` | Enable checkpointing of file edits |
| `blackbox configure` | Change settings, providers, models |
| `blackbox update` | Update the CLI to the latest version |
| `blackbox mcp` | Manage MCP servers |
| `blackbox extensions` | Manage CLI extensions |
| `blackbox voice <action>` / `blackbox shortcut` | Configure voice input / the `b` shortcut |

## Vision Support

Blackbox automatically detects images in input and can switch to multimodal analysis. VLM modes:
- `"once"` — Switch model for current query only
- `"session"` — Switch for entire session
- `"persist"` — Stay on current model (no switch)

## Token Limits

Control token usage via `.blackboxcli/settings.json`:
```json
{
  "sessionTokenLimit": 32000
}
```

## Rules

1. **Always use `pty=true`** — Blackbox CLI is an interactive terminal app and will hang without a PTY
2. **Use `workdir`** — keep the agent focused on the right directory
3. **Background for long tasks** — use `background=true` and monitor with `process` tool
4. **Don't interfere** — monitor with `poll`/`log`, don't kill sessions because they're slow
5. **Report results** — after completion, check what changed and summarize for the user
6. **Credits cost money** — Blackbox uses a credit-based system; multi-model mode consumes credits faster
7. **Check prerequisites** — verify `blackbox` CLI is installed before attempting delegation
