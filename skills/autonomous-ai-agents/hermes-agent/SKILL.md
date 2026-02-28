---
name: hermes-agent-spawning
description: Spawn additional Hermes Agent instances as autonomous subprocesses for independent long-running tasks. Supports non-interactive one-shot mode (-q) and interactive PTY mode for multi-turn collaboration. Different from delegate_task — this runs a full separate hermes process.
version: 1.1.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [Agent, Hermes, Multi-Agent, Orchestration, Subprocess, Interactive]
    homepage: https://github.com/NousResearch/hermes-agent
    related_skills: [claude-code, codex]
---

# Spawning Hermes Agent Instances

Run additional Hermes Agent processes as autonomous subprocesses. Unlike `delegate_task` (which spawns lightweight subagents sharing the same process), this launches fully independent `hermes` CLI processes with their own sessions, tools, and terminal environments.

## When to Use This vs delegate_task

| Feature | `delegate_task` | Spawning `hermes` process |
|---------|-----------------|--------------------------|
| Context isolation | Separate conversation, shared process | Fully independent process |
| Tool access | Subset of parent's tools | Full tool access (all toolsets) |
| Session persistence | Ephemeral (no DB entry) | Full session logging + DB |
| Duration | Minutes (bounded by parent's loop) | Hours/days (runs independently) |
| Monitoring | Parent waits for result | Background process, monitor via `process` tool |
| Interactive | No | Yes (PTY mode supports back-and-forth) |
| Use case | Quick parallel subtasks | Long autonomous missions, interactive collaboration |

## Prerequisites

- `hermes` CLI installed and on PATH
- API key configured in `~/.hermes/.env`

### Installation

Requires an interactive shell (the installer runs a setup wizard):

```
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

This installs uv, Python 3.11, clones the repo, sets up the venv, and launches an interactive setup wizard to configure your API provider and model. See the [GitHub repo](https://github.com/NousResearch/hermes-agent) for details.

## Resuming Previous Sessions

Resume a prior CLI session instead of starting fresh. Useful for continuing long tasks across process restarts:

```
# Resume the most recent CLI session
terminal(command="hermes --continue", background=true, pty=true)

# Resume a specific session by ID (shown on exit)
terminal(command="hermes --resume 20260225_143052_a1b2c3", background=true, pty=true)
```

The full conversation history (messages, tool calls, responses) is restored from SQLite. The agent sees everything from the previous session.

## Mode 1: One-Shot Query (-q flag)

Run a single query non-interactively. The agent executes, does its work, and exits:

```
terminal(command="hermes chat -q 'Research the latest GRPO training papers and write a summary to ~/research/grpo.md'", timeout=300)
```

Background for long tasks:
```
terminal(command="hermes chat -q 'Set up CI/CD for ~/myapp'", background=true)
# Returns session_id, monitor with process tool
```

## Mode 2: Interactive PTY Session

Launch a full interactive Hermes session with PTY for back-and-forth collaboration. You can send messages, review its work, give feedback, and steer it.

Note: Hermes uses prompt_toolkit for its CLI UI. Through a PTY, this works because ptyprocess provides a real terminal — input sent via `submit` arrives as keystrokes. The output log will contain ANSI escape sequences from the UI rendering — focus on the text content, not the formatting.

```
# Start interactive hermes in background with PTY
terminal(command="hermes", workdir="~/project", background=true, pty=true)
# Returns session_id

# Send it a task
process(action="submit", session_id="<id>", data="Set up a Python project with FastAPI, add auth endpoints, and write tests")

# Wait for it to work, then check progress
process(action="log", session_id="<id>")

# Give feedback on what it produced
process(action="submit", session_id="<id>", data="The tests look good but add edge cases for invalid tokens")

# Check its response
process(action="log", session_id="<id>")

# Ask it to iterate
process(action="submit", session_id="<id>", data="Now add rate limiting middleware")

# When done, exit the session
process(action="submit", session_id="<id>", data="/exit")
```

### Interactive Collaboration Patterns

**Code review loop** — spawn hermes, send code for review, iterate on feedback:
```
terminal(command="hermes", workdir="~/project", background=true, pty=true)
process(action="submit", session_id="<id>", data="Review the changes in src/auth.py and suggest improvements")
# ... read its review ...
process(action="submit", session_id="<id>", data="Good points. Go ahead and implement suggestions 1 and 3")
# ... it makes changes ...
process(action="submit", session_id="<id>", data="Run the tests to make sure nothing broke")
```

**Research with steering** — start broad, narrow down based on findings:
```
terminal(command="hermes", background=true, pty=true)
process(action="submit", session_id="<id>", data="Search for the latest papers on KV cache compression techniques")
# ... read its findings ...
process(action="submit", session_id="<id>", data="The MQA approach looks promising. Dig deeper into that one and compare with GQA")
# ... more detailed research ...
process(action="submit", session_id="<id>", data="Write up everything you found to ~/research/kv-cache-compression.md")
```

**Multi-agent coordination** — spawn two agents working on related tasks, pass context between them:
```
# Agent A: backend
terminal(command="hermes", workdir="~/project/backend", background=true, pty=true)
process(action="submit", session_id="<agent-a>", data="Build a REST API for user management with CRUD endpoints")

# Agent B: frontend
terminal(command="hermes", workdir="~/project/frontend", background=true, pty=true)
process(action="submit", session_id="<agent-b>", data="Build a React dashboard that will connect to a REST API at localhost:8000/api/users")

# Check Agent A's progress, relay API schema to Agent B
process(action="log", session_id="<agent-a>")
process(action="submit", session_id="<agent-b>", data="Here's the API schema Agent A built: GET /api/users, POST /api/users, etc. Update your fetch calls to match.")
```

## Parallel Non-Interactive Instances

Spawn multiple independent agents for unrelated tasks:

```
terminal(command="hermes chat -q 'Research competitor landing pages and write a report to ~/research/competitors.md'", background=true)
terminal(command="hermes chat -q 'Audit security of ~/myapp and write findings to ~/myapp/SECURITY_AUDIT.md'", background=true)
process(action="list")
```

## With Custom Model

```
terminal(command="hermes chat -q 'Summarize this codebase' --model google/gemini-2.5-pro", workdir="~/project", background=true)
```

## Gateway Cron Integration

For scheduled autonomous tasks, use the `schedule_cronjob` tool instead of spawning processes — cron jobs handle delivery, retry, and persistence automatically.

## Key Differences Between Modes

| | `-q` (one-shot) | Interactive (PTY) | `--continue` / `--resume` |
|---|---|---|---|
| User interaction | None | Full back-and-forth | Full back-and-forth |
| PTY required | No | Yes (`pty=true`) | Yes (`pty=true`) |
| Multi-turn | Single query | Unlimited turns | Continues previous turns |
| Best for | Fire-and-forget tasks | Iterative work, steering | Picking up where you left off |
| Exit | Automatic after completion | Send `/exit` or kill | Send `/exit` or kill |

## Known Issues

- **Interactive PTY + prompt_toolkit**: The `submit` action sends `\n` (line feed) but prompt_toolkit in raw mode expects `\r` (carriage return) for Enter. Text appears in the prompt but never submits. **Workaround**: Use **tmux** instead of raw PTY mode. tmux's `send-keys Enter` sends the correct `\r`:

```
# Start hermes inside tmux
tmux new-session -d -s hermes-session -x 120 -y 40 "hermes"
sleep 10  # Wait for banner/startup

# Send messages
tmux send-keys -t hermes-session "your message here" Enter

# Read output
sleep 15  # Wait for LLM response
tmux capture-pane -t hermes-session -p

# Multi-turn: just send more messages and capture again
tmux send-keys -t hermes-session "follow-up message" Enter

# Exit when done
tmux send-keys -t hermes-session "/exit" Enter
tmux kill-session -t hermes-session
```

## Rules

1. **Use `-q` for autonomous tasks** — agent works independently and exits
2. **Use `pty=true` for interactive sessions** — required for the full CLI UI
3. **Use `submit` not `write`** — `submit` adds a newline (Enter), `write` doesn't
4. **Read logs before sending more** — check what the agent produced before giving next instruction
5. **Set timeouts for `-q` mode** — complex tasks may take 5-10 minutes
6. **Prefer `delegate_task` for quick subtasks** — spawning a full process has more overhead
7. **Each instance is independent** — they don't share conversation context with the parent
8. **Check results** — after completion, read the output files or logs the agent produced
