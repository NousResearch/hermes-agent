---
name: cursor
description: Delegate coding tasks to Cursor's headless agent CLI (cursor-agent). Supports non-interactive print mode, interactive sessions, plan/ask modes, sandboxed execution, session resume, and cloud handoff. Requires the cursor-agent CLI and a Cursor account.
version: 1.0.0
author: Hermes Agent (Nous Research)
license: MIT
required_environment_variables:
  - name: CURSOR_API_KEY
    prompt: Cursor API key
    help: Generate one at https://cursor.com/integrations (Settings → API Keys). Only needed for headless/CI use; interactive installs can complete browser OAuth on first run.
    required_for: headless and CI use (skip if you'll authenticate via browser OAuth)
prerequisites:
  commands: [cursor-agent]
metadata:
  hermes:
    tags: [Coding-Agent, Cursor, Anthropic, OpenAI, PTY, Automation]
    related_skills: [claude-code, codex, blackbox, hermes-agent]
---

# Cursor CLI

Delegate coding tasks to [Cursor](https://cursor.com/cli) (the headless agent mode of the Cursor IDE) via the Hermes terminal. `cursor-agent` is an autonomous coding agent backed by Cursor's model routing — Anthropic Claude, OpenAI GPT, and Cursor's own models — and can read files, write code, run shell commands, and hand off long tasks to Cursor's Cloud Agent.

## When to use

- Building features
- Refactoring
- PR reviews and design analysis (`--mode plan`)
- Long-running tasks you want to push to Cloud Agent
- Multi-model experimentation (Cursor routes across providers)

## Prerequisites

- **Install (macOS/Linux/WSL):** `curl https://cursor.com/install -fsS | bash`
- **Install (Windows PowerShell):** `irm 'https://cursor.com/install?win32=true' | iex`
- **Auth:** first-run `cursor-agent` opens a browser for OAuth; or set `CURSOR_API_KEY` for headless boxes
- **Verify:** `cursor-agent --version`
- **Update:** `cursor-agent update`
- **Binary alias:** newer Cursor builds also expose the binary as `agent`. If `cursor-agent` is not on `$PATH`, try `command -v agent` and symlink (`ln -s "$(command -v agent)" ~/.local/bin/cursor-agent`) or substitute `agent` for `cursor-agent` in the commands below.
- Use `pty=true` in terminal calls — `cursor-agent` is an interactive terminal app and hangs without a PTY

## Two Orchestration Modes

`cursor-agent` supports both a non-interactive print mode and a full interactive REPL. Choose based on the task.

### Mode 1: Print Mode (`-p`) — Non-Interactive (PREFERRED for one-shots)

Print mode runs a one-shot task and exits when done. Cleanest integration path: no interactive prompts to navigate, output streams back to the terminal call.

```
terminal(command="cursor-agent -p 'Add JWT auth with refresh tokens to the Express API'", workdir="~/project", pty=true)
```

For scratch work, drop into a temp dir first:
```
terminal(command="cd $(mktemp -d) && git init && cursor-agent -p 'Build a REST API for todos with SQLite'", pty=true)
```

**When to use print mode:**
- One-shot coding tasks (fix a bug, add a feature, refactor)
- CI/CD automation
- Any task that doesn't need multi-turn back-and-forth

### Mode 2: Interactive Session — Multi-Turn

Drop the `-p` flag and `cursor-agent` opens an interactive REPL. Use `background=true` and the `process` tool to send follow-up turns.

```
# Start the session in the background
terminal(command="cursor-agent", workdir="~/project", background=true, pty=true)
# Returns session_id

# Send the first turn
process(action="submit", session_id="<id>", data="Refactor the auth module to use JWT")

# Send a follow-up after the first turn finishes
process(action="submit", session_id="<id>", data="Now add unit tests for the new JWT code")

# Watch progress
process(action="log", session_id="<id>")

# Exit when done
process(action="submit", session_id="<id>", data="/exit")
```

**When to use interactive mode:**
- Multi-turn iterative work (refactor → review → fix → test cycle)
- Open-ended exploration where each turn depends on the previous output

## Background Mode (Long Tasks)

For tasks that take minutes, background mode lets you monitor progress while keeping the parent agent free.

```
# Start in background with PTY
terminal(command="cursor-agent -p 'Refactor the auth module to use OAuth 2.0'", workdir="~/project", background=true, pty=true)
# Returns session_id

# Monitor progress
process(action="poll", session_id="<id>")
process(action="log", session_id="<id>")

# Send input if cursor-agent asks a question
process(action="submit", session_id="<id>", data="yes")

# Kill if needed
process(action="kill", session_id="<id>")
```

## Modes (`--mode`)

| Mode | Behavior |
|------|----------|
| `agent` (default) | Writes code, runs commands, makes edits |
| `plan` | Reasons and proposes a plan without modifying files. Use for design and PR review |
| `ask` | Q&A only. No file edits, no command execution |

```
terminal(command="cursor-agent --mode plan -p 'How should we split the monolith into services?'", workdir="~/project", pty=true)
```

## Sessions & Resume

`cursor-agent` persists chat history per workspace. Use this to revisit or extend prior work.

```
# List previous chats in this workspace
terminal(command="cursor-agent ls", workdir="~/project", pty=true)

# Resume the latest conversation
terminal(command="cursor-agent resume", workdir="~/project", pty=true)

# Resume a specific chat by ID
terminal(command="cursor-agent --resume='chat-abc123'", workdir="~/project", pty=true)

# Continue the most recent session in print mode
terminal(command="cursor-agent --continue -p 'Now add rate limiting to the endpoints'", workdir="~/project", pty=true)
```

## PR Reviews

Clone to a temp directory and use `--mode plan` so cursor-agent doesn't accidentally edit during review:

```
terminal(command="REVIEW=$(mktemp -d) && git clone https://github.com/user/repo.git $REVIEW && cd $REVIEW && gh pr checkout 42 && cursor-agent --mode plan -p 'Review this PR against main. Check for bugs, security issues, race conditions, and missing tests.'", pty=true)
```

## Parallel Issue Fixing with Worktrees

```
# Create worktrees
terminal(command="git worktree add -b fix/issue-78 /tmp/issue-78 main", workdir="~/project")
terminal(command="git worktree add -b fix/issue-99 /tmp/issue-99 main", workdir="~/project")

# Launch cursor-agent in each
terminal(command="cursor-agent -p 'Fix issue #78: <description>. Commit when done.'", workdir="/tmp/issue-78", background=true, pty=true)
terminal(command="cursor-agent -p 'Fix issue #99: <description>. Commit when done.'", workdir="/tmp/issue-99", background=true, pty=true)

# Monitor
process(action="list")

# After completion, push and create PRs
terminal(command="cd /tmp/issue-78 && git push -u origin fix/issue-78")
terminal(command="gh pr create --repo user/repo --head fix/issue-78 --title 'fix: ...' --body '...'")

# Cleanup
terminal(command="git worktree remove /tmp/issue-78", workdir="~/project")
```

## Cloud Handoff (`&` prefix)

Prepending `&` to a prompt pushes the task to Cursor's Cloud Agent — useful when the local box can't sit on a PTY for an hour or you want the work to keep going after Hermes exits.

```
terminal(command="cursor-agent -p '& Refactor the entire authentication subsystem to OAuth 2.0 with refresh tokens. Open a PR when done.'", workdir="~/project", pty=true)
```

The local terminal call returns once the task is dispatched; the Cloud Agent continues running on Cursor's infrastructure and reports back via the workspace.

## Key Flags

| Flag | Effect |
|------|--------|
| `-p`, `--prompt "task"` | Non-interactive print mode; exits when done |
| `--model <name>` | Pin a specific model (e.g., `gpt-5.2`, `claude-sonnet-4-6`) |
| `--mode {agent\|plan\|ask}` | Execution mode (default: `agent`) |
| `--sandbox {enabled\|disabled}` | Toggle sandboxed command execution (default: enabled) |
| `--output-format {text\|json}` | Plain text or structured JSON output |
| `--resume="<chat-id>"` | Resume a specific chat by ID |
| `--continue` | Continue the most recent chat in this workspace |
| `cursor-agent ls` | List previous chats |
| `cursor-agent resume` | Resume the latest chat |
| `cursor-agent update` | Update the CLI |

## Pitfalls

1. **Binary alias.** Newer Cursor builds rename the binary from `cursor-agent` to `agent`. If `command -v cursor-agent` fails, try `command -v agent` and either symlink it or substitute `agent` in commands.
2. **First-run requires a browser** for OAuth. On a headless box, set `CURSOR_API_KEY` before the first invocation or the agent will hang on a localhost callback URL.
3. **No `--workspace` flag.** The cwd is the workspace. Always pass `workdir` to `terminal()` instead of trying to flag the path.
4. **PTY is mandatory.** `cursor-agent` is a TUI; without `pty=true` it hangs.
5. **Cloud handoff is fire-and-forget.** A `&`-prefixed prompt dispatches to Cloud Agent and the local call returns. Don't expect file changes locally — pull them via `git fetch` once the cloud run completes.
6. **`--mode ask` cannot edit files.** If `cursor-agent` says it can't write to a path, check whether you accidentally launched ask mode.

## Rules

1. **Always use `pty=true`** — `cursor-agent` is an interactive terminal app and hangs without a PTY
2. **Use `workdir`** — there is no `--workspace` flag; the cwd is the workspace
3. **Prefer print mode (`-p`) for one-shots** — cleaner, no interactive prompts to navigate
4. **Background for long tasks** — use `background=true` and monitor with the `process` tool
5. **Use `--mode plan`** for review and design tasks where you don't want file edits
6. **Don't interfere** — monitor with `poll`/`log`; don't kill sessions because they're slow
7. **Verify the binary is on `$PATH`** before delegating: `command -v cursor-agent` (fall back to `agent` on newer builds)
8. **Report results** — after completion, summarize what changed for the user
