---
name: codex
description: Delegate coding tasks to OpenAI Codex CLI agent. Use for building features, refactoring, PR reviews, and batch issue fixing. Requires the codex CLI and a git repository.
version: 2.0.0
author: Hermes Agent + Nous Research
license: MIT
metadata:
  hermes:
    tags: [Coding-Agent, Codex, OpenAI, Code-Review, Refactoring]
    related_skills: [claude-code, hermes-agent, opencode]
---

# Codex CLI — OpenAI 官方编程 Agent

Delegate coding tasks to [Codex](https://github.com/openai/codex) via the Hermes terminal. Codex is OpenAI's autonomous coding agent CLI that reads files, edits code, runs commands, and manages git workflows.

## 官方资源

- Docs: https://developers.openai.com/codex
- CLI Reference: https://developers.openai.com/codex/cli/reference
- Features: https://developers.openai.com/codex/cli/features
- GitHub: https://github.com/openai/codex

## Prerequisites

- **Install:** `npm install -g @openai/codex` or `brew install codex`
- **Auth:** `codex login` (ChatGPT OAuth), `codex login --device-auth` (headless), or `printenv OPENAI_API_KEY | codex login --with-api-key`
- **Check auth:** `codex login status`
- **Version:** `codex --version`
- **Must run inside a git repository** — Codex refuses outside git dirs
- **Use `pty=true`** in terminal calls — Codex is an interactive TUI app

## Two Core Modes

### Interactive Mode (TUI)

```bash
codex
codex "Explain this codebase"
codex "Build a snake game in Python"
```

### Non-Interactive Mode (`exec`) — PREFERRED for automation

```bash
codex exec "fix the auth bug"
codex e "add dark mode"          # alias
```

## Key Flags

| Flag | Values | Description |
|------|--------|-------------|
| `--full-auto` | boolean | Shortcut: `--ask-for-approval on-request` + `--sandbox workspace-write`. Recommended for building. |
| `--ask-for-approval, -a` | `untrusted / on-request / never` | Human approval control |
| `--sandbox, -s` | `read-only / workspace-write / danger-full-access` | Sandbox policy |
| `--model, -m` | string | Override configured model, e.g. `gpt-5.4` |
| `--dangerously-bypass-approvals-and-sandbox, --yolo` | boolean | No sandbox, no approvals. Fastest, most dangerous. |
| `--cd, -C` | path | Set working directory |
| `--skip-git-repo-check` | | Allow running outside git repos |
| `--json` | | Output newline-delimited JSON events |
| `--output-last-message, -o` | path | Write final message to file |
| `--output-schema` | path | JSON Schema for structured response validation |
| `--ephemeral` | | Don't persist session files |
| `--search` | `cached / live / false` | Web search mode (default: `"cached"`) |
| `--oss` | boolean | Use Ollama local open-source model |
| `--profile, -p` | string | Load config profile from `~/.codex/config.toml` |
| `--remote` | `ws://host:port` | Connect TUI to remote app-server |
| `--remote-auth-token-env` | `ENV_VAR` | Bearer token for `--remote` connections |

## Models

| Model | Description |
|-------|-------------|
| `gpt-5.4` | **Recommended** — frontier coding + reasoning + computer use |
| `gpt-5.3-codex` | Standard Codex tasks |
| `gpt-5.3-codex-spark` | ChatGPT Pro subscribers (research preview) |

## Sandbox Constraints

| Level | Writable paths |
|-------|---------------|
| `read-only` | None |
| `workspace-write` | workdir, `/tmp`, `$TMPDIR`, `~/.codex/memories` |
| `danger-full-access` | Everything (dangerous) |

**Recommended:** `--full-auto` enables `workspace-write` automatically.

## One-Shot Tasks

```bash
# Basic
terminal(command="codex exec --full-auto 'Add dark mode toggle'", workdir="~/project", pty=true)

# With specific model
terminal(command="codex exec --full-auto --model gpt-5.4 'Refactor the auth module'", workdir="~/project", pty=true)

# Scratch work (needs git repo)
terminal(command="cd $(mktemp -d) && git init && codex exec --full-auto 'Build a snake game'", pty=true)

# Skip git repo check
terminal(command="codex exec --skip-git-repo-check 'Analyze this file'", workdir="/tmp", pty=true)
```

## Background Mode (Long Tasks)

```bash
# Start in background with PTY
terminal(command="codex exec --full-auto 'Refactor the entire backend'", workdir="~/project", background=true, pty=true)
# Returns session_id

# Monitor progress
process(action="poll", session_id="<id>")
process(action="log", session_id="<id>", limit=50)

# Send input if Codex asks
process(action="submit", session_id="<id>", data="yes")

# Kill if needed
process(action="kill", session_id="<id>")
```

## Session Management

```bash
# Resume previous session (interactive)
codex resume

# Resume in exec mode
codex exec resume --last
codex exec resume <SESSION_ID>
codex exec resume --all           # include sessions from any directory

# Fork current session
codex fork
```

## Code Review

```bash
# Review specific file in TUI
/codex review
codex review src/auth.py

# Safe review (clone to temp dir)
terminal(command="REVIEW=$(mktemp -d) && git clone https://github.com/user/repo.git $REVIEW && cd $REVIEW && gh pr checkout 42 && codex review --base origin/main", pty=true)
```

## Cloud Tasks

```bash
# Interactive cloud picker
codex cloud

# Submit task directly
codex cloud exec --env ENV_ID "Summarize open bugs"

# Best-of-N (1-4 attempts)
codex cloud exec --env ENV_ID --attempts 3 "Summarize open bugs"
```

## MCP Integration

```bash
# Add MCP server (stdio transport)
codex mcp add <name> -- <command>

# Add via URL (HTTP)
codex mcp add --url https://example.com/mcp <name>

# List and manage
codex mcp list
codex mcp get <name>
codex mcp remove <name>
codex mcp login/logout <name>
```

## Parallel Issue Fixing with Worktrees

```bash
# Create worktrees
terminal(command="git worktree add -b fix/issue-78 /tmp/issue-78 main", workdir="~/project")
terminal(command="git worktree add -b fix/issue-99 /tmp/issue-99 main", workdir="~/project")

# Launch Codex in each (parallel)
terminal(command="codex exec --yolo 'Fix issue #78: <description>. Commit when done.'", workdir="/tmp/issue-78", background=true, pty=true)
terminal(command="codex exec --yolo 'Fix issue #99: <description>. Commit when done.'", workdir="/tmp/issue-99", background=true, pty=true)

# Monitor
process(action="list")

# Push and create PRs
terminal(command="cd /tmp/issue-78 && git push -u origin fix/issue-78", workdir="/tmp/issue-78")
terminal(command="gh pr create --repo user/repo --head fix/issue-78 --title 'fix: ...' --body '...'", workdir="~/project")

# Cleanup
terminal(command="git worktree remove /tmp/issue-78", workdir="~/project")
```

## Remote App Server

```bash
# On remote host: start app server
codex app-server --listen ws://127.0.0.1:4500

# With auth token
codex app-server --listen wss://0.0.0.0:4500 --remote-auth-token-env CODEX_TOKEN

# Local: connect TUI to remote
codex --remote ws://127.0.0.1:4500
```

## exec Output Format (JSON)

```bash
codex exec "build a feature" --json
```

Outputs newline-delimited JSON events:
- `type: "done"` — task completed
- `type: "error"` — error occurred
- `type: "message"` — informational message

## Configuration

`~/.codex/config.toml`:
```toml
model_provider = "custom"
model = "gpt-5.4"
model_reasoning_effort = "high"

[model_providers.custom]
name = "custom"
wire_api = "responses"
requires_openai_auth = true
base_url = "http://127.0.0.1:8317/v1"
OPENAI_API_KEY = "sk-cpa-..."

# Auth storage
cli_auth_credentials_store = "keyring"  # or "file" or "auto"
```

**Note:** Codex only supports `wire_api = "responses"`, not `chat_completions`.

## Image Inputs

```bash
codex -i screenshot.png "Explain this error"
codex --image img1.png,img2.jpg "Summarize these diagrams"
```

Supports PNG, JPEG; comma-separate or repeat flag for multiple images.

## Feature Flags

```bash
codex features list
codex features enable unified_exec
codex features disable shell_snapshot
```

## Shell Completions

```bash
codex completion bash >> ~/.bashrc
codex completion zsh >> ~/.zshrc
```

## Pitfalls

1. **Omitting `--full-auto`** → sandbox falls back to read-only, cannot write files. Always include `--full-auto` for building tasks.
2. **Default reasoning effort (`none`)** → Codex won't self-correct on errors. Set `model_reasoning_effort = "high"` in config.toml.
3. **Sandbox not writable for some paths** → `~/Desktop`, home subdirs outside workdir get `Operation not permitted`. Always write to `/tmp/` then manually move final artifacts.
4. **`--reasoning-effort` does not work with `review` subcommand** — omit it for review tasks.
5. **No `--max-turns` equivalent** — Codex exec runs until completion. For long tasks, use `background=true` and monitor with `process`.

## Hermes Agent Rules

1. **Always use `pty=true`** — Codex is an interactive TUI; hangs without PTY
2. **Git repo required** — use `mktemp -d && git init` for scratch work
3. **Use `exec` for one-shots** — `codex exec "prompt"` runs and exits cleanly
4. **`--full-auto` for building** — enables workspace-write sandbox automatically
5. **Background for long tasks** — use `background=true` and monitor with `process`
6. **Parallel is fine** — run multiple Codex processes for batch work
7. **Clean up background sessions** — use `process(action="kill")` when done
