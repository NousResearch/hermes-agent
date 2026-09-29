---
name: gemini-cli
description: Delegate coding tasks to Google Gemini CLI agent. Use for whole-repo audits, code reviews, refactors, batch issue fixing, and structured-output extraction. Requires the gemini CLI.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [Coding-Agent, Gemini, Google, Code-Review, Refactoring, Audit]
    related_skills: [claude-code, codex, opencode, hermes-agent]
---

# Gemini CLI

Delegate coding tasks to [Gemini CLI](https://github.com/google-gemini/gemini-cli) via the Hermes terminal. Gemini CLI is Google's autonomous coding agent with a 1M-token context window — well-suited for whole-repo reads, multi-file refactors, and read-only audits.

## Prerequisites

- Gemini CLI installed: `npm install -g @google/gemini-cli` (or `brew install gemini-cli`)
- Auth: run `gemini` once to sign in with Google (free tier: 60 req/min, 1k req/day), or set `GEMINI_API_KEY` for API-key auth
- **No git repo required** (unlike Codex) — Gemini runs in any directory
- **No PTY required** for `-p`/non-interactive mode — use plain `terminal()` calls

## One-Shot Tasks

```
terminal(command="gemini -p 'Add dark mode toggle to settings' --approval-mode yolo", workdir="~/project", timeout=180)
```

For scratch work (no git init needed):
```
terminal(command="cd $(mktemp -d) && gemini -p 'Build a snake game in Python' --approval-mode yolo", timeout=180)
```

## Approval Modes

| Mode | Effect | Use case |
|------|--------|----------|
| `--approval-mode default` | Prompts for every action | Interactive only — avoid in scripts |
| `--approval-mode auto_edit` | Auto-approves edits, prompts for shell | Bounded write tasks |
| `--approval-mode yolo` (or `-y`) | Auto-approves everything | One-shot builds, batch fixes |
| `--approval-mode plan` | Read-only — refuses to write or run shell | Audits, code reviews, summaries |

**`plan` mode is the killer feature** — safer than read-only flags on other CLIs because Gemini's planner explicitly knows it cannot mutate state. Use it for any task where you only need an answer.

## Read-Only Audits (1M Context)

Gemini's 1M-token window makes whole-repo questions practical:

```
terminal(command="gemini -p 'List every file that touches the auth flow and explain the request lifecycle' --approval-mode plan", workdir="~/project", timeout=240)
```

Multi-directory context:
```
terminal(command="gemini -p 'Compare the two auth implementations' --include-directories ../legacy-auth,../new-auth --approval-mode plan", workdir="~/project", timeout=240)
```

## Structured Output (JSON / stream-json)

Parseable result with token + cost stats:
```
terminal(command="gemini -p 'Find all TODOs in src/' --approval-mode plan -o json", workdir="~/project", timeout=180)
```

Returns a JSON envelope with `stats.tokens`, `tools.totalCalls`, `files.totalLinesAdded/Removed`, etc.

Streaming events for live monitoring:
```
terminal(command="gemini -p 'Refactor the user model' --approval-mode yolo -o stream-json", workdir="~/project", timeout=300)
```

Newline-delimited events: `init` → `message` (with `delta:true` chunks) → `result` (with stats). Filter with `jq`:
```
gemini -p "..." -o stream-json | jq -rj 'select(.role=="assistant" and .delta) | .content'
```

## Background Mode (Long Tasks)

```
# Start in background — no PTY needed for -p mode
terminal(command="gemini -p 'Refactor the entire auth module' --approval-mode yolo", workdir="~/project", background=true)
# Returns session_id

# Monitor progress
process(action="poll", session_id="<id>")
process(action="log", session_id="<id>")

# Kill if needed
process(action="kill", session_id="<id>")
```

For interactive (multi-turn) sessions, use `pty=true` and send input via `process(action="submit", ...)`.

## Parallel Issue Fixing with Worktrees

Note: Gemini's built-in `-w/--worktree` flag is interactive-only — for scripted parallel work, use manual `git worktree add`:

```
# Create worktrees
terminal(command="git worktree add -b fix/issue-78 /tmp/issue-78 main", workdir="~/project")
terminal(command="git worktree add -b fix/issue-99 /tmp/issue-99 main", workdir="~/project")

# Launch Gemini in each
terminal(command="gemini -p 'Fix issue #78: <description>. Commit when done.' --approval-mode yolo", workdir="/tmp/issue-78", background=true)
terminal(command="gemini -p 'Fix issue #99: <description>. Commit when done.' --approval-mode yolo", workdir="/tmp/issue-99", background=true)

# Monitor
process(action="list")

# After completion, push and create PRs
terminal(command="cd /tmp/issue-78 && git push -u origin fix/issue-78")
terminal(command="gh pr create --repo user/repo --head fix/issue-78 --title 'fix: ...' --body '...'")

# Cleanup
terminal(command="git worktree remove /tmp/issue-78", workdir="~/project")
```

## PR Reviews

Clone to a temp directory for safe review (plan mode = no risk of mutation):

```
terminal(command="REVIEW=$(mktemp -d) && git clone https://github.com/user/repo.git $REVIEW && cd $REVIEW && gh pr checkout 42 && gemini -p 'Review the diff against origin/main. Flag bugs, security issues, missing tests.' --approval-mode plan", timeout=300)
```

## Batch PR Reviews in Parallel

```
# Fetch all PR refs
terminal(command="git fetch origin '+refs/pull/*/head:refs/remotes/origin/pr/*'", workdir="~/project")

# Review multiple PRs in parallel
terminal(command="gemini -p 'Review PR #86. git diff origin/main...origin/pr/86' --approval-mode plan", workdir="~/project", background=true)
terminal(command="gemini -p 'Review PR #87. git diff origin/main...origin/pr/87' --approval-mode plan", workdir="~/project", background=true)

# Post results
terminal(command="gh pr comment 86 --body '<review>'", workdir="~/project")
```

## Model Selection

| Model | When to use |
|-------|-------------|
| `gemini-2.5-flash` (default for free tier) | Default — fast, cheap, handles most tasks |
| `gemini-2.5-pro` | Deep reasoning, large refactors, tricky reviews |

```
terminal(command="gemini -p '...' -m gemini-2.5-pro --approval-mode yolo", workdir="~/project")
```

## Subcommands

| Command | Purpose |
|---------|---------|
| `gemini mcp list` | List configured MCP servers |
| `gemini mcp add <name> -- <cmd>` | Add an MCP server |
| `gemini extensions list` | List installed extensions |
| `gemini skills list` | List available skills |
| `gemini hooks list` | List configured hooks |
| `gemini --list-sessions` | List resumable sessions |
| `gemini -r latest -p "continue"` | Resume the most recent session |

## Rules

1. **Use `-p` for one-shots** — `gemini -p "prompt"` runs and exits cleanly, no PTY needed
2. **Always set an approval mode in scripts** — default mode prompts and will hang. Use `yolo` for builds, `plan` for reads, `auto_edit` for bounded writes
3. **Prefer `plan` mode for read-only work** — it's a stronger guarantee than allowlists; the agent itself refuses to mutate
4. **Use 1M context for whole-repo questions** — Gemini is the right tool when you need cross-file understanding in one pass
5. **Use `-o json` or `-o stream-json` when piping into other tools** — text mode is for humans
6. **Background for long tasks** — use `background=true` and monitor with `process` tool. No PTY needed for `-p` mode
7. **Don't use `-w/--worktree` in scripted mode** — it's interactive-only. Use `git worktree add` instead
8. **Parallel is fine** — run multiple Gemini processes for batch work; mind the free-tier 60 req/min ceiling
