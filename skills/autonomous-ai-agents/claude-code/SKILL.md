---
name: claude-code
description: Delegate coding tasks to Claude Code (Anthropic's CLI agent). Use for building features, refactoring, PR reviews, and iterative coding. Requires the claude CLI installed.
version: 3.0.0
author: Hermes Agent + Nous Research + Teknium
license: MIT
metadata:
  hermes:
    tags: [Coding-Agent, Claude, Anthropic, Code-Review, Refactoring, PTY, Automation]
    related_skills: [codex, hermes-agent, opencode]
---

# Claude Code — Anthropic 官方编程 Agent

Delegate coding tasks to [Claude Code](https://docs.anthropic.com/en/docs/claude-code) via the Hermes terminal. Claude Code v2.x reads files, writes code, runs shell commands, spawns subagents, and manages git workflows autonomously.

## 官方资源

- Docs: https://docs.anthropic.com/en/docs/claude-code
- CLI Reference: https://docs.anthropic.com/en/docs/claude-code/cli-reference
- Memory: https://docs.anthropic.com/en/docs/claude-code/memory
- Permission Modes: https://docs.anthropic.com/en/docs/claude-code/permission-modes
- GitHub: https://github.com/anthropic/claude-code

## Prerequisites

- **Install:** `npm install -g @anthropic-ai/claude-code` or `curl -fsSL https://claude.ai/install.sh | bash`
- **Auth:** `claude` once (browser OAuth) or `claude auth login --console` (API key)
- **SSO:** `claude auth login --sso` for Enterprise
- **Check status:** `claude auth status --text`
- **Health check:** `claude doctor`
- **Version:** `claude --version` (requires v2.x+)
- **Update:** `claude update` or `claude upgrade`

## Two Orchestration Modes

### Mode 1: Print Mode `-p` — Non-Interactive (PREFERRED for most tasks)

Print mode runs a one-shot task, returns the result, and exits. No PTY needed. No interactive prompts.

```bash
terminal(command="claude -p 'Add error handling to all API calls' --allowedTools 'Read,Edit' --max-turns 10", workdir="/path/to/project", timeout=120)
```

**Use print mode for:**
- One-shot coding tasks (fix a bug, add a feature, refactor)
- CI/CD automation and scripting
- Structured data extraction with `--json-schema`
- Piped input processing (`cat file | claude -p "analyze this"`)
- Any task where you don't need multi-turn conversation

**Print mode skips ALL interactive dialogs** — no workspace trust prompt, no permission confirmations.

### Mode 2: Interactive PTY via tmux — Multi-Turn Sessions

Full conversational REPL with follow-up prompts and slash commands. **Requires tmux orchestration.**

```bash
# Start a tmux session
terminal(command="tmux new-session -d -s claude-work -x 140 -y 40")

# Launch Claude Code inside it
terminal(command="tmux send-keys -t claude-work 'cd /path/to/project && claude' Enter")

# Wait for startup, then send your task (~3-5 seconds)
terminal(command="sleep 5 && tmux send-keys -t claude-work 'Refactor the auth module to use JWT tokens' Enter")

# Monitor progress
terminal(command="sleep 15 && tmux capture-pane -t claude-work -p -S -50")

# Send follow-up tasks
terminal(command="tmux send-keys -t claude-work 'Now add unit tests' Enter")

# Exit when done
terminal(command="tmux send-keys -t claude-work '/exit' Enter")
```

**Use interactive mode for:**
- Multi-turn iterative work (refactor → review → fix → test cycle)
- Tasks requiring human-in-the-loop decisions
- Exploratory coding sessions
- When you need Claude's slash commands (`/compact`, `/review`, `/model`)

## PTY Dialog Handling (CRITICAL for Interactive Mode)

Claude Code presents up to two confirmation dialogs on first launch. You MUST handle these via tmux send-keys:

### Dialog 1: Workspace Trust (first visit to a directory)
```
❯ 1. Yes, I trust this folder    ← DEFAULT (just press Enter)
  2. No, exit
```
**Handling:** `tmux send-keys -t <session> Enter`

### Dialog 2: Bypass Permissions Warning (only with --dangerously-skip-permissions)
```
❯ 1. No, exit                    ← DEFAULT (WRONG choice!)
  2. Yes, I accept
```
**Handling:** Must navigate DOWN first, then Enter:
```bash
tmux send-keys -t <session> Down && sleep 0.3 && tmux send-keys -t <session> Enter
```

### Robust Dialog Handling Pattern
```bash
# Launch with permissions bypass
terminal(command="tmux send-keys -t claude-work 'claude --dangerously-skip-permissions \"task\"' Enter")

# Handle trust dialog (Enter for default "Yes")
terminal(command="sleep 4 && tmux send-keys -t claude-work Enter")

# Handle permissions dialog (Down then Enter for "Yes, I accept")
terminal(command="sleep 3 && tmux send-keys -t claude-work Down && sleep 0.3 && tmux send-keys -t claude-work Enter")

# Wait for Claude to work
terminal(command="sleep 15 && tmux capture-pane -t claude-work -p -S -60")
```

**Note:** After the first trust acceptance for a directory, the trust dialog won't appear again. Only the permissions dialog recurs each time you use `--dangerously-skip-permissions`.

## CLI Subcommands

| Subcommand | Purpose |
|------------|---------|
| `claude` | Start interactive REPL |
| `claude "query"` | Start REPL with initial prompt |
| `claude -p "query"` | Print mode (non-interactive, exits when done) |
| `cat file \| claude -p "query"` | Pipe content as stdin context |
| `claude -c` | Continue the most recent conversation in this directory |
| `claude -r "id"` | Resume a specific session by ID or name |
| `claude auth login` | Sign in (`--console` for API billing, `--sso` for Enterprise) |
| `claude auth status` | Check login status (`--text` for human-readable) |
| `claude mcp add <name> -- <cmd>` | Add an MCP server |
| `claude mcp list` | List configured MCP servers |
| `claude mcp remove <name>` | Remove an MCP server |
| `claude agents` | List configured agents |
| `claude doctor` | Run health checks |
| `claude update / upgrade` | Update Claude Code to latest version |
| `claude remote-control` | Start Remote Control server |
| `claude install [target]` | Install native build (stable, latest, or specific version) |
| `claude plugin / plugins` | Manage plugins |
| `claude auto-mode` | Inspect auto mode classifier configuration |

## Print Mode Deep Dive

### Structured JSON Output
```bash
terminal(command="claude -p 'Analyze auth.py for security issues' --output-format json --max-turns 5", workdir="/project", timeout=120)
```

Returns JSON with `session_id`, `num_turns`, `total_cost_usd`, `subtype` (`success`, `error_max_turns`, `error_budget`).

### Streaming JSON
```bash
terminal(command="claude -p 'task' --output-format stream-json --verbose --include-partial-messages", timeout=60)
```

Filter for live text:
```bash
claude -p "task" --output-format stream-json --verbose --include-partial-messages | \
  jq -rj 'select(.type == "stream_event" and .event.delta.type? == "text_delta") | .event.delta.text'
```

### Piped Input
```bash
cat src/auth.py | claude -p 'Review for bugs' --max-turns 1
git diff HEAD~3 | claude -p 'Summarize changes' --max-turns 1
```

### JSON Schema
```bash
claude -p 'List all functions' --output-format json \
  --json-schema '{"type":"object","properties":{"functions":{"type":"array","items":{"type":"string"}}}, "required":["functions"]}' \
  --max-turns 5
```

### Session Continuation
```bash
# Start task, save session ID
claude -p 'Start refactoring' --output-format json --max-turns 10 > /tmp/session.json

# Resume with session ID
claude -p 'Continue with pooling' --resume $(cat /tmp/session.json | python3 -c 'import json,sys; print(json.load(sys.stdin)["session_id"])') --max-turns 5

# Or resume most recent session
claude -p 'What did you do?' --continue --max-turns 1
```

### Bare Mode (Fastest CI mode)
```bash
claude --bare -p 'Run all tests' --allowedTools 'Read,Bash' --max-turns 10
```
`--bare` skips hooks, plugins, MCP discovery, CLAUDE.md loading. Requires `ANTHROPIC_API_KEY`.

### Fallback Model
```bash
claude -p 'task' --fallback-model haiku --max-turns 5
```
Auto-falls back when default model is overloaded.

## Complete CLI Flags Reference

### Session & Environment
| Flag | Effect |
|------|--------|
| `-p, --print` | Non-interactive one-shot mode |
| `-c, --continue` | Resume most recent conversation |
| `-r, --resume <id>` | Resume specific session |
| `--fork-session` | New session ID when resuming |
| `--no-session-persistence` | Don't save to disk (print mode) |
| `-w, --worktree [name]` | Isolated git worktree at `.claude/worktrees/<name>` |
| `--tmux` | Create tmux session for worktree |
| `--add-dir <paths...>` | Grant access to additional directories |
| `--from-pr [number]` | Resume session linked to GitHub PR |

### Model & Performance
| Flag | Effect |
|------|--------|
| `--model <alias>` | `sonnet`, `opus`, `haiku`, or full name |
| `--effort <level>` | `low`, `medium`, `high`, `max`, `auto` |
| `--max-turns <n>` | Limit agentic loops (print mode only) |
| `--max-budget-usd <n>` | Cap API spend (print mode only, min ~$0.05) |
| `--fallback-model <model>` | Auto-fallback when overloaded (print mode only) |

### Permission & Safety
| Flag | Effect |
|------|--------|
| `--dangerously-skip-permissions` | Skip all permission prompts |
| `--permission-mode <mode>` | `default`, `acceptEdits`, `plan`, `auto`, `dontAsk`, `bypassPermissions` |
| `--allowedTools <tools...>` | Whitelist tools (comma-separated) |
| `--disallowedTools <tools...>` | Blacklist tools |

### Output & Input Format
| Flag | Effect |
|------|--------|
| `--output-format <fmt>` | `text` (default), `json`, `stream-json` |
| `--json-schema <schema>` | Force structured JSON output |
| `--verbose` | Full turn-by-turn output |
| `--include-partial-messages` | Include partial streaming events |

### System Prompt & Context
| Flag | Effect |
|------|--------|
| `--append-system-prompt <text>` | Add to default system prompt |
| `--append-system-prompt-file <path>` | Add file contents to system prompt |
| `--system-prompt <text>` | Replace entire system prompt |
| `--bare` | Skip hooks, plugins, MCP, CLAUDE.md (fastest) |
| `--mcp-config <path>` | Load MCP servers from JSON |
| `--settings <file-or-json>` | Load additional settings |

### Tool Name Syntax for --allowedTools / --disallowedTools
```
Read                    # All file reading
Edit                    # File editing
Write                   # File creation
Bash                    # All shell commands
Bash(git *)             # Only git commands
Bash(npm run lint:*)    # Pattern matching
WebSearch               # Web search
WebFetch                # Web page fetching
mcp__<server>__<tool>   # Specific MCP tool
```

## Permission Modes

| Mode | Behavior |
|------|----------|
| `default` | Reads only |
| `acceptEdits` | Auto-approves file edits and common shell commands |
| `plan` | Read-only, `/plan` requests execution plan |
| `auto` | Runs everything with background safety classifier |
| `bypassPermissions` | Skip all checks (isolated containers only) |

## PR Review Pattern

### Quick Review (Print Mode)
```bash
git diff main...feature-branch | claude -p 'Review this diff for bugs and security issues' --max-turns 1
```

### Deep Review (Interactive + Worktree)
```bash
tmux new-session -d -s review -x 140 -y 40
tmux send-keys -t review 'cd /path && claude -w pr-review' Enter
sleep 5 && tmux send-keys -t review Enter        # trust dialog
sleep 2 && tmux send-keys -t review 'Review all changes vs main' Enter
sleep 30 && tmux capture-pane -t review -p -S -60
```

### From PR Number
```bash
claude -p 'Review this PR' --from-pr 42 --max-turns 10
```

## Parallel Tasks

```bash
terminal(command="tmux new-session -d -s task1 -x 140 -y 40 && tmux send-keys -t task1 'cd ~/project && claude -p \"Fix the auth bug\" --allowedTools Read,Edit --max-turns 10' Enter")
terminal(command="tmux new-session -d -s task2 -x 140 -y 40 && tmux send-keys -t task2 'cd ~/project && claude -p \"Write integration tests\" --allowedTools Read,Write,Bash --max-turns 15' Enter")
terminal(command="sleep 30 && for s in task1 task2; do echo \"=== $s ===\"; tmux capture-pane -t $s -p -S -5 2>/dev/null; done")
```

## CLAUDE.md — Project Context

Claude Code auto-loads `CLAUDE.md` from project root. Use it to persist context:

```markdown
# Project: My API

## Architecture
- FastAPI + SQLAlchemy + PostgreSQL + Redis

## Key Commands
- `make test` — full test suite
- `make lint` — ruff + mypy

## Code Standards
- Type hints on all public functions
- 2-space indentation for YAML, 4-space for Python
```

### Modular Rules (`.claude/rules/`)
```
.claude/
├── CLAUDE.md           # Main instructions
└── rules/
    ├── code-style.md   # Code style
    ├── testing.md       # Testing conventions
    └── security.md     # Security requirements
```

### Auto-Memory
Claude stores learned project context in `~/.claude/projects/<project>/memory/`. First 200 lines or 25KB loads at session start. Configurable via `"autoMemoryEnabled": false` in settings.

## Hooks

```json
{
  "hooks": {
    "PostToolUse": [{
      "matcher": "Write(*.py)",
      "hooks": [{"type": "command", "command": "ruff check --fix $CLAUDE_FILE_PATHS"}]
    }],
    "PreToolUse": [{
      "matcher": "Bash",
      "hooks": [{"type": "command", "command": "if echo \"$CLAUDE_TOOL_INPUT\" | grep -q 'rm -rf'; then echo 'Blocked!' && exit 2; fi"}]
    }]
  }
}
```

**8 Hook types:** `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `Notification`, `Stop`, `SubagentStop`, `PreCompact`, `SessionStart`

## MCP Integration

```bash
claude mcp add -s user github -- npx @modelcontextprotocol/server-github
claude mcp add -s local postgres -- npx @anthropic-ai/server-postgres --connection-string postgresql://localhost/mydb
```

**Scopes:** `-s user` (global), `-s local` (project, gitignored), `-s project` (project, git-tracked).

## Monitoring Interactive Sessions

```bash
tmux capture-pane -t dev -p -S -10
```

Status indicators:
- `❯` = waiting for input (done or asking question)
- `●` lines = actively using tools
- `⏵⏵ bypass permissions on` = permissions mode indicator

Context window health (`/context`):
- **< 70%** — Normal
- **70-85%** — Consider `/compact`
- **> 85%** — Use `/compact` or `/clear`

## Environment Variables

| Variable | Effect |
|----------|--------|
| `ANTHROPIC_API_KEY` | API authentication |
| `CLAUDE_CODE_EFFORT_LEVEL` | Default effort: `low`/`medium`/`high`/`max`/`auto` |
| `MAX_THINKING_TOKENS` | Cap thinking tokens (0 = disable) |
| `MAX_MCP_OUTPUT_TOKENS` | Cap MCP server output |
| `CLAUDE_CODE_NO_FLICKER=1` | Enable alt-screen rendering |

## Pitfalls

1. **Interactive mode REQUIRES tmux** — Claude Code is a full TUI app. `pty=true` alone works but tmux gives `capture-pane` monitoring and `send-keys` input.
2. **`--dangerously-skip-permissions` dialog defaults to "No"** — must send Down then Enter. Print mode (`-p`) skips this entirely.
3. **`--max-budget-usd` minimum ~$0.05** — system prompt cache creation costs this much.
4. **`--max-turns` is print-mode only** — ignored in interactive sessions.
5. **Trust dialog only appears once per directory** — then cached.
6. **Background tmux sessions persist** — always `tmux kill-session -t <name>` when done.
7. **Context degradation is real** — quality drops above 70% context usage. Monitor with `/context` and proactively `/compact`.
8. **Slash commands only work in interactive mode** — use natural language in `-p` mode instead.

## Hermes Agent Rules

1. **Prefer print mode (`-p`) for single tasks** — cleaner, no dialog handling, structured output
2. **Use tmux for multi-turn interactive work** — the only reliable TUI orchestration
3. **Always set `workdir`** — keep Claude focused on the right project directory
4. **Set `--max-turns` in print mode** — prevents infinite loops and runaway costs
5. **Monitor tmux sessions** — use `tmux capture-pane` to check progress
6. **Look for the `❯` prompt** — indicates Claude waiting for input
7. **Clean up tmux sessions** — kill them when done to avoid resource leaks
8. **Report results to user** — summarize what Claude did and what changed
9. **Use `--allowedTools`** — restrict to only what's needed
10. **Don't kill slow sessions** — check progress instead; Claude may be doing multi-step work
