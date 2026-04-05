---
name: claude-code
description: Delegate coding tasks to Claude Code (Anthropic's CLI agent). Use for building features, refactoring, PR reviews, and iterative coding. Requires the claude CLI installed.
version: 2.0.0
author: Hermes Agent + Teknium
license: MIT
metadata:
  hermes:
    tags: [Coding-Agent, Claude, Anthropic, Code-Review, Refactoring, PTY, Automation]
    related_skills: [codex, hermes-agent, opencode]
---

# Claude Code — Hermes Orchestration Guide

Delegate coding tasks to [Claude Code](https://code.claude.com/docs/en/cli-reference) (Anthropic's autonomous coding agent CLI) via the Hermes terminal. Claude Code v2.x can read files, write code, run shell commands, spawn subagents, and manage git workflows autonomously.

## Prerequisites

- **Install:** `npm install -g @anthropic-ai/claude-code`
- **Auth:** run `claude` once to log in (browser OAuth for Pro/Max, or set `ANTHROPIC_API_KEY`)
- **Version check:** `claude --version` (requires v2.x+)

## Two Orchestration Modes

Hermes interacts with Claude Code in two fundamentally different ways. Choose based on the task.

### Mode 1: Print Mode (`-p`) — Non-Interactive (PREFERRED for most tasks)

Print mode runs a one-shot task, returns the result, and exits. No PTY needed. No interactive prompts. This is the cleanest integration path.

```
terminal(command="claude -p 'Add error handling to all API calls in src/' --allowedTools 'Read,Edit' --max-turns 10", workdir="/path/to/project", timeout=120)
```

**When to use print mode:**
- One-shot coding tasks (fix a bug, add a feature, refactor)
- CI/CD automation and scripting
- Structured data extraction with `--json-schema`
- Piped input processing (`cat file | claude -p "analyze this"`)
- Any task where you don't need multi-turn conversation

**Print mode skips ALL interactive dialogs** — no workspace trust prompt, no permission confirmations. This makes it ideal for automation.

### Mode 2: Interactive PTY via tmux — Multi-Turn Sessions

Interactive mode gives you a full conversational REPL where you can send follow-up prompts, use slash commands, and watch Claude work in real time. **Requires tmux orchestration.**

```
# Start a tmux session
terminal(command="tmux new-session -d -s claude-work -x 140 -y 40")

# Launch Claude Code inside it
terminal(command="tmux send-keys -t claude-work 'cd /path/to/project && claude' Enter")

# Wait for startup, then send your task
# (after ~3-5 seconds for the welcome screen)
terminal(command="sleep 5 && tmux send-keys -t claude-work 'Refactor the auth module to use JWT tokens' Enter")

# Monitor progress by capturing the pane
terminal(command="sleep 15 && tmux capture-pane -t claude-work -p -S -50")

# Send follow-up tasks
terminal(command="tmux send-keys -t claude-work 'Now add unit tests for the new JWT code' Enter")

# Exit when done
terminal(command="tmux send-keys -t claude-work '/exit' Enter")
```

**When to use interactive mode:**
- Multi-turn iterative work (refactor → review → fix → test cycle)
- Tasks requiring human-in-the-loop decisions
- Exploratory coding sessions
- When you need to use Claude's slash commands (`/compact`, `/review`, `/model`)

## PTY Dialog Handling (CRITICAL for Interactive Mode)

Claude Code presents up to two confirmation dialogs on first launch. You MUST handle these via tmux send-keys:

### Dialog 1: Workspace Trust (first visit to a directory)
```
❯ 1. Yes, I trust this folder    ← DEFAULT (just press Enter)
  2. No, exit
```
**Handling:** `tmux send-keys -t <session> Enter` — default selection is correct.

### Dialog 2: Bypass Permissions Warning (only with --dangerously-skip-permissions)
```
❯ 1. No, exit                    ← DEFAULT (WRONG choice!)
  2. Yes, I accept
```
**Handling:** Must navigate DOWN first, then Enter:
```
tmux send-keys -t <session> Down && sleep 0.3 && tmux send-keys -t <session> Enter
```

### Robust Dialog Handling Pattern
```
# Launch with permissions bypass
terminal(command="tmux send-keys -t claude-work 'claude --dangerously-skip-permissions \"your task\"' Enter")

# Handle trust dialog (Enter for default "Yes")
terminal(command="sleep 4 && tmux send-keys -t claude-work Enter")

# Handle permissions dialog (Down then Enter for "Yes, I accept")
terminal(command="sleep 3 && tmux send-keys -t claude-work Down && sleep 0.3 && tmux send-keys -t claude-work Enter")

# Now wait for Claude to work
terminal(command="sleep 15 && tmux capture-pane -t claude-work -p -S -60")
```

**Note:** After the first trust acceptance for a directory, the trust dialog won't appear again. Only the permissions dialog recurs each time you use `--dangerously-skip-permissions`.

## Print Mode Deep Dive

### Structured JSON Output
```
terminal(command="claude -p 'Analyze auth.py for security issues' --output-format json --max-turns 5", workdir="/project", timeout=120)
```

Returns a JSON object with:
```json
{
  "type": "result",
  "subtype": "success",
  "result": "The analysis text...",
  "session_id": "75e2167f-...",
  "num_turns": 3,
  "total_cost_usd": 0.0787,
  "duration_ms": 10276,
  "stop_reason": "end_turn",
  "terminal_reason": "completed",
  "usage": { "input_tokens": 5, "output_tokens": 603, ... }
}
```

Use `session_id` to resume later. `num_turns` shows how many agentic loops it took. `total_cost_usd` tracks spend.

### Piped Input
```
# Pipe a file for analysis
terminal(command="cat src/auth.py | claude -p 'Review this code for bugs' --max-turns 1", timeout=60)

# Pipe multiple files
terminal(command="cat src/*.py | claude -p 'Find all TODO comments' --max-turns 1", timeout=60)

# Pipe command output
terminal(command="git diff HEAD~3 | claude -p 'Summarize these changes' --max-turns 1", timeout=60)
```

### JSON Schema for Structured Extraction
```
terminal(command="claude -p 'List all functions in src/' --output-format json --json-schema '{\"type\":\"object\",\"properties\":{\"functions\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}}},\"required\":[\"functions\"]}' --max-turns 5", workdir="/project", timeout=90)
```

Parse `structured_output` from the JSON result.

### Session Continuation
```
# Start a task
terminal(command="claude -p 'Start refactoring the database layer' --output-format json --max-turns 10 > /tmp/session.json", workdir="/project", timeout=180)

# Resume with session ID
terminal(command="claude -p 'Continue and add connection pooling' --resume $(cat /tmp/session.json | python3 -c 'import json,sys; print(json.load(sys.stdin)[\"session_id\"])') --max-turns 5", workdir="/project", timeout=120)

# Or resume the most recent session in the same directory
terminal(command="claude -p 'What did you do last time?' --continue --max-turns 1", workdir="/project", timeout=30)
```

### Bare Mode for CI/Scripting
```
terminal(command="claude --bare -p 'Run all tests and report failures' --allowedTools 'Read,Bash' --max-turns 10", workdir="/project", timeout=180)
```

`--bare` skips hooks, plugins, MCP discovery, and CLAUDE.md loading. Fastest startup. Requires `ANTHROPIC_API_KEY` (skips OAuth).

## Key Flags Reference

### Essential Flags
| Flag | Effect | Mode |
|------|--------|------|
| `-p, --print` | Non-interactive one-shot mode | Both |
| `-c, --continue` | Resume most recent conversation | Both |
| `-r, --resume <id>` | Resume specific session by ID | Both |
| `--model <alias>` | Model selection: `sonnet`, `opus`, `haiku`, or full name | Both |
| `--effort <level>` | Reasoning depth: `low`, `medium`, `high`, `max` | Both |
| `--max-turns <n>` | Limit agentic loops (prevents runaway) | Print only |
| `--max-budget-usd <n>` | Cap API spend in dollars | Print only |

### Permission & Safety Flags
| Flag | Effect |
|------|--------|
| `--dangerously-skip-permissions` | Auto-approve ALL tool use (file writes, bash, etc.) |
| `--permission-mode <mode>` | `default`, `acceptEdits`, `plan`, `auto`, `dontAsk`, `bypassPermissions` |
| `--allowedTools <tools>` | Whitelist specific tools: `"Read,Edit,Bash"` |
| `--disallowedTools <tools>` | Blacklist specific tools |

### Output & Integration Flags
| Flag | Effect |
|------|--------|
| `--output-format <fmt>` | `text` (default), `json` (structured), `stream-json` (streaming) |
| `--json-schema <schema>` | Force structured JSON output matching a schema |
| `--verbose` | Full turn-by-turn output |
| `--bare` | Skip hooks/plugins/MCP/CLAUDE.md for fast scripting |
| `--append-system-prompt <text>` | Add instructions to the system prompt (preserves built-ins) |
| `--system-prompt <text>` | REPLACE the entire system prompt (use --append instead usually) |
| `--add-dir <path>` | Grant access to additional directories |
| `-w, --worktree <name>` | Run in an isolated git worktree |

### Tool Name Syntax for --allowedTools
- `Read` — file reading
- `Edit` — file editing  
- `Write` — file creation
- `Bash` — shell commands
- `Bash(git *)` — only git commands
- `Bash(npm run lint:*)` — pattern matching
- `WebSearch` — web search capability

## Interactive Session Patterns

### Multi-Turn Development Cycle
```
# 1. Create tmux session
terminal(command="tmux new-session -d -s dev -x 140 -y 40")

# 2. Launch Claude in project
terminal(command="tmux send-keys -t dev 'cd ~/myproject && claude' Enter")
terminal(command="sleep 5")  # Wait for welcome screen

# 3. First task: implement feature
terminal(command="tmux send-keys -t dev 'Implement a caching layer for the API client in src/client.py' Enter")
terminal(command="sleep 30 && tmux capture-pane -t dev -p -S -60")  # Check progress

# 4. Follow-up: add tests
terminal(command="tmux send-keys -t dev 'Now write comprehensive tests for the cache' Enter")
terminal(command="sleep 20 && tmux capture-pane -t dev -p -S -40")

# 5. Follow-up: run tests
terminal(command="tmux send-keys -t dev 'Run the tests and fix any failures' Enter")
terminal(command="sleep 20 && tmux capture-pane -t dev -p -S -40")

# 6. Compact context if running long
terminal(command="tmux send-keys -t dev '/compact focus on the caching implementation' Enter")

# 7. Exit
terminal(command="tmux send-keys -t dev '/exit' Enter")
terminal(command="sleep 2 && tmux kill-session -t dev")
```

### Monitoring Long Operations
```
# Periodic capture to check if Claude is still working or waiting for input
terminal(command="tmux capture-pane -t dev -p -S -10")
```

Look for these indicators:
- `❯` at bottom = waiting for your input (Claude is done or asking a question)
- `●` lines = Claude is actively using tools (reading, writing, running commands)
- `⏵⏵ bypass permissions on` = status bar indicator
- `◐ medium · /effort` = current effort level

### Using Claude's Built-In Slash Commands (Interactive Only)
| Command | Purpose |
|---------|---------|
| `/compact [focus]` | Summarize context to save tokens (add focus topic) |
| `/clear` | Wipe conversation history |
| `/model` | Switch models mid-session |
| `/review` | Request code review of current changes |
| `/init` | Create CLAUDE.md for the project |
| `/memory` | Edit CLAUDE.md directly |
| `/context` | Visualize context window usage |
| `/vim` | Enable vim-style editing |
| `/exit` or Ctrl+D | End session |

### Keyboard Shortcuts (Interactive Only)
| Key | Action |
|-----|--------|
| `Tab` | Toggle Extended Thinking mode |
| `Shift+Tab` | Cycle permission modes |
| `Ctrl+C` | Cancel current generation |
| `Ctrl+R` | Search command history |
| `Esc Esc` | Rewind conversation or code |
| `!` prefix | Execute bash directly (e.g., `!npm test`) |
| `@` prefix | Reference files (e.g., `@./src/api/`) |
| `#` prefix | Quick add to CLAUDE.md memory |

## PR Review Pattern

### Quick Review (Print Mode)
```
terminal(command="cd /path/to/repo && git diff main...feature-branch | claude -p 'Review this diff for bugs, security issues, and style problems. Be thorough.' --max-turns 1", timeout=60)
```

### Deep Review (Interactive + Worktree)
```
terminal(command="tmux new-session -d -s review -x 140 -y 40")
terminal(command="tmux send-keys -t review 'cd /path/to/repo && claude -w pr-review' Enter")
terminal(command="sleep 5 && tmux send-keys -t review Enter")  # Trust dialog
terminal(command="sleep 2 && tmux send-keys -t review 'Review all changes vs main. Check for bugs, security issues, race conditions, and missing tests.' Enter")
terminal(command="sleep 30 && tmux capture-pane -t review -p -S -60")
```

### PR Review from Number
```
terminal(command="claude -p 'Review this PR thoroughly' --from-pr 42 --max-turns 10", workdir="/path/to/repo", timeout=120)
```

## Parallel Claude Instances

Run multiple independent Claude tasks simultaneously:

```
# Task 1: Fix backend
terminal(command="tmux new-session -d -s task1 -x 140 -y 40 && tmux send-keys -t task1 'cd ~/project && claude -p \"Fix the auth bug in src/auth.py\" --allowedTools \"Read,Edit\" --max-turns 10' Enter")

# Task 2: Write tests  
terminal(command="tmux new-session -d -s task2 -x 140 -y 40 && tmux send-keys -t task2 'cd ~/project && claude -p \"Write integration tests for the API endpoints\" --allowedTools \"Read,Write,Bash\" --max-turns 15' Enter")

# Task 3: Update docs
terminal(command="tmux new-session -d -s task3 -x 140 -y 40 && tmux send-keys -t task3 'cd ~/project && claude -p \"Update README.md with the new API endpoints\" --allowedTools \"Read,Edit\" --max-turns 5' Enter")

# Monitor all
terminal(command="sleep 30 && for s in task1 task2 task3; do echo '=== '$s' ==='; tmux capture-pane -t $s -p -S -5 2>/dev/null; done")
```

## CLAUDE.md — Project Context File

Claude Code auto-loads `CLAUDE.md` from the project root. Use it to persist project context:

```markdown
# Project: My API

## Architecture
- FastAPI backend with SQLAlchemy ORM
- PostgreSQL database, Redis cache
- pytest for testing with 90% coverage target

## Key Commands
- `make test` — run full test suite
- `make lint` — ruff + mypy
- `make dev` — start dev server on :8000

## Code Standards
- Type hints on all public functions
- Docstrings in Google style
- 2-space indentation for YAML, 4-space for Python
- No wildcard imports
```

Global context: `~/.claude/CLAUDE.md` (applies to all projects).

## Hooks — Automation on Events

Configure in `.claude/settings.json` or `~/.claude/settings.json`:

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
    }],
    "Stop": [{
      "hooks": [{"type": "command", "command": "echo 'Claude finished a response' >> /tmp/claude-activity.log"}]
    }]
  }
}
```

**Hook types:** UserPromptSubmit, PreToolUse, PostToolUse, Notification, Stop, SubagentStop, PreCompact, SessionStart.

**Environment variables in hooks:** `CLAUDE_PROJECT_DIR`, `CLAUDE_FILE_PATHS`, `CLAUDE_TOOL_INPUT`.

## Custom Subagents

Define specialized agents in `.claude/agents/`:

```markdown
# .claude/agents/security-reviewer.md
---
name: security-reviewer
description: Security-focused code review
model: opus
tools: [Read, Bash]
---
You are a senior security engineer. Review code for:
- Injection vulnerabilities (SQL, XSS, command injection)
- Authentication/authorization flaws
- Secrets in code
- Unsafe deserialization
```

Invoke via: `@security-reviewer review the auth module`

## MCP Integration

Add external tool servers:
```
terminal(command="claude mcp add github -- npx @modelcontextprotocol/server-github", timeout=30)
terminal(command="claude mcp add postgres -- npx @anthropic-ai/server-postgres --connection-string postgresql://localhost/mydb", timeout=30)
```

Scopes: `-s user` (global), `-s local` (project, gitignored), `-s project` (team-shared).

## Custom Slash Commands

Create `.claude/commands/<name>.md` for project shortcuts:

```markdown
# .claude/commands/deploy.md
Run the deploy pipeline:
1. Run all tests
2. Build the Docker image
3. Push to registry
4. Update the staging deployment
Environment: $ARGUMENTS (default: staging)
```

Usage in interactive session: `/deploy production`

Parameterized with `$ARGUMENTS` for dynamic input.

## Cost & Performance Tips

1. **Use `--max-turns`** in print mode to prevent runaway loops. Start with 5-10 for most tasks.
2. **Use `--max-budget-usd`** for cost caps. Note: minimum ~$0.05 for system prompt cache creation.
3. **Use `--effort low`** for simple tasks (faster, cheaper). `high` or `max` for complex reasoning.
4. **Use `--bare`** for CI/scripting to skip plugin/hook discovery overhead.
5. **Use `--allowedTools`** to restrict to only what's needed (e.g., `Read` only for reviews).
6. **Use `/compact`** in interactive sessions when context gets large (precision drops at 70% context usage, hallucinations spike at 85%).
7. **Pipe input** instead of having Claude read files when you just need analysis of known content.
8. **Use `--model haiku`** for simple tasks (cheaper) and `--model opus` for complex multi-step work.

## Pitfalls & Gotchas

1. **Interactive mode REQUIRES tmux** — Claude Code is a full TUI app. Using `pty=true` alone in Hermes terminal works but tmux gives you `capture-pane` for monitoring and `send-keys` for input, which is essential for orchestration.
2. **`--dangerously-skip-permissions` dialog defaults to "No, exit"** — you must send Down then Enter to accept. Print mode (`-p`) skips this entirely.
3. **`--max-budget-usd` minimum is ~$0.05** — system prompt cache creation alone costs this much. Setting lower will error immediately.
4. **`--max-turns` is print-mode only** — ignored in interactive sessions.
5. **Claude may use `python` instead of `python3`** — on systems without a `python` symlink, Claude's bash commands will fail on first try but it self-corrects.
6. **Session resumption requires same directory** — `--continue` finds the most recent session for the current working directory.
7. **`--json-schema` needs enough `--max-turns`** — Claude must read files before producing structured output, which takes multiple turns.
8. **Trust dialog only appears once per directory** — first-time only, then cached.
9. **Background tmux sessions persist** — always clean up with `tmux kill-session -t <name>` when done.

## Rules for Hermes Agents

1. **Prefer print mode (`-p`) for single tasks** — cleaner, no dialog handling, structured output
2. **Use tmux for multi-turn interactive work** — the only reliable way to orchestrate the TUI
3. **Always set `workdir`** — keep Claude focused on the right project directory
4. **Set `--max-turns` in print mode** — prevents infinite loops and runaway costs
5. **Monitor tmux sessions** — use `tmux capture-pane -t <session> -p -S -50` to check progress
6. **Look for the `❯` prompt** — indicates Claude is waiting for input (done or asking a question)
7. **Clean up tmux sessions** — kill them when done to avoid resource leaks
8. **Report results to user** — after completion, summarize what Claude did and what changed
9. **Don't kill slow sessions** — Claude may be doing multi-step work; check progress instead
10. **Use `--allowedTools`** — restrict capabilities to what the task actually needs
