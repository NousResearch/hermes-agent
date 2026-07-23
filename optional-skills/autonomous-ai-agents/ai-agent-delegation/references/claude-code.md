# Claude Code

**Full skill:** `autonomous-ai-agents/claude-code` (archived)

## Quick Reference

```bash
# Install
npm install -g @anthropic-ai/claude-code

# Auth: browser OAuth
claude auth login
# Or API key billing (no browser)
claude auth login --console

# Verify auth
claude auth status --text

# Health check
claude doctor
```

## One-Shot (Print Mode) — Preferred for Automation

```bash
terminal(command="claude -p 'Add error handling to all API calls in src/' --allowedTools 'Read,Edit' --max-turns 10", workdir="/path/to/project", timeout=120)
```

Print mode features:
- `--output-format json` — structured result with `session_id`, `total_cost_usd`, `num_turns`
- `--output-format stream-json --verbose --include-partial-messages` — real-time token streaming
- `--json-schema '<schema>'` — forced structured JSON output
- `--max-budget-usd <n>` — spend cap
- `--fallback-model haiku` — auto-fallback on overload
- Piped input: `cat file | claude -p "analyze"` or `git diff | claude -p "review"`

## Interactive Mode — Via tmux

```bash
# Create tmux session
terminal(command="tmux new-session -d -s claude-work -x 140 -y 40")

# Launch Claude Code
terminal(command="tmux send-keys -t claude-work 'claude --dangerously-skip-permissions \"task\"' Enter")

# Handle workspace trust dialog (default "Yes" — just Enter)
terminal(command="sleep 4 && tmux send-keys -t claude-work Enter")

# Handle permissions dialog (need to go DOWN first)
terminal(command="sleep 3 && tmux send-keys -t claude-work Down && sleep 0.3 && tmux send-keys -t claude-work Enter")

# Monitor pane
terminal(command="sleep 15 && tmux capture-pane -t claude-work -p -S -60")

# Send follow-up task
terminal(command="tmux send-keys -t claude-work 'Add unit tests for the JWT code' Enter")

# Exit
terminal(command="tmux send-keys -t claude-work '/exit' Enter")
```

## Key Flags

| Flag | Purpose |
|------|---------|
| `-p 'task'` | Print mode (non-interactive, exits when done) |
| `-c` / `--continue` | Resume most recent conversation in directory |
| `-r <id>` / `--resume <id>` | Resume specific session by ID or name |
| `--worktree <name>` | Run in isolated git worktree |
| `--tmux` | Create tmux session for worktree |
| `--from-pr <N>` | Resume session linked to GitHub PR |
| `--model <alias>` | Model: `sonnet`, `opus`, `haiku`, or full name |
| `--effort <level>` | Reasoning depth: `low`, `medium`, `high`, `max`, `auto` |
| `--max-turns <n>` | Limit agentic loops (print mode) |
| `--max-budget-usd <n>` | Cap API spend (print mode) |
| `--fallback-model <model>` | Auto-fallback when overloaded (print mode) |
| `--dangerously-skip-permissions` | Auto-approve ALL tool use |
| `--allowedTools <tools>` | Whitelist specific tools |
| `--disallowedTools <tools>` | Blacklist specific tools |
| `--output-format json\|stream-json` | Output format |
| `--json-schema '<schema>'` | Force structured JSON output |
| `--bare` | Skip hooks, plugins, MCP discovery, CLAUDE.md (fastest startup) |
| `--append-system-prompt-file <path>` | Add file to system prompt |
| `--mcp-config <path>` | Load MCP servers from JSON |
| `-d, --debug [filter]` | Debug logging |

### Tool Name Syntax for `--allowedTools`

```
Read                    # All file reading
Edit                    # File editing
Write                   # File creation
Bash                    # All shell commands
Bash(git *)             # Only git commands
Bash(git commit *)      # Only specific patterns
WebSearch               # Web search
WebFetch                # Web page fetching
mcp__<server>__<tool>   # Specific MCP tool
```

## PR Review

```bash
# Quick review — pipe diff
git diff main...feature | claude -p 'Review for bugs, security, style' --max-turns 1

# Deep review — interactive worktree
tmux new-session -d -s review -x 140 -y 40
tmux send-keys -t review 'cd /path/repo && claude -w pr-review' Enter
sleep 5 && tmux send-keys -t review Enter  # Trust dialog
sleep 2 && tmux send-keys -t review 'Review all changes vs main' Enter
sleep 30 && tmux capture-pane -t review -p -S -60

# From PR number
claude -p 'Review this PR' --from-pr 42 --max-turns 10
```

## Session Continuation

```bash
# Start task
claude -p 'Start refactoring the database layer' --output-format json --max-turns 10 > /tmp/session.json

# Resume with session ID
claude -p 'Continue and add connection pooling' --resume $(cat /tmp/session.json | python3 -c 'import json,sys; print(json.load(sys.stdin)["session_id"])') --max-turns 5

# Continue most recent session
claude -p 'What did you do last time?' --continue --max-turns 1

# Fork session (new ID, keeps history)
claude -p 'Try a different approach' --resume <id> --fork-session --max-turns 10
```

## Settings & Memory

Settings hierarchy (highest → lowest): CLI flags → `.claude/settings.local.json` → `.claude/settings.json` → `~/.claude/settings.json`

Memory files: `~/.claude/CLAUDE.md` (global) → `./CLAUDE.md` (project) → `.claude/CLAUDE.local.md` (local overrides)

7. **For performance: check hook overhead first** — before blaming model latency, benchmark the hook chain. Hook overhead compounds on every operation and is the primary source of Claude Code's perceived slowness vs Hermes.
8. **Use `--bare` flag** when speed is critical — skips all hooks, plugins, MCP, and CLAUDE.md for fastest startup.

## Performance Benchmark (OpenCode Zen + MiniMax M2.5)

Both Claude Code and Hermes route through `opencode.ai/zen/v1` to `minimax-m2.5-free` via the same backend. Benchmark data from the user's actual system:

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| Per-task hook overhead | 0.591s | 0.247s |
| Session startup overhead | 0.895s | 0.431s |
| Claude Code vs Hermes slowdown | 1.19–1.30x | reduced |

**Hook timing per operation (measured via `cd /mnt/c/Users/<user> && node ...` from WSL):**

| Hook | Avg Time | Purpose |
|------|---------|---------|
| `session-restore` | ~101ms | Restore session state + init intelligence graph |
| `session-end` | ~82ms | Persist session + consolidate intelligence |
| `pre-bash` | ~74ms | Command safety check |
| `post-bash` | ~85ms | Record metric |
| `post-edit` | ~88ms | Record edit for learning |
| `auto-memory import` | ~108ms | Import auto memory into bridge (REMOVED — ran every session) |
| `auto-memory sync` | ~53ms | Sync insights back to MEMORY.md (REMOVED) |
| `route (UserPromptSubmit)` | ~170ms | Intelligence context injection (REMOVED — ran every message) |

**API latency:** ~1.5–2.8s avg on OpenCode Zen → MiniMax M2.5. Model latency dominates — hook overhead is secondary.

## Hook Architecture — settings.json

Claude Code hooks are defined in `%USERPROFILE%\.claude\settings.json` (Windows) or `~/.claude/settings.json` (Mac/Linux). For the user, the actual path is `/mnt/c/Users/<username>/.claude/settings.json` from WSL.

**Full hook event reference:**

| Hook Event | Fires When |
|---|---|
| `SessionStart` | Claude Code launches |
| `SessionEnd` | Session closes |
| `Stop` | Session stops |
| `UserPromptSubmit` | Before every prompt sent to model |
| `PreToolUse` | Before any tool call |
| `PostToolUse` | After any tool call |
| `PreCompact` | Before context compaction |
| `SubagentStart` | Subagent starts |
| `SubagentStop` | Subagent completes |
| `Notification` | Notifications |

**Optimized settings.json for this user** — keep only essential hooks:

```json
{
  "hooks": {
    "SessionStart": [{
      "command": "cmd /c node %CLAUDE_PROJECT_DIR%/.claude/helpers/hook-handler.cjs session-restore",
      "timeout": 8000
    }],
    "SessionEnd": [{
      "command": "cmd /c node %CLAUDE_PROJECT_DIR%/.claude/helpers/hook-handler.cjs session-end",
      "timeout": 8000
    }],
    "PreToolUse": [{
      "matcher": "Bash",
      "command": "cmd /c node %CLAUDE_PROJECT_DIR%/.claude/helpers/hook-handler.cjs pre-bash",
      "timeout": 3000
    }],
    "PostToolUse": [
      {
        "matcher": "Bash",
        "command": "cmd /c node %CLAUDE_PROJECT_DIR%/.claude/helpers/hook-handler.cjs post-bash",
        "timeout": 3000
      },
      {
        "matcher": "Write|Edit|MultiEdit",
        "command": "cmd /c node %CLAUDE_PROJECT_DIR%/.claude/helpers/hook-handler.cjs post-edit",
        "timeout": 3000
      }
    ]
  }
}
```

**hook-handler.cjs path** (user's system): `/mnt/c/Users/<username>/.claude/helpers/hook-handler.cjs`

**Note:** Commands use `cmd /c node ...` because Claude Code runs on Windows and spawns CMD processes from the shell. From WSL, test hooks with `cd /mnt/c/Users/<user> && node .claude/helpers/hook-handler.cjs <command>`.

**WSL-to-Windows hook testing pattern** (for benchmarking hook overhead):

```python
# Python benchmark — measure hook latency from WSL
import subprocess, time, json

WORKDIR = "/mnt/c/Users/<user>"  # Windows home from WSL
HOOKS = f"{WORKDIR}/.claude/helpers"

def run_hook(name, args=""):
    cmd = (f"cd {WORKDIR} && node {HOOKS}/hook-handler.cjs {args}"
           if "mjs" not in args
           else f"cd {WORKDIR} && node {HOOKS}/{args}")
    start = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
    return time.time() - start, r.returncode

# Test each hook 5x for average
for name, args in [("pre-bash","pre-bash"), ("post-bash","post-bash"),
                   ("post-edit","post-edit"), ("session-restore","session-restore")]:
    times = [run_hook(name, args)[0] for _ in range(5)]
    print(f"  {name}: {sum(times)/len(times):.3f}s avg")
```

**API latency benchmark** (measure model response time):

```bash
# 10 calls to measure avg latency
for i in $(seq 1 10); do
  curl -s -w "\nTIME:%{time_total}" -X POST https://opencode.ai/zen/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"minimax-m2.5-free","messages":[{"role":"user","content":"hi"}],"max_tokens":5}' \
    | tail -1
done | awk '{sum+=$1; n++} END {print "Avg:", sum/n "s"}'
```

**Why benchmark first:** Model latency (1.5–2.8s for MiniMax M2.5 via OpenCode Zen) dominates over hook overhead (~0.25s/task). Benchmarks tell you whether optimizing hooks will matter or whether the bottleneck is elsewhere.

## Latest Updates (June 2026)

### Claude Opus 4.8 (current default)
- Default model on Max, Team Premium, Enterprise, and Anthropic API
- Defaults to high effort; use `/effort xhigh` for harder tasks
- Requires Claude Code v2.1.154 or later

### Dynamic Workflows (research preview)
- Orchestration scripts that Claude runs across many subagents in background
- Use cases: codebase-wide audits, big migrations, cross-checked research
- Manage with `/workflows`

### Auto Mode expanded
- Now available on Bedrock, Vertex, and Foundry for Opus 4.7 and 4.8
- Opt in: `CLAUDE_CODE_ENABLE_AUTO_MODE=1`

### Plugin system
- Plugins in `.claude/skills/` load automatically — no marketplace needed
- `claude plugin init <name>` scaffolds a new plugin
- Autocomplete for `/plugin` arguments

### Claude Code on the web (research preview)
- Runs on Anthropic-managed cloud infrastructure at `claude.ai/code`
- Sessions persist after closing browser
- Monitor from Claude mobile app
- Move sessions between web and terminal with `--remote` and `--teleport`

### Other fixes
- `rm -rf $HOME` with trailing slash on HOME now properly blocked
- Background session and worktree bugs fixed
- VS Code rendering corruption fixed
- Windows update failure messages improved

## Absorbed Agent System (98 files)

### What Was Absorbed (practical)
- **Core Agents (5)**: coder, planner, researcher, reviewer, tester — now native as `agent-personas` skill
- **Security patterns**: Regex for secrets, injection, XSS — now native as `security-scanning-patterns` skill
- **SPARC methodology**: 5-phase dev approach — now native as `sparc-methodology` skill
- **Hook architecture**: Central dispatcher, ALWAYS exit 0, timeout protection

### What Was Skipped (theoretical/distributed)
- Consensus protocols (Byzantine, Raft, Gossip, CRDT) — distributed systems I'm not running
- Swarm coordinators (hierarchical, mesh, adaptive) — multi-agent infrastructure I don't need
- V3 specialized agents (15+) — Claude Flow internal infrastructure
- Intelligence system (PageRank, HNSW, trigram matching) — Claude Flow helper internals
- Learning service (SQLite + ONNX embeddings) — Claude Flow helper internals
- Swarm communication (file-based messaging, consensus voting) — distributed coordination
- Background workers (7 periodic tasks) — I use `cronjob` instead