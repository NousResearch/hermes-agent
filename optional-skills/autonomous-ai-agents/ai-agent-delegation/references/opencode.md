# OpenCode CLI

**Full skill:** `autonomous-ai-agents/opencode` (archived)

## Quick Reference

```bash
# Install
npm i -g opencode-ai@latest
# or: brew install anomalyco/tap/opencode

# Auth
opencode auth login
# Verify
opencode auth list
opencode --version
```

## Binary Resolution

If behavior differs between your terminal and Hermes, check for multiple installations:
```bash
terminal(command="which -a opencode")
terminal(command="$HOME/.opencode/bin/opencode --version")
```
Pin explicit path if needed: `$HOME/.opencode/bin/opencode run '...'`

## One-Shot Tasks

Use `opencode run` for bounded, non-interactive tasks (no PTY needed):

```bash
terminal(command="opencode run 'Add retry logic to API calls and update tests'", workdir="~/project")
```

With attached context files:
```bash
terminal(command="opencode run 'Review this config for security issues' -f config.yaml -f .env.example", workdir="~/project")
```

Other options:
- `--thinking` — show model thinking blocks
- `--model openrouter/anthropic/claude-sonnet-4` — force specific model
- `--agent build|plan` — choose agent mode
- `--variant high|max|minimal` — reasoning effort
- `--title <name>` — name the session

## Interactive TUI Sessions

Start with `background=true, pty=true`:

```bash
terminal(command="opencode", workdir="~/project", background=true, pty=true)
# Returns session_id

# Send a prompt
process(action="submit", session_id="<id>", data="Implement OAuth refresh flow and add tests")

# Monitor progress
process(action="poll", session_id="<id>")
process(action="log", session_id="<id>")

# Send follow-up input
process(action="submit", session_id="<id>", data="Now add error handling for token expiry")

# Exit — use Ctrl+C, NOT /exit
process(action="write", session_id="<id>", data="\x03")
# Or kill
process(action="kill", session_id="<id>")
```

### TUI Keybindings

| Key | Action |
|-----|--------|
| `Enter` | Submit (press twice if needed) |
| `Tab` | Switch between agents (build/plan) |
| `Ctrl+P` | Open command palette |
| `Ctrl+X L` | Switch session |
| `Ctrl+X M` | Switch model |
| `Ctrl+X N` | New session |
| `Ctrl+X E` | Open editor |
| `Ctrl+C` | Exit OpenCode |

### Session Resumption

After exiting, OpenCode prints a session ID:
```bash
terminal(command="opencode -c", workdir="~/project", background=true, pty=true)       # Continue last
terminal(command="opencode -s ses_abc123", workdir="~/project", background=true, pty=true)  # Specific
```

## Common Flags

| Flag | Use |
|------|-----|
| `run 'prompt'` | One-shot execution and exit |
| `--continue` / `-c` | Continue the last OpenCode session |
| `--session <id>` / `-s` | Continue a specific session |
| `--agent <name>` | Choose OpenCode agent (build or plan) |
| `--model provider/model` | Force specific model |
| `--format json` | Machine-readable output/events |
| `--file <path>` / `-f` | Attach file(s) to the message |
| `--thinking` | Show model thinking blocks |
| `--variant <level>` | Reasoning effort (high, max, minimal) |
| `--title <name>` | Name the session |
| `--attach <url>` | Connect to a running opencode server |

## PR Review

Built-in PR command:
```bash
terminal(command="opencode pr 42", workdir="~/project", pty=true)
```

In a temporary clone for isolation:
```bash
terminal(command="REVIEW=$(mktemp -d) && git clone https://github.com/user/repo.git $REVIEW && cd $REVIEW && opencode run 'Review this PR vs main. Report bugs, security risks, test gaps, and style issues.' -f $(git diff origin/main --name-only | head -20 | tr '\n' ' ')", pty=true)
```

## Session & Cost Management

```bash
terminal(command="opencode session list")
terminal(command="opencode stats")
terminal(command="opencode stats --days 7 --models anthropic/claude-sonnet-4")
```

## Smoke Test

```bash
terminal(command="opencode run 'Respond with exactly: OPENCODE_SMOKE_OK'")
# Success: output includes OPENCODE_SMOKE_OK
```

## Rules

1. Prefer `opencode run` for one-shot automation — simpler, no PTY needed
2. Use interactive background mode only when iteration is needed
3. Always scope OpenCode sessions to a single repo/workdir
4. For long tasks, provide progress updates from `process` logs
5. Report concrete outcomes (files changed, tests, remaining risks)
6. **Exit interactive sessions with Ctrl+C or kill — NEVER `/exit`** (it opens an agent selector dialog)
7. Enter may need to be pressed twice to submit in the TUI (once to finalize text, once to send)