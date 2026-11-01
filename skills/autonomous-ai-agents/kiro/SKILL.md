---
name: kiro
description: "Delegate coding to Kiro CLI (AWS agentic dev environment)."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Coding-Agent, Kiro, AWS, Code-Review, Refactoring]
    related_skills: [claude-code, codex, opencode, hermes-agent]
---

# Kiro CLI

Delegate coding tasks to [Kiro](https://app.kiro.dev/) via the Hermes terminal. Kiro is AWS's agentic development environment — the successor to Amazon Q Developer. It provides AI-assisted coding through CLI with multi-model support (Claude, DeepSeek, MiniMax, etc.) and spec-driven development.

## When to use

- Building features
- Refactoring
- PR reviews
- Batch issue fixing
- Spec-driven development (Kiro's strength — turning prompts into structured requirements, designs, and tasks)

Requires the `kiro-cli` binary. No git repo required (unlike Codex).

## Prerequisites

- **Install Kiro CLI:**
  ```bash
  curl -fsSL https://cli.kiro.dev/install | bash
  ```
  Other install methods:
  - **Linux .deb (Ubuntu):**
    ```bash
    wget https://desktop-release.q.us-east-1.amazonaws.com/latest/kiro-cli.deb
    sudo dpkg -i kiro-cli.deb
    sudo apt-get install -f
    ```
  - **Linux zip (x86_64 or aarch64):**
    ```bash
    curl -sSf 'https://desktop-release.q.us-east-1.amazonaws.com/latest/kirocli-x86_64-linux.zip' -o kirocli.zip
    unzip kirocli.zip
    ./kirocli/install.sh
    ```
  - **Windows (PowerShell):**
    ```powershell
    irm 'https://cli.kiro.dev/install.ps1' | iex
    ```

- **Authentication** — multiple methods (see `references/kiro-auth-setup.md`):
  - GitHub, Google, AWS Builder ID, AWS IAM Identity Center
  - API Key (headless mode, requires Pro/Pro+/Power tier)

- **Use `pty=true` in terminal calls** — Kiro CLI is an interactive terminal app with rich TUI

- **Use `--no-interactive` for headless mode** — non-interactive execution for automated tasks

- Binary names: `kiro-cli` (main), `kiro-cli-chat` (chat interface)

## One-Shot Tasks

```bash
# Interactive (requires PTY)
terminal(command="kiro-cli chat", workdir="~/project", pty=true)

# Headless one-shot (no interactive UI)
terminal(command="kiro-cli chat --no-interactive 'Add dark mode toggle to settings'", workdir="~/project", pty=true)
```

For scratch work (no git repo needed):
```bash
terminal(command="cd $(mktemp -d) && kiro-cli chat --no-interactive 'Build a snake game in Python'", pty=true)
```

## Background Mode (Long Tasks)

```bash
# Start in background with PTY
terminal(command="kiro-cli chat --no-interactive 'Refactor the auth module'", workdir="~/project", background=true, pty=true)
# Returns session_id

# Monitor progress
process(action="poll", session_id="<id>")
process(action="log", session_id="<id>")

# Send input if Kiro asks a question
process(action="submit", session_id="<id>", data="yes")

# Kill if needed
process(action="kill", session_id="<id>")
```

## Resume Sessions

```bash
terminal(command="kiro-cli chat --resume-id <SESSION_ID>", workdir="~/project", pty=true)
```

## Key CLI Flags and Commands

| Command / Flag | Effect |
|----------------|--------|
| `kiro-cli chat` | Launch interactive session |
| `kiro-cli chat --no-interactive "prompt"` | Headless one-shot execution |
| `kiro-cli chat --resume-id <ID>` | Resume a previous session |
| `kiro-cli login` | Authenticate (opens browser) |
| `kiro-cli logout` | Sign out |
| `kiro-cli whoami` | Check auth status |
| `kiro-cli doctor` | Diagnose issues |
| `kiro-cli settings <key> <value>` | Configure settings |
| `kiro-cli settings list` | List all settings |

### Slash Commands (in-session)

| Command | Description |
|---------|-------------|
| `/help` | All available commands |
| `/plan` | Enter plan mode (also Shift+Tab) |
| `/model` | Switch active model |
| `/mcp` | View MCP servers and registry |
| `/tools` | View/reset tool permissions |
| `/hooks` | View configured hooks |
| `/code` | Code intelligence panel |
| `/context` | Context breakdown with per-file token % |
| `/usage` | Usage limits with progress bar |
| `/spawn` | Run parallel agent session |
| `/effort` | Set reasoning effort (low/medium/high/xhigh/max) |
| `/rewind` | Fork conversation at earlier turn |
| `/copy` | Copy last assistant response |
| `/clear` | Clear conversation display |
| `/<skill-name>` | Invoke a skill (e.g., `/pr-review`) |
| `!command` | Shell escape — run shell commands |

### File References

Use `@` with tab completion:
```
> Review @src/api/routes.ts for security issues
```

## Models Available

Kiro supports a mix of frontier and open-weight models. Default is **Auto** (model router).

| Model | Context | Cost | Free | Pro | Pro+ | Power |
|-------|---------|------|------|-----|------|-------|
| Claude Opus 4.7 | 1M | 2.2x | — | ✓ | ✓ | ✓ |
| Claude Opus 4.6 | 1M | 2.2x | — | ✓ | ✓ | ✓ |
| Claude Sonnet 4.6 | 1M | 1.3x | — | ✓ | ✓ | ✓ |
| Claude Sonnet 4.5 | 200K | 1.3x | ✓ | ✓ | ✓ | ✓ |
| Claude Sonnet 4.0 | 200K | 1.3x | ✓ | ✓ | ✓ | ✓ |
| Auto (router) | — | 1.0x | ✓ | ✓ | ✓ | ✓ |
| Claude Haiku 4.5 | 200K | 0.4x | — | ✓ | ✓ | ✓ |
| DeepSeek 3.2 | 128K | 0.25x | ✓ | ✓ | ✓ | ✓ |
| MiniMax M2.5 | 200K | 0.25x | ✓ | ✓ | ✓ | ✓ |
| GLM-5 | 200K | 0.5x | ✓ | ✓ | ✓ | ✓ |
| Qwen3 Coder Next | 256K | 0.05x | ✓ | ✓ | ✓ | ✓ |

Switch model:
```bash
kiro-cli settings chat.defaultModel claude-opus-4.7
```
Or in-session: `/model` → `/model set-current-as-default`

## Auth Setup

Kiro supports multiple auth methods. See `references/kiro-auth-setup.md` for detailed setup.

### 1. Interactive Login (browser-based)

```bash
kiro-cli login
```

Opens a browser where you choose your auth provider (GitHub, Google, AWS Builder ID, etc.), then redirects back to terminal.

### 2. API Key (headless / CI/CD) — Pro/Pro+/Power tiers only

```bash
export KIRO_API_KEY=ksk_xxxxxxxx
kiro-cli chat --no-interactive "your prompt here"
```

### Auth Precedence

1. Active browser session (from `kiro-cli login`)
2. `KIRO_API_KEY` environment variable
3. No credentials → CLI prompts sign-in

### Check Auth / Logout

```bash
kiro-cli whoami    # check auth status
kiro-cli logout    # sign out
```

## Pricing

| Tier | Price | Credits |
|------|-------|---------|
| Kiro Free | $0/mo | 50 credits |
| Kiro Pro | $20/mo | 1,000 credits |
| Kiro Pro+ | $40/mo | 2,000 credits |
| Kiro Power | $200/mo | 10,000 credits |

Free tier includes open-weight models + Claude Sonnet 4.5. Paid tiers include premium models.

## PR Reviews

```bash
terminal(command="REVIEW=$(mktemp -d) && git clone https://github.com/user/repo.git $REVIEW && cd $REVIEW && gh pr checkout 42 && kiro-cli chat --no-interactive 'Review this PR. diff: $(git diff origin/main...HEAD)'", pty=true)
```

## Parallel Issue Fixing

```bash
# Create worktrees
terminal(command="git worktree add -b fix/issue-78 /tmp/issue-78 main", workdir="~/project")

# Launch Kiro in each
terminal(command="kiro-cli chat --no-interactive 'Fix issue #78: <description>. Commit when done.'", workdir="/tmp/issue-78", background=true, pty=true)

# Monitor
process(action="list")

# After completion, push and create PRs
terminal(command="cd /tmp/issue-78 && git push -u origin fix/issue-78")
terminal(command="gh pr create --repo user/repo --head fix/issue-78 --title 'fix: ...' --body '...'")
```

## Rules

1. **Always use `pty=true`** — Kiro CLI is an interactive terminal app and hangs without a PTY
2. **No git repo required** — Unlike Codex, Kiro works outside git repos
3. **Use `--no-interactive` for automated tasks** — headless mode for non-interactive execution
4. **Background for long tasks** — use `background=true` and monitor with `process` tool
5. **Don't interfere** — monitor with `poll`/`log`, be patient with long-running tasks
6. **Parallel is fine** — run multiple Kiro processes at once for batch work
7. **Model selection** — use `/model` or `kiro-cli settings` to pick model; default is Auto

## Key Differences from Codex

| Feature | Codex | Kiro |
|---------|-------|------|
| Provider | OpenAI | AWS |
| Install | `npm install -g @openai/codex` | `curl -fsSL https://cli.kiro.dev/install \| bash` |
| Git required | Yes | No |
| Auth | OAuth (ChatGPT) or API Key | GitHub, Google, AWS Builder ID, API Key |
| Models | `gpt-5.5`, `o3`, `o1` | Claude Opus/Sonnet, DeepSeek, MiniMax, GLM-5, Qwen3 |
| One-shot | `codex exec "prompt"` | `kiro-cli chat --no-interactive "prompt"` |
| Interactive | `codex exec` (PTY) | `kiro-cli chat` (PTY) |
| Spec-driven | No | Yes (turns prompts into structured requirements) |
| MCP support | No | Yes |
| Agent hooks | No | Yes (pre/post command hooks) |

## Pitfalls

1. **`pty=true` is mandatory** — Kiro CLI hangs without PTY, same as Codex
2. **API key requires paid tier** — Free tier cannot use `KIRO_API_KEY` for headless mode
3. **Auto-update can break** — Disable with `kiro-cli settings "app.disableAutoupdates" "true"`
4. **Credit metering** — Kiro uses credits; monitor with `/usage` in-session
5. **No npm package** — Kiro CLI is a binary, not an npm package; install via curl/PowerShell
6. **Session resume** — Use `--resume-id` to continue a previous session; sessions are tracked by Kiro
