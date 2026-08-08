---
name: copilot-cli
description: Delegate coding tasks to GitHub Copilot CLI. Use for building features, refactoring, PR reviews, and iterative coding via either the standalone `copilot -p` print mode or the Hermes-native ACP transport. Requires the `copilot` CLI installed and authenticated with a GitHub account that has Copilot access.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [Coding-Agent, Copilot, GitHub, ACP, Code-Review, Refactoring, Automation]
    related_skills: [claude-code, codex, opencode, hermes-agent]
---

# GitHub Copilot CLI — Hermes Orchestration Guide

Delegate coding tasks to [GitHub Copilot CLI](https://docs.github.com/copilot/how-tos/copilot-cli) (`copilot`). Copilot CLI is GitHub's autonomous coding agent — it can read files, write code, run shell commands, manage MCP servers, and operate in either an interactive TUI or a one-shot scripted mode. Hermes can drive it in **three** different ways depending on the task.

## Prerequisites

- **Install:** `npm install -g @github/copilot` (Node 18+) or via your package manager
- **Auth:** run `copilot` once and complete `/login`, or set `COPILOT_GITHUB_TOKEN` / `GH_TOKEN`
- **Account:** the GitHub account must have an active Copilot subscription (Individual, Business, or Enterprise)
- **Verify:** `copilot --version` and `copilot --help`

## Three Orchestration Modes

| Mode | When to use | Transport |
|------|-------------|-----------|
| **1. ACP delegation** (`delegate_task(acp_command="copilot")`) | **PREFERRED.** You want Copilot to act as a fully integrated subagent inside Hermes — sharing the parent's working context, returning a summary, no PTY juggling. | `copilot --acp --stdio` spawned by Hermes' built-in `agent/copilot_acp_client.py` |
| **2. Print mode** (`copilot -p`) | One-shot scripting from terminal — analysing a diff, applying a single fix, generating a structured report you want as raw text. | `terminal(command="copilot -p ...")` |
| **3. Interactive TUI** | Multi-turn iterative work where you want to send follow-up prompts and watch the TUI. Same orchestration pattern as Codex / OpenCode. | `terminal(..., background=true, pty=true)` + `process(...)` |

Each mode is documented below. **For most Hermes-orchestrated coding work, use Mode 1.**

---

## Mode 1: ACP Delegation (PREFERRED)

`delegate_task` with `acp_command="copilot"` spawns a fresh Copilot ACP server (`copilot --acp --stdio`) as the child agent's backend. The child runs in its own conversation, executes Copilot tools (file edits, shell, GitHub MCP) under Copilot's permissions model, and returns a final summary back to the parent. **No PTY, no tmux, no log scraping.**

### Standard Single-Task Pattern

```
delegate_task(
    acp_command="copilot",
    goal="Add error handling to all API calls in src/api/ — wrap fetch() in a retry helper with exponential backoff and 3 max attempts. Update tests.",
    context=(
        "Project: ~/src/myapp (Next.js 14 + TypeScript)\n"
        "Convention: 2-space indent, no semicolons, eslint-config-next\n"
        "Tests live in src/api/__tests__/, run with `npm test`\n"
        "Don't touch src/legacy/ — that module is frozen.\n"
        "Existing retry helper (if any) lives in src/lib/retry.ts"
    ),
    toolsets=["terminal", "file"],
)
```

**What the subagent gets:** the `goal` becomes the initial user prompt, `context` becomes a system-prompt prefix that tells Copilot the project layout, conventions, and constraints. It does **not** inherit the parent's chat history — pass everything Copilot needs in `context`.

### Parallel Batch Pattern (independent tasks)

```
delegate_task(
    tasks=[
        {
            "goal": "Fix the auth bug in src/auth.py and add a regression test",
            "context": "Repo: ~/src/backend. Bug: token refresh races with logout. Test framework: pytest.",
            "toolsets": ["terminal", "file"],
        },
        {
            "goal": "Update README.md with the new /api/v2/ endpoints",
            "context": "Repo: ~/src/backend. New endpoints listed in src/api/v2/routes.py.",
            "toolsets": ["file"],
        },
        {
            "goal": "Run the full test suite and report any failures with stack traces",
            "context": "Repo: ~/src/backend. Run `make test`. Don't try to fix anything — just report.",
            "toolsets": ["terminal"],
        },
    ],
    acp_command="copilot",
)
```

Up to 3 children run concurrently by default (configurable via `delegation.max_concurrent_children` in `~/.hermes/config.yaml`). All results return as a single array.

### When to override `acp_args`

Default args are `["--acp", "--stdio"]`. Override via `acp_args=[...]` to add Copilot flags that the ACP child should inherit:

```
delegate_task(
    acp_command="copilot",
    acp_args=["--acp", "--stdio", "--allow-all-tools", "--add-dir", "/Users/me/src/extra-repo"],
    goal="Cross-repo refactor: align logging conventions between backend and shared lib.",
    context="...",
)
```

Most useful flags to pass through:
- `--allow-all-tools` — required for non-interactive operation; without it Copilot may pause for permission prompts that ACP can't surface to the user
- `--add-dir <path>` — grant access to additional directories beyond the cwd
- `--model <id>` — pin a specific model (e.g. `gpt-5.2`, `claude-sonnet-4-6`)
- `--effort <level>` — `low` / `medium` / `high` / `xhigh`

### Environment Variables that Affect ACP Behaviour

| Variable | Effect |
|----------|--------|
| `HERMES_COPILOT_ACP_COMMAND` | Override the Copilot binary path (default: `copilot` from PATH) |
| `HERMES_COPILOT_ACP_ARGS` | Override the default args string (default: `--acp --stdio`) |
| `COPILOT_GITHUB_TOKEN` / `GH_TOKEN` / `GITHUB_TOKEN` | Auth token, in that precedence order |
| `COPILOT_ALLOW_ALL` | Set to `true` to auto-approve all tools (equivalent to `--allow-all`) |
| `COPILOT_MODEL` | Default model when `--model` not specified |
| `COPILOT_HOME` | Config directory (default: `~/.copilot`) |

### ACP Pitfalls

1. **Approval prompts deadlock the child.** If Copilot asks "Allow shell command X?" the ACP transport has no way to display it. Always pass `--allow-all-tools` (or set `COPILOT_ALLOW_ALL=true`) when delegating non-trivially.
2. **Working directory** — the ACP child inherits Hermes' working directory unless you pass `--add-dir` or instruct it in `context` to `cd` first. State the absolute path in `context`.
3. **No grandchildren.** Hermes blocks recursive `delegate_task` calls inside subagents. The Copilot child cannot itself spawn more children.
4. **Long sessions cost tokens.** ACP spawns a fresh session per delegation; conversation history doesn't persist across calls. Pass everything in `context` rather than chaining many tiny delegations.

---

## Mode 2: Print Mode (`copilot -p`)

Use when you want to call Copilot directly from `terminal()` — typically for a single shell-pipe-style transformation or a quick scripted task that doesn't need to come back as a structured subagent result.

```
terminal(
    command="copilot -p 'Review src/auth.py for security issues. List findings with line numbers.' --allow-all-tools --silent",
    workdir="/Users/me/src/myapp",
    timeout=180,
)
```

### Print Mode Flags

| Flag | Effect |
|------|--------|
| `-p, --prompt <text>` | Run prompt non-interactively, exit when done |
| `--allow-all-tools` | **Required for non-interactive mode** — auto-approve every tool call |
| `--allow-all` / `--yolo` | Equivalent to `--allow-all-tools --allow-all-paths --allow-all-urls` |
| `-s, --silent` | Suppress stats — output only the agent response (good for piping) |
| `--model <id>` | Pin model |
| `--effort <level>` | `low`, `medium`, `high`, `xhigh` |
| `--add-dir <path>` | Grant additional directory access (repeatable) |
| `--allow-tool <pattern>` | Whitelist specific tool patterns, e.g. `'shell(git:*)'` |
| `--deny-tool <pattern>` | Blacklist specific tool patterns; takes precedence over allow |
| `--allow-url <domain>` | Whitelist URLs for the web tools |
| `--share <path>` | Save session as Markdown after completion |
| `--share-gist` | Save session as a secret GitHub gist |
| `--continue` | Resume the most recent session in this directory |
| `--resume[=id]` | Resume a specific session by ID or prefix |
| `--stream on\|off` | Streaming mode toggle |
| `--plan` | Start in plan mode (Copilot drafts a plan before acting) |
| `--autopilot` | Start in autopilot mode |

### Permission Pattern Syntax

```
shell(git:*)              # all git subcommands
shell(git push)           # only git push
shell                     # all shell commands
write                     # all file write/edit tools
url(https://github.com)   # exact URL
url(https://*.github.com) # wildcard subdomain
<mcp-server>(<tool>)      # specific MCP tool from a named server
```

Denial always wins over allow.

### Piping Input

```
terminal(command="git diff main...HEAD | copilot -p 'Summarise these changes for a PR description' --allow-all-tools --silent", workdir="/Users/me/src/myapp", timeout=120)
```

### Capturing the Session

```
terminal(command="copilot -p 'Refactor the database layer to use async SQLAlchemy' --allow-all --share /tmp/copilot-refactor.md", workdir="/Users/me/src/myapp", timeout=600)
```

Then `read_file("/tmp/copilot-refactor.md")` to get the full transcript.

---

## Mode 3: Interactive TUI (background + PTY)

Same pattern as the `codex` and `opencode` skills. Use when you need a multi-turn conversation that you'll keep poking with follow-ups.

```
# Launch in background with PTY
terminal(command="copilot --allow-all", workdir="/Users/me/src/myapp", background=true, pty=true)
# → returns session_id, e.g. "bg_abc123"

# Wait for the prompt to settle, then send your task
process(action="submit", session_id="bg_abc123", data="Implement OAuth refresh flow and add tests")

# Monitor progress
process(action="poll",  session_id="bg_abc123")
process(action="log",   session_id="bg_abc123")

# Send follow-up
process(action="submit", session_id="bg_abc123", data="Now wire up rate-limit handling for the refresh endpoint")

# Exit cleanly with Ctrl+C
process(action="write", session_id="bg_abc123", data="\x03")
process(action="kill",  session_id="bg_abc123")
```

### Useful Slash Commands (Interactive)

| Command | Purpose |
|---------|---------|
| `/init` | Generate `COPILOT.md` repo instructions |
| `/plan` | Draft an implementation plan before coding |
| `/diff` | Show changes made this session |
| `/pr` | Create or update a PR for the current branch |
| `/review` | Run a code review pass on current changes |
| `/research` | Deep research using GitHub search + web sources |
| `/agent`, `/skills`, `/mcp`, `/plugin` | Manage Copilot's own subagents, skills, MCP servers, plugins |
| `/model`, `/effort` | Switch model or reasoning effort mid-session |
| `/context`, `/usage`, `/compact` | Inspect / shrink context window |
| `/share`, `/share-gist` | Export the session |
| `/resume`, `/new`, `/continue` | Session management |
| `/exit` | End the session |

### Custom Subagents

Copilot can run its own subagents via `--agent <name>` or the `/agent` picker. Define them in `~/.copilot/agents/<name>.md`. From Hermes, just pass `--agent` through `acp_args` (Mode 1) or `command` (Mode 2/3).

---

## Choosing Between Copilot, Claude Code, Codex, OpenCode

| Situation | Pick |
|-----------|------|
| Default for this user (皇上) | **Copilot via ACP** (`delegate_task(acp_command="copilot")`) |
| Need deepest reasoning + Anthropic ecosystem (CLAUDE.md, hooks, worktrees) | `claude-code` |
| OpenAI ecosystem, codex-style PR review batch | `codex` |
| Provider-agnostic OSS agent, full TUI with `/plan` mode | `opencode` |
| Tight GitHub integration (`/pr`, `/review`, GitHub MCP, gist sharing) | `copilot-cli` |
| Tasks that benefit from GitHub's deep code search & repo grounding | `copilot-cli` |

## Cost & Performance Tips

1. **Default to ACP delegation** — cleaner integration, no terminal noise, automatic result capture.
2. **Always pass `--allow-all-tools` for non-interactive runs** — otherwise Copilot blocks waiting for an approval prompt that nothing can answer.
3. **Use `--effort low`** for trivial tasks (rename, comment, single-file fix) and `xhigh` only for genuine multi-file reasoning.
4. **Prefer `--silent` in Mode 2** — strips the stats footer so the output is clean for downstream parsing.
5. **Pin a `--model`** when you want reproducibility; otherwise Copilot picks based on your subscription tier.
6. **Set `--add-dir` once, generously** — Copilot otherwise refuses access outside cwd, causing partial work.
7. **Use `--share` or `--share-gist`** for tasks where you want a permanent transcript without re-running.
8. **Resume long sessions** with `--continue` rather than restarting — preserves context and saves tokens.
9. **Cap multi-task delegations at 3** (`max_concurrent_children` default) to avoid rate-limit bursts.

## Pitfalls & Gotchas

1. **Approval prompts kill non-interactive runs.** Use `--allow-all-tools` (or `COPILOT_ALLOW_ALL=true`) every time you're not driving the TUI by hand.
2. **`--allow-all-tools` does not grant path or URL access.** For full bypass use `--allow-all` or `--yolo`. For surgical control, combine `--allow-tool` / `--allow-url` patterns.
3. **Path verification is strict.** Files outside the cwd or `--add-dir` whitelist trigger prompts (even with `--allow-all-tools`). Add `--allow-all-paths` if you genuinely need wildcard file access.
4. **`COPILOT_OFFLINE=true` disables web, GitHub MCP, telemetry, and auth** — only useful with a local model via `COPILOT_PROVIDER_BASE_URL`.
5. **The TUI is a full-screen app.** It hangs without `pty=true` in Mode 3. Use Ctrl+C (`\x03`) to exit, **not** `/exit` (that may open a picker dialog rather than quit, depending on version).
6. **Slash commands only work in the TUI.** In `-p` mode, describe what you want in natural language ("create a PR titled ..." instead of `/pr`).
7. **GitHub MCP toolset is restricted by default.** Use `--add-github-mcp-tool '*'` or `--add-github-mcp-toolset all` to expose more.
8. **Auto-update can run in the middle of a session** unless `COPILOT_AUTO_UPDATE=false` or `--no-auto-update`. Disable for reproducible CI runs.
9. **`--continue` is per-directory.** Resuming from a different cwd won't find the session.
10. **BYOK (custom provider) requires four env vars together** — `COPILOT_PROVIDER_BASE_URL`, `COPILOT_PROVIDER_TYPE`, `COPILOT_PROVIDER_API_KEY` (or `_BEARER_TOKEN`), and `COPILOT_PROVIDER_WIRE_API` (use `responses` for GPT-5 series). Setting only one silently falls back to GitHub-routed Copilot.

## Rules for Hermes Agents

1. **Default to ACP delegation** (Mode 1) for any non-trivial coding task — it's the cleanest integration.
2. **Always pass `--allow-all-tools` (or `COPILOT_ALLOW_ALL=true`) for non-interactive runs** — otherwise the child deadlocks on approval prompts.
3. **State the working directory explicitly in `context`** — ACP children don't inherit your chat's mental model of "the current project".
4. **Keep `goal` action-oriented and `context` factual.** `goal` is "what to do", `context` is "the world it's doing it in".
5. **Restrict `toolsets`** to what the task actually needs (`["terminal", "file"]` is enough for most coding work).
6. **For batch work, use the `tasks=[...]` array** — up to 3 parallel by default. Don't spawn 10 sequential `delegate_task` calls.
7. **For one-shot transformations of stdin/stdout, prefer Mode 2** (`copilot -p ... --silent`) — simpler than spinning up an ACP child.
8. **Reserve Mode 3 (interactive TUI)** for genuine multi-turn iterative sessions where you'll send follow-ups manually.
9. **Clean up background sessions** — `process(action="kill", ...)` when done; long-lived TUI sessions hold tokens and a PTY.
10. **Report back to the user** with what Copilot did, what changed, and any open questions — don't just dump Copilot's raw output.
