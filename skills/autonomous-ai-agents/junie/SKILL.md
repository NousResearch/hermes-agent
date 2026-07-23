---
name: junie
description: "Delegate coding to the JetBrains Junie CLI (plan, implement, review)."
version: 1.0.0
author: Hermes Agent + JetBrains
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Coding-Agent, Junie, JetBrains, Plan-Mode, Code-Review, PTY, Automation]
    related_skills: [claude-code, codex, hermes-agent, opencode]
---

# Junie — Hermes Orchestration Guide

Delegate coding tasks to [JetBrains Junie](https://junie.jetbrains.com/docs/junie-cli.html) — JetBrains' autonomous, LLM-agnostic coding agent CLI — via the Hermes terminal. Junie reads/edits files, runs commands, plans before acting, and reviews diffs. It brings its own agent harness (plan mode, code review, orchestrated sub-agents), so you delegate a *goal*, not step-by-step tool calls.

> Note: this is the **delegation** skill — Hermes (on any model) drives the `junie` CLI as a tool. It is distinct from the `junie-acp` **provider**, where Junie *is* the model backend driving Hermes. Use this skill when your current agent wants to hand a coding subtask to Junie.

## Prerequisites

- **Install:** `curl -fsSL https://junie.jetbrains.com/install.sh | bash` (EAP: `install-eap.sh`). Also PowerShell on Windows. Binary lands at `~/.local/bin/junie`.
- **Auth (pick one):**
  - JetBrains/Junie token: `export JUNIE_API_KEY='perm-...'` (generate at https://junie.jetbrains.com/tokens) or pass `--auth "$JUNIE_API_KEY"`.
  - Interactive login: run `junie` once and sign in via the Account screen (browser).
  - **BYOK** (bring your own model key): `--openai-api-key`, `--anthropic-api-key`, `--google-api-key`, `--grok-api-key`, `--openrouter-api-key`, or a LiteLLM proxy (`--litellm-url` + `--litellm-api-key`).
- **Version:** `junie --version`. Add `--skip-update-check` in automation to avoid the startup update check.

## Two Orchestration Modes

### Mode 1: Headless / Non-Interactive (PREFERRED for most tasks)

One-shot: give Junie a task, it works autonomously and exits. No PTY, no prompts.

```
terminal(command="junie --auth=\"$JUNIE_API_KEY\" --skip-update-check --output-format json --json-output-file result.json 'Add error handling to all API calls in src/'", workdir="/path/to/project", timeout=180)
```

- The task is the positional arg (or `--task "..."`).
- `--output-format text|json|json-stream`; prefer **`json` + `--json-output-file result.json`** then read the file — plain `text` output carries ANSI color codes that are messy to parse.
- `--input-format text|json` accepts structured/piped input.
- `-p, --project <dir>` sets the project dir (or just set `workdir`). ⚠️ In Junie `-p` means **project**, not print — there is no `-p` print flag like Claude Code.

**When to use:** bug fixes, features, refactors, CI/CD automation, structured extraction. Junie runs its own plan→implement→verify loop internally.

### Mode 2: Interactive PTY via tmux — Multi-Turn Sessions

```
terminal(command="tmux new-session -d -s junie-work -x 140 -y 40")
terminal(command="tmux send-keys -t junie-work 'cd /path/to/project && junie' Enter")
# wait for the welcome screen (~5s), then send the task
terminal(command="sleep 6 && tmux send-keys -t junie-work 'Refactor the auth module to use JWT' Enter")
# monitor
terminal(command="sleep 15 && tmux capture-pane -t junie-work -p -S -60")
# follow-up
terminal(command="tmux send-keys -t junie-work 'Now add unit tests for the JWT code' Enter")
# exit
terminal(command="tmux send-keys -t junie-work '/exit' Enter")
```

**When to use:** iterative work, human-in-the-loop, exploratory sessions, or to use Junie's slash commands (`/plan`, `/review`, `/usage`).

## Junie-Specific Capabilities

- **Plan mode** — `--plan` (or `/plan` in a session): read-only analysis that proposes a plan before editing. Approve/refine, then implement. Good for risky or large changes.
- **Code review** — `--review` (or `/review`): reviews a git diff (vs main, last commit, or a described scope). Requires a git repo.
- **Orchestrated mode** — `--goal "..."` runs a multi-step task decomposed across sub-agents (plan → code → review → git). CLI/TUI only.
- **Model control** — `--model <id>` (e.g. `claude-opus-4-8`, `gemini-3-flash-preview`, `gpt-5.x`, `grok-4.3`), `--effort low|medium|high`, `--provider <byok>`.
- **Brave Mode** — `--brave` executes commands without asking (interactive). Otherwise Junie asks before acting.
- **MCP** — Junie is an MCP client: configure servers in `.junie/mcp/mcp.json` (project) or `~/.junie/mcp/mcp.json` (user); `/mcp` lists them.
- **Skills / commands / guidelines** — user-authored, loaded from `~/.junie/` and `.junie/` (Junie does not auto-create skills or persist cross-session memory in the CLI).

## Session Continuation

- `--resume` resumes the last session; `--session-id <id>` follows a specific one.

## Pitfalls & Gotchas

- **ANSI in text output:** use `--output-format json --json-output-file` for clean, parseable results in automation.
- **Cold start:** the first invocation pays a JVM/agent startup cost (several seconds). Interactive tmux sessions reuse the process across turns.
- **Auth in automation:** pass `--auth "$JUNIE_API_KEY"` (perm-...) explicitly, or ensure `JUNIE_API_KEY` is in the process env; headless runs won't open a browser login.
- **`--goal` is CLI/TUI only** and not available in headless JSON pipelines the same way — for scripted use prefer a plain task or `--plan`.
- **No `-p` print flag:** `-p` is `--project`. Non-interactive is simply the positional task.

## Rules for Hermes Agents

1. Prefer Mode 1 (headless JSON) for anything scriptable; reach for tmux only when you need multi-turn interaction.
2. Always run inside the target project (`workdir` / `--project`), and set `--skip-update-check` in automation.
3. For large or destructive changes, run `--plan` first, inspect the plan, then implement.
4. Read `result.json` for the outcome rather than scraping colored terminal text.
