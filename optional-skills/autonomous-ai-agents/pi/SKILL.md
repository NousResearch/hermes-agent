---
name: pi
description: "Delegate coding to the Pi Coding Agent (pi.dev) CLI."
version: 1.0.0
author: Neumannzc
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Coding-Agent, Pi, pi.dev, Code-Review, Refactoring, Automation]
    related_skills: [claude-code, codex, opencode, coding-agent-delegation, hermes-agent]
---

# Pi Coding Agent — Hermes Orchestration Guide

Delegate coding tasks to [Pi](https://pi.dev) (the minimal terminal coding harness, npm package `@earendil-works/pi-coding-agent`) via the Hermes terminal. Pi is provider-agnostic (Anthropic, OpenAI, Google, DeepSeek, local models, etc.) and ships a small core (read/write/edit/bash + grep/find/ls) extended via skills, extensions, prompt templates, and themes.

## When to Use

- User explicitly asks to use Pi / pi.dev
- You want an external coding agent for implementation, refactor, review, or batch fixes
- You need a minimal, fast harness without Claude-Code-style permission dialogs
- Parallel work in isolated workdirs/worktrees

## Prerequisites

- **Install:** `npm install -g --ignore-scripts @earendil-works/pi-coding-agent` (or `curl -fsSL https://pi.dev/install.sh | sh`)
- **Auth:** set a provider API key env var (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, …) or run `pi` once and `/login` for a subscription (Anthropic Pro/Max, ChatGPT Plus/Pro, GitHub Copilot)
- **Verify:** `pi --version` (v0.84.1 verified on this machine); `pi auth` prints credentials / checks provider readiness
- **Model catalogs:** refresh with `pi update --models` if a provider's models look stale

## ⚠️ Security Model (READ FIRST)

**Pi has NO built-in permission system.** It runs with the permissions of the user/process that launched it — there are no approval dialogs, no sandbox, no `--dangerously-skip-permissions` equivalent to opt into. Hermes must enforce boundaries itself:

1. **Prefer `--tools` allowlist** for every delegated run:
   ```
   pi -p "Review src/ for bugs" --tools read,grep,find,ls      # read-only
   pi -p "Add retry logic to api.py" --tools read,edit,write,bash,grep,find,ls
   ```
2. **`--no-tools` / `-nt`** disables ALL tools (planning/analysis only).
3. **Containerize for untrusted work** — official patterns in `packages/coding-agent/docs/containerization.md`: Gondolin extension (host `pi` + auth, tools routed into a Linux micro-VM), plain Docker, or OpenShell policy sandbox.
4. **Scoped workdir + clean git status** before launch, `git diff` review after — process boundaries are your safety layer, exactly as with Codex.

## Invocation Modes

### Print Mode (`-p`) — PREFERRED for one-shots

```
terminal(command="pi -p 'Add error handling to all API calls in src/'", workdir="/path/to/project", timeout=180)
```

Piped stdin is merged into the initial prompt:

```
terminal(command="cat src/auth.py | pi -p 'Review this code for bugs'", timeout=120)
```

Attach files with `@` prefix:

```
terminal(command="pi -p @src/api.ts @src/auth.ts 'Review these files for security issues'", workdir="/project", timeout=180)
```

### JSON Event Stream Mode (`--mode json`)

```
terminal(command="pi --mode json 'List all TODO comments'", workdir="/project", timeout=180)
```

Prints JSON Lines to stdout. First line is the session header `{"type":"session","version":3,"id":"...","cwd":"..."}`, then lifecycle events: `agent_start`, `turn_start`, `message_start`, `message_update` (with `assistantMessageEvent.type == "text_delta"` for live text), `tool_execution_start/update/end`, `message_end`, `turn_end`, `agent_end`. Filter with jq:

```
pi --mode json "Explain X" | jq -rj 'select(.type=="message_update" and .assistantMessageEvent.type=="text_delta") | .assistantMessageEvent.delta'
```

### Interactive TUI (background, PTY)

```
terminal(command="pi", workdir="/project", background=true, pty=true)
# Returns session_id — monitor with process(action="poll"|"log"), send input with submit
```

Exit with `process(action="write", session_id="<id>", data="\u0003")` (Ctrl+C) or `process(action="kill")`.

### RPC Mode (`--mode rpc`)

Process-integration protocol for embedding pi in other tools (see `docs/rpc.md`). Advanced; prefer JSON mode for Hermes automation.

## Key Flags (verified on 0.84.1)

| Flag | Effect |
|------|--------|
| `-p, --print` | Non-interactive: process prompt and exit |
| `--mode <text\|json\|rpc>` | Output mode (json = JSON Lines event stream) |
| `--provider <name>` | Provider (anthropic, openai, google, deepseek, …) |
| `--model <pattern>` | Model pattern or ID; supports `provider/id` and `:thinking` shorthand (`sonnet:high`) |
| `--thinking <level>` | `off, minimal, low, medium, high, xhigh, max` |
| `--list-models [search]` | List available models |
| `-t, --tools <list>` | **Allowlist** tool names (built-in: read, bash, edit, write, grep, find, ls) |
| `-xt, --exclude-tools <list>` | Denylist tool names |
| `-nt, --no-tools` | Disable all tools |
| `-nbt, --no-builtin-tools` | Disable built-in tools, keep extension/custom |
| `-nc, --no-context-files` | Disable AGENTS.md / CLAUDE.md discovery |
| `--skill <path>` | Load a skill (repeatable) |
| `-e, --extension <source>` | Load an extension from path/npm/git (repeatable) |
| `--system-prompt <text>` | Replace default prompt |
| `--append-system-prompt <text>` | Append to system prompt (repeatable) |
| `-c, --continue` | Continue most recent session |
| `-r, --resume` | Browse/select a session |
| `--session <path\|id>` | Use specific session file or partial UUID |
| `--fork <path\|id>` | Fork a session into a new one |
| `--no-session` | Ephemeral mode (don't save) |
| `-n, --name <name>` | Session display name |
| `@file` | Include file contents in the message |

Package management: `pi install/remove/uninstall <source> [-l]`, `pi update [--all|--models|--self]`, `pi list`, `pi config`.

## Hermes Delegation Workflow

1. Verify: `pi --version`, `pi auth` (or env API key present).
2. Isolate: fresh workdir/worktree; clean git status.
3. Scope tools: `--tools read,edit,write,bash,grep,find,ls` (or read-only set for reviews).
4. Run one-shot: `pi -p "<task>"` in that workdir; background + `process` tools for long tasks.
5. Verify independently: `git diff`, `git diff --check`, targeted tests — never trust the agent's narrative alone.
6. Report real changes and real verification output.

## Verification (smoke test)

```
terminal(command="pi -p 'Reply with exactly: PI_SMOKE_OK'", timeout=120)
```

Success: output contains `PI_SMOKE_OK`, exit code 0, no provider/model errors. For JSON mode, confirm first line is `{"type":"session",...}` and a final `agent_end` event arrives.

## Pitfalls

- **No permission system** — a bare `pi -p` can touch anything the user can. Always scope with `--tools`; containerize untrusted input. (See Security Model above.)
- **Interactive TUI requires `pty=true`** — it is a full terminal app; `-p` does NOT need pty.
- **`--mode json` is not `-p`** — print mode outputs text and exits; json mode streams events until agent_end. Combine `pi --mode json` for events, `pi -p` for plain answers.
- **Model names/catalogs drift** — run `pi update --models` when a model pattern stops resolving; `--list-models` shows what's current.
- **Context files auto-load** — pi reads AGENTS.md/CLAUDE.md by default; use `-nc` for deterministic bare runs (mirror of `claude --bare`).
- **No `--max-turns` equivalent** — cap runaway loops with Hermes `timeout` on the terminal call instead.
- **Version drift** — flags verified on 0.84.1; check `pi --help` after upgrades.

## Rules for Hermes Agents

1. Prefer `pi -p` for one-shots; JSON mode when you need structured/streaming output.
2. ALWAYS scope tools (`--tools` allowlist) — pi has no permission system.
3. Always set `workdir`; use background + `process` tools for long tasks.
4. Add `-nc` for deterministic bare runs when project context isn't wanted.
5. Verify with `git diff` + tests after delegated edits; report real output.
6. Exit interactive TUI sessions with Ctrl+C / kill, and clean up background sessions.
