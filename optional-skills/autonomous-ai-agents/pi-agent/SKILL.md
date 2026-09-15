---
name: pi-agent
description: Delegate coding to the Pi terminal coding agent CLI.
version: 1.0.0
author: Isaac (isaac-tes), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Coding-Agent, Pi, Earendil, PTY, RPC, Automation]
    related_skills: [claude-code, codex, opencode, hermes-agent]
---

# Pi Agent — Hermes Orchestration Guide

Delegate coding tasks to [Pi](https://pi.dev) (`@earendil-works/pi-coding-agent`), a minimal terminal coding harness, via the Hermes `terminal`. Pi reads files, writes code, edits, and runs bash, extended through TypeScript extensions, skills, prompt templates, and themes. Every command below was verified against the live `pi` binary (v0.85.1).

## When to Use

- Hand a coding task to Pi from Hermes and report the result back
- Run a one-shot, non-interactive coding task (fix a bug, add a feature, refactor) with structured output
- Drive a multi-turn interactive Pi session over tmux
- Integrate Pi headlessly over its stdin/stdout RPC protocol

**Don't use for:** Hermes' own coding tasks (use Hermes' built-in tools) or Claude Code / Codex / OpenCode tasks (each has its own skill). Pi is a separate agent binary with its own auth and sessions.

## Prerequisites

- **Install:** `npm install -g @earendil-works/pi-coding-agent` (binary: `pi`).
- **Auth:** one of — set a provider API key env var (e.g. `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`), or run `pi` once and use `/login` for subscription providers (Anthropic Pro/Max, ChatGPT/Codex, GitHub Copilot).
- **Auth check:** `terminal(command="pi auth check --provider anthropic --json")` → `{"status":"ready",...}` or `{"status":"not_ready","reason":"credentials_not_configured"}`. Add `--credentials` to also emit the key.
- **Version check:** `terminal(command="pi --version")`
- **Model list:** `terminal(command="pi --list-models [search]")`
- **No `doctor`** — Pi has no health-check subcommand. Use `pi auth check --provider <p> --json` as the readiness probe.

Pi's default tools are `read`, `write`, `edit`, `bash`. `grep`, `find`, `ls` exist but are **off by default**.

## ⚠️ Security Model

Pi has **no built-in permission system** — no approval dialogs, no sandbox. It runs with the permissions of the user/process that launched it, so every delegated run can touch anything that user can. Hermes must enforce boundaries itself:

1. **Scope tools with `--tools`** for every delegated run: `--tools read,grep,find,ls` (read-only review) or `--tools read,edit,write,bash,grep,find,ls` (full).
2. **`--no-tools` / `-nt`** for analysis-only runs.
3. **Isolate the workdir** (`workdir=` + a clean git state; `git diff` review after — process boundaries are the safety layer).
4. **Containerize untrusted work** — Pi's docs cover a Gondolin micro-VM extension, plain Docker, and OpenShell policy sandboxing.

Note the distinction: the interactive *trust dialog* (below) only governs loading project-local resources — it does not restrict what Pi's tools may do.

## Three Orchestration Modes

Pi has four run modes (interactive, print/JSON, RPC, SDK). The first three matter for Hermes.

### Mode 1: Print mode (`-p`) — Non-Interactive (PREFERRED for one-shot)

`-p` runs a one-shot task, prints the result as plain text, and exits. No PTY, no dialogs.

```
terminal(command="pi -p 'Add error handling to all API calls in src/'", workdir="/path/to/project", timeout=120)
```

**Structured output** — add `--mode json` for a JSON-lines event stream (see Deep Dive):

```
terminal(command="pi -p --mode json 'Analyze auth.py for bugs'", workdir="/project", timeout=120)
```

**Read-only mode** — restrict tools for review tasks:
```
terminal(command="pi --tools read,grep,find,ls -p 'Review the code in src/'", workdir="/project", timeout=90)
```

**Pipe input:** `cat src/auth.py | pi -p 'Find the bugs'`.

**File args:** prefix `@` — `pi -p @screenshot.png 'What's in this image?'`.

**When to use:** one-shot tasks, CI/scripting, any task without multi-turn conversation.

### Mode 2: RPC (`--mode rpc`) — Headless process integration

Strict LF-delimited JSONL over stdin/stdout. Send one JSON command per line; receive `response` objects plus streamed agent events.

```python
import subprocess, json
p = subprocess.Popen(["pi", "--mode", "rpc"], cwd="/project",
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
p.stdin.write(json.dumps({"id": "req-1", "type": "prompt",
                          "message": "Add a unit test for auth"}) + "\n")
p.stdin.flush()
for line in p.stdout:            # iterate lines — split on \n ONLY
    obj = json.loads(line)
    if obj.get("type") == "response" and obj.get("id") == "req-1":
        print("accepted:", obj["success"])
```

Key RPC commands: `prompt`, `steer` (queue mid-stream), `follow_up`, `abort`, `clear_queue`, `new_session`, `get_state`, `get_messages`, `set_model`, `cycle_model`. A response looks like `{"type":"response","command":"prompt","success":true}`. `success:true` on `prompt` means *accepted* — failures after acceptance arrive as events, not a second response.

**Framing rule:** split records on `\n` only, strip a trailing `\r`. Do **not** use Node `readline` — it also splits on U+2028/U+2029, which are legal inside JSON strings.

**When to use:** custom UIs, IDEs, or when you need steering mid-run from a program.

### Mode 3: Interactive PTY via tmux — Multi-Turn Sessions

```
# Start a tmux session
terminal(command="tmux new-session -d -s pi-work -x 140 -y 40")

# Launch Pi inside it
terminal(command="tmux send-keys -t pi-work 'cd /path/to/project && pi' Enter")

# Handle the first-launch trust dialog (see below)
terminal(command="sleep 8 && tmux capture-pane -t pi-work -p -S -40")
terminal(command="tmux send-keys -t pi-work Enter")   # 'Trust' is the default selection

# Wait for startup, then send your task
terminal(command="sleep 3 && tmux send-keys -t pi-work 'Refactor the auth module to use JWT' Enter")

# Monitor
terminal(command="sleep 15 && tmux capture-pane -t pi-work -p -S -50")

# Exit
terminal(command="tmux send-keys -t pi-work '/quit' Enter")
terminal(command="tmux kill-session -t pi-work")
```

**When to use:** multi-turn iterative work, human-in-the-loop decisions, when you need Pi's slash commands (`/model`, `/compact`, `/tree`).

## Interactive First-Launch Trust Dialog (CRITICAL)

On first launch **in a project folder that has project-local settings/resources** (a `.pi/` dir, project extensions, or `.agents/skills`), Pi shows:

```
 Trust project folder?
 /path/to/project
 → Trust            ← DEFAULT (just press Enter)
   Trust parent folder
   Trust (this session only)
   Do not trust
   Do not trust (this session only)
```

**Handle:** `tmux send-keys -t <session> Enter` — the default `Trust` is the right choice. The decision is saved to `~/.pi/agent/trust.json` and won't reappear for that folder.

**Non-interactive modes (`-p`, `--mode json`, `--mode rpc`) never show this dialog.** Without a saved trust decision they default to *ignoring* project-local resources. To load a project's `.pi/` resources in a headless run, pass `--approve` / `-a` (trust for one run) or `--no-approve` / `-na` (force ignore). There is **no** Claude-Code-style "bypass permissions" second dialog.

## Print/JSON Output Shape (verified)

`pi -p` (no `--mode`) prints plain text and exits — no JSON, no `session_id`/`cost` envelope. `pi -p --mode json` emits one JSON object per line, LF-delimited:

1. Header: `{"type":"session","version":3,"id":"<uuid>","timestamp":"...","cwd":"/path"}` — use `id` to resume.
2. Lifecycle events: `agent_start`, `turn_start`, `message_start`, `message_update` (delta-only), `message_end`, `turn_end`, `agent_end`, `agent_settled`.
3. The final assistant `message_end` carries the authoritative `message`: `stopReason` (e.g. `stop`), `usage` (`input`, `output`, `totalTokens`, `cost.total`), and `content` (array of `{type:"text"|"thinking"|"tool"}` blocks).

Unlike Claude Code, **there is no single final result object.** To get the answer, filter for the last `message_end` with `role:assistant` and read its `content` text. To get cost, read `usage.cost.total` on that same message (often `0` for free/local providers).

```bash
pi -p --mode json 'task' 2>/dev/null | jq -c 'select(.type=="message_end" and .message.role=="assistant")'
```

## CLI Quick Reference (verified from `pi --help`)

| Flag / command | Purpose |
| --- | --- |
| `pi` | Interactive REPL |
| `pi "query"` | REPL with initial prompt; `@file` args attach files |
| `pi -p "query"` | Non-interactive, run and exit (plain text) |
| `pi -p --mode json "q"` | Non-interactive, JSON event stream |
| `pi --mode rpc` | Headless JSONL over stdin/stdout |
| `pi --provider <name>` | Provider (default `google`) |
| `pi --model <pattern>` | Model, e.g. `openai/gpt-4o` or `sonnet:high` |
| `pi --thinking <lvl>` | `off, minimal, low, medium, high, xhigh, max` |
| `pi --tools, -t <names>` | Allowlist tools (e.g. `read,grep,find,ls`) |
| `pi --exclude-tools, -xt <names>` | Denylist tools |
| `pi --no-tools, -nt` | Disable all tools |
| `pi -c, --continue` | Continue most recent session in this dir |
| `pi -r, --resume` | Pick a past session to resume |
| `pi --session <path\|id>` | Use a specific session file/partial UUID |
| `pi --fork <path\|id>` | Fork a session into a new one |
| `pi --session-id <id>` | Use exact session ID (create if missing) |
| `pi --no-session` | Ephemeral — don't save the session |
| `pi --name, -n <name>` | Set session display name |
| `pi --export <file>` | Export session to HTML and exit |
| `pi --approve, -a` | Trust project-local files for this run |
| `pi --no-approve, -na` | Ignore project-local files for this run |
| `pi --list-models [search]` | List available models |
| `pi auth check --provider <p> [--json]` | Check provider readiness |
| `pi auth print-api-key / print-bearer-token` | Emit a credential for external clients |
| `pi install / remove / update / list` | Manage extensions & packages |
| `pi config` | TUI to enable/disable package resources |

## Interactive Slash Commands

`/login` `/logout`, `/model`, `/thinking`, `/scoped-models`, `/settings`, `/resume`, `/new`, `/name <n>`, `/session`, `/tree` (jump to any point and branch), `/fork`, `/clone`, `/compact [prompt]`, `/copy`, `/export`, `/import`, `/share`, `/reload`, `/hotkeys`, `/changelog`, `/quit`. Skills surface as `/skill:name`; prompt templates as `/template`.

## Monitoring the TUI

```
terminal(command="tmux capture-pane -t pi-work -p -S -20")
```

Read the **footer** (bottom lines), top to bottom: working directory, then `↑<in> ↓<out> <ctx>%/window (auto)` (token/cache usage), then `(provider) model • thinking-level`. A streaming border around the editor = Pi is working; when the editor is empty and waiting, Pi is done or asking. There is no Claude-Code-style `❯`/`●` marker — use the footer token counts and message text to judge progress.

## Pitfalls & Gotchas

- **No permission system — scope every run with `--tools`.** A bare `pi -p` can do anything the launching user can; the trust dialog does not change this. Read-only set (`read,grep,find,ls`) for reviews, full set only when the task needs writes/bash.
- **`-p` is plain text, not JSON.** You must add `--mode json` for structured output; don't parse `-p` output as JSON.
- **No final result blob.** Unlike Claude Code's single result object, Pi streams events — filter `message_end` with `role:assistant` to get the answer and cost.
- **No `--max-turns` / `--max-budget-usd`.** Pi has no built-in turn or spend cap, so a runaway loop isn't auto-stopped — set a generous `timeout=` and/or use `--no-session` in CI and `kill` the process if it hangs.
- **Trust dialog is interactive-only and once-per-folder.** Non-interactive runs silently ignore project `.pi/` resources unless you pass `--approve`; a fresh headless run in a project with extensions won't load them otherwise.
- **RPC: split on `\n` only.** Node `readline` breaks the protocol (splits on U+2028/U+2029 inside JSON strings).
- **The bash tool can't write outside the working dir.** Observed: Pi refuses to write to a path outside `cwd` and falls back to writing inside it. Set `workdir` to the project root so the target path is in-bounds.
- **`grep`, `find`, `ls` are off by default.** Only `read`, `write`, `edit`, `bash` are enabled; enable the others explicitly with `--tools` when the task needs them.
- **tmux extended-keys.** If `set -g extended-keys on` is missing in `~/.tmux.conf`, modified Enter keys (the `Alt+Enter` follow-up shortcut) won't register. Plain Enter is unaffected.
- **Sessions auto-save to `~/.pi/agent/sessions/`** keyed by working directory; use `--no-session` for ephemeral CI runs to avoid accumulating files.

## Rules for Hermes Agents

1. **Prefer `pi -p` for single tasks** — clean, no dialog handling; add `--mode json` when you need the structured event stream.
2. **Use tmux for multi-turn interactive work** — the only reliable way to orchestrate the TUI.
3. **Always set `workdir`** — Pi's bash tool can't write outside it, and sessions are keyed to it.
4. **Set a generous `timeout=`** — Pi has no turn cap; monitor and kill a hung run.
5. **In interactive mode, send `Enter` for the trust dialog** on first launch in a project folder.
6. **Monitor with `tmux capture-pane -t <session> -p -S -50`** and read the footer token counts.
7. **Clean up tmux sessions** with `tmux kill-session -t <name>` when done.
8. **Report results to the user** after completion.
9. **Verify independently** — after delegated edits, check `git diff` and run targeted tests yourself; never trust the agent's narrative alone.
10. **Use `-nc` for deterministic bare runs** when project AGENTS.md/CLAUDE.md context isn't wanted (Pi's mirror of `claude --bare`).

## Verification

Confirm the skill works end-to-end:

```
terminal(command="pi --version")                      # expect a version number
terminal(command="pi auth check --provider anthropic --json")   # expect status: ready
terminal(command="pi -p 'Write hello.py that prints hi and run it'", workdir="/tmp/pi-test", timeout=120)
```

Success criteria: `pi --version` prints a version; `auth check` returns `ready`; the `-p` run actually creates and runs `hello.py` in the working dir (verify with `read_file`) and prints the result. For JSON mode, confirm the first line is `{"type":"session",...}` and a final `message_end` with `role:assistant` exists.
