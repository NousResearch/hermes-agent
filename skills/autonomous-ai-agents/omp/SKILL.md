---
name: omp
description: "Delegate coding to oh-my-pi (omp) — structured RPC/JSON transport + tmux TUI mode."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [Coding-Agent, omp, oh-my-pi, JSON, RPC, Automation, session-orchestration]
    related_skills: [claude-code, session-orchestration, hermes-agent]
---

# omp (oh-my-pi) — Hermes Orchestration Guide

[oh-my-pi (omp)](https://github.com/oh-my-pi/omp) v16+ is an autonomous coding agent CLI with a structured transport layer that makes it the cleaner orchestration target when compared to fully TUI-based agents. `omp` supports:

- **`--mode=json`** — one-shot NDJSON streaming output (primary structured transport)
- **`--mode=rpc`** — interactive RPC-over-stdio session (for persistent interactive use)
- **`--hook`** — lifecycle callbacks for liveness accelerant
- **`--auto-approve`** — skips all tool-approval dialogs (no dialog dance required)
- **`-p` / `--print`** — non-interactive one-shot mode
- **`-c` / `--continue`** — resume the most recent session
- **`-r <id>` / `--resume <id>`** — resume a specific session by ID
- **`--max-time <seconds>`** — cap execution wall time

> **Managed mode (preferred):** When `session_orchestration.enabled` is on, use `/so-spawn agent:omp prompt:"…"` — the `OmpAdapter` handles launch, liveness tracking, and feed routing automatically. See the `session-orchestration` skill.

## Probed Transport (omp v16.1.15)

The following was verified live against omp v16.1.15.

### One-Shot Mode: `omp -p --mode=json`

This is the **primary structured transport** used by `OmpAdapter`. It runs omp as a subprocess and emits NDJSON to stdout — no tmux session required.

```bash
omp -p --mode=json [--model <model>] [--cwd <workdir>] [--max-time <seconds>] [--no-session] "<prompt>"
```

Each line of stdout is a JSON object with a `"type"` discriminator:

```jsonc
// Session start
{"type":"session","version":3,"id":"<uuid>","timestamp":"<ISO8601>","cwd":"<path>"}

// Agent lifecycle
{"type":"agent_start"}
{"type":"turn_start"}

// User message
{"type":"message_start","message":{"role":"user","content":[{"type":"text","text":"…"}], …}}
{"type":"message_end","message":{"role":"user","content":[…], …}}

// Assistant message (streamed in chunks)
{"type":"message_start","message":{"role":"assistant","content":[{"type":"text","text":"…"}], …}}
{"type":"message_update","assistantMessageEvent":{…},"message":{…}}
{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"<final answer>"}],
    "model":"<model>","usage":{…},"stopReason":"stop","timestamp":<ms>,…}}

// Turn and agent end
{"type":"turn_end","message":{…}}
{"type":"agent_end","messages":[<all messages in conversation order>]}
```

**Extracting the final answer (two strategies):**

1. **Preferred** — scan for `agent_end`, walk `messages` in reverse, find last entry with `role == "assistant"`, return `content[0]["text"]`.
2. **Fallback** — stream for `message_end` events where `message.role == "assistant"` and keep the last one's `content[0]["text"]`.

> **`--mode=rpc` is NOT one-shot.** When launched without `-p`, `--mode=rpc` emits `{"type":"ready"}` followed by `{"type":"available_commands_update","commands":[…]}` and then waits on stdin for JSON commands. It is an interactive RPC session, not a subprocess-call transport. Use `--mode=json -p` for one-shot structured output.

### Interactive TUI Mode (tmux)

For sessions that require persistent state, multi-turn conversation, or interactive slash commands, `OmpAdapter` uses tmux with `--auto-approve`:

```bash
omp --auto-approve [--cwd <workdir>] [--model <model>] [--max-time <seconds>] [--hook <hookfile>] "<prompt>"
```

`--auto-approve` suppresses all tool-approval dialogs. Unlike `claude-code`, there is **no dialog dance** required — omp starts and is immediately ready.

**Driving an interactive omp session (adapter method):**

The adapter uses `load-buffer` / `paste-buffer` (NOT `send-keys`) to deliver prompts — same rationale as claude-code: avoids metacharacter expansion, preserves multi-line content.

```
# What the adapter does internally:
tmux load-buffer -b hermes-omp-<uid> -   # (content piped on stdin)
tmux paste-buffer -d -b hermes-omp-<uid> -t <pane>
tmux send-keys -t <pane> Enter
```

**Prompt readiness detection in TUI mode:**

omp shows `>` or `❯` at the bottom of the pane when waiting for user input. The adapter polls for either pattern before injecting.

**Activity detection (pane-hash stale guard):**

omp shows braille spinner characters (`⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏`) and the text `Running tool` during active tool use. The watcher uses this to distinguish genuine work from a stalled session.

## Resume Patterns

### One-shot continuation
```bash
omp -p --mode=json -c "<prompt>"          # Continue most recent session
omp -p --mode=json -r <session-id> "<prompt>"  # Resume specific session by ID
```

The session ID is in the `session` event emitted at the start of each `--mode=json` run (`"id"` field).

### Interactive TUI continuation
```bash
omp --auto-approve -c "<prompt>"           # Continue most recent
omp --auto-approve --resume <session-id> "<prompt>"  # Resume specific
```

`OmpAdapter.resume()` uses `-c` (continue last) since the `SessionHandle` in v1 does not carry the omp session ID. The watcher triggers `resume()` on `PAUSED_HANDOFF` detection.

## `--hook` Liveness Accelerant

omp supports `--hook <hookfile>` for lifecycle callbacks. The watcher can use this as a **positive-liveness accelerant**: a hook that fires on tool-use completion can reset the per-session heartbeat counter, reducing false-positive hang alerts during long builds.

```bash
omp --auto-approve --hook /path/to/hermes-hook.sh "<prompt>"
```

The hook is optional; if it never fires, cron-only detection still eventually marks hang after the static threshold. The hook can only act as a **positive-liveness reset** — it cannot suppress detection of a real hang.

## Handoff Checkpoints (Managed Sessions)

omp should emit the exact string:

```
HERMES_HANDOFF
```

on a line by itself when it reaches a natural stopping point. This is the sentinel scanned by `OmpAdapter.detect()` (via `HANDOFF_MARKER = "HERMES_HANDOFF"` in `session_orchestration/adapters/omp.py`). On detection, `detect()` returns `PAUSED_HANDOFF` and the watcher/relay triggers `resume()` — which re-launches omp with `-c` and the next prompt.

**Instruction to omp agent** (include in your initial prompt):

> When you have completed a phase of work and are ready for the next instruction, output the literal string `HERMES_HANDOFF` on a line by itself. This signals Hermes that you are at a handoff checkpoint.

## State Detection Summary

| Pane signal | `detect()` returns |
|---|---|
| `HERMES_HANDOFF` anywhere in last 60 lines | `PAUSED_HANDOFF` |
| `>` or `❯` at end of a line | `WAITING_USER` |
| Braille spinner or `Running tool` | `RUNNING` |
| Pane cannot be captured (session dead) | `ERROR` |
| Otherwise | `RUNNING` (watcher tracks staleness) |

`STALLED` and `DONE` are inferred by the watcher from pane-hash staleness + elapsed time, not by `detect()`.

## One-Shot Usage (No tmux)

For simple single-turn queries that don't need persistent session state:

```python
from session_orchestration.adapters.omp import OmpAdapter

adapter = OmpAdapter()
result = adapter.run_oneshot(
    prompt="Explain the architecture of src/",
    model="openai-codex/gpt-5.5",   # optional
    workdir="/path/to/project",       # optional
    max_time=120,                     # optional, seconds
)
print(result)  # final assistant text
```

`run_oneshot` calls `omp -p --mode=json --no-session <prompt>` and parses the NDJSON result. No tmux session is created.

## CLI Quick Reference

| Flag | Effect |
|------|--------|
| `-p` / `--print` | Non-interactive one-shot mode (exits when done) |
| `--mode=json` | NDJSON streaming output (use with `-p`) |
| `--mode=rpc` | Interactive RPC-over-stdio (NOT one-shot; requires persistent process) |
| `--auto-approve` | Skip all tool-approval dialogs (essential for unattended sessions) |
| `-c` / `--continue` | Continue most recent session |
| `-r <id>` / `--resume <id>` | Resume specific session by ID |
| `--hook <file>` | Lifecycle hook script for liveness accelerant |
| `--cwd <path>` | Set working directory |
| `--model <model>` | Override model (e.g. `openai-codex/gpt-5.5`) |
| `--max-time <seconds>` | Cap wall-clock execution time |
| `--no-session` | Don't persist the session to disk (useful for one-shots) |

## Rules for Hermes Agents

1. **Use `/so-spawn agent:omp` for managed multi-turn sessions** — adapter handles launch, liveness, and routing
2. **Prefer `run_oneshot()` for single queries** — no tmux needed, structured output, simpler error handling
3. **`--mode=json -p` is the one-shot structured transport** — `--mode=rpc` is interactive only
4. **No dialog handling needed** — `--auto-approve` suppresses all approval prompts
5. **Use `--hook` for long builds** — reduces false-positive hang alerts during extended tool use
6. **Monitor via the feed channel** — in managed mode, state transitions are pushed to the unified feed
7. **Emit `HERMES_HANDOFF` at checkpoints** — the adapter scans for this exact string to detect handoff state
