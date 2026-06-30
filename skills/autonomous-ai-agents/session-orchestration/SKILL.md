---
name: session-orchestration
description: "Manage a long-running coding-agent session (omp or claude-code) for the user from Discord — spawn it, let the watcher poll and check in, forward its questions to the user and their replies back, and stop/restart on request. A workflow over deterministic commands; the heavy lifting (polling, state, DMs) is already automated."
version: 2.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [session-orchestration, coding-agent, omp, claude-code, tmux, watcher, Discord, delegate]
    related_skills: [claude-code, omp, hermes-agent]
---

# session-orchestration — managing a coding-agent session

Use this when the user wants you to **run a coding task on a real coding agent** (omp or claude-code) and supervise it for them: "spawn an omp session on hermes-agent and have it fix X", "kick off claude on the qt-bot repo with this prompt", "check on the session you started", "stop that run".

You are the supervisor, not the coder. You spawn a managed CLI session in a background tmux pane, then a deterministic watcher cron polls it, checks in, and forwards its questions to the user and their replies back. **Most of the work is already automated** — your job is to fire the right deterministic command at the right moment and relay context. Do not `tmux send-keys`, edit the registry, or poll panes yourself.

## Precondition (verify once)

Everything here is gated on config:

```yaml
# ~/.hermes/config.yaml
session_orchestration:
  enabled: true
```

When disabled, the commands below are inert and no sessions run. Two more things must be true for check-ins to actually appear:

- The **watcher cron must be running** — it is the engine that polls and checks in. Registered as a `--no-agent` job (`scripts/session-orchestration-watch.sh`, every 1 min) via `ensure_watcher_cron()` at startup. If `/control-status` shows sessions whose state never updates, the watcher cron is not running — fix that, not the spawn.
- A **feed channel must be configured** — `session_orchestration.feed_channel_id` in config. This is the unified Discord channel that shows every session needing action. If it is empty, feed posts are silently skipped and the user only sees per-session thread posts (plus the occasional DM).

## The workflow

### 1. Spawn

Two equivalent entry points; both default the agent to **omp** when unspecified.

- **Natural language (preferred for users):** the user `@`-mentions you asking to start/run/delegate a coding task. Call the `session_spawn` tool with `{repo, prompt, agent?, z_command?}`. `repo` is an alias / basename (e.g. `hermes-agent`) or absolute path; the repo registry resolves it to a workdir. `z_command` optionally prepends a z-harness slash command (e.g. `/z-plan`).
- **Deterministic command:** `/so-spawn agent=<name> workdir=<path> [z_command=<cmd>] <prompt>` (alias `/spawn-session`). Use this when you already have an absolute workdir and want a no-LLM spawn.

Spawn launches the tmux session, injects `HERMES_MARKER_FILE` (the agent writes status markers there), creates a Discord **project thread** for the conversation, and records the registry row (task_id, agent, workdir, the user, the thread). Tell the user the task_id and link the thread.

Pick the agent deliberately: **omp** for autonomous `--auto-approve` runs and z-harness workflows; **claude / claude-code** for interactive Claude Code. Default to omp if the user doesn't say.

**Repo is mandatory — ask, never guess.** A spawn cannot run without a repo. If the user's request doesn't name one (e.g. "@hermes so omp 'fix the flaky test'" with no repo), do **not** call `session_spawn` with a placeholder or a guessed repo — ask the user which repo (a name/alias the registry knows, or an absolute path) and spawn once they answer. The tool enforces this too: called with no repo, or with a repo it can't resolve, it returns a plain-language question for you to relay rather than an error — surface that question to the user and wait for their answer.

### 2. Let the watcher check in (automatic — do not poll manually)

Once spawned, the watcher cron drives everything each tick. You don't loop or scrape; you react to what it surfaces. **The primary surfaces are the unified feed channel and the session's project thread** — every state transition posts to both (once, debounced), with per-state icons (🔔 needs input, ⏸️ handoff, ✅ done, ▶ running). A **DM** is only an *extra* ping layered on top in the attention cases below; it is not the main channel.

- **State** is marker-driven and authoritative (the agent writes `status`/`heartbeat`/`needs_input`/`handoff_continue`/`done` to its marker file); pane-scraping is fallback only.
- **Liveness:** a recent `heartbeat` suppresses the hang guard even if the pane looks unchanged.
- **Needs input:** on `WAITING_USER`, the feed + thread message includes the agent's question (LLM-summarized, not a raw pane dump), and the user also gets a DM ping.
- **Handoffs:** a `handoff_continue` marker triggers an automatic `/clear`+resume with no human in the loop. A `handoff_decision` marker is NOT auto-resumed — it posts to the feed + thread (paused) and additionally DMs the decision, then waits.
- **Hang:** a confirmed stale `RUNNING` session escalates through a bounded auto-nudge ladder → posts to the feed + a user DM → marks the session `ERROR`. There is no `STALLED` registry state; stale is just an attention condition on a `RUNNING` row.
- **Cadence:** active sessions tick faster (~30s) than idle ones (~120s); no per-second polling.

So after spawning, your role is mostly to **wait for the watcher to surface something in the feed/thread** and then act on it.

### 3. Check in on demand

`/control-status` (alias `/cs`) — read-only snapshot of every active session: count per state, per-task state/agent/age, and the oldest `WAITING_USER` task highlighted. Use it when the user asks "what's running?" or before deciding whether to spawn another. It is a backstop to the feed, not a substitute for it.

### 4. Forward replies (drive)

When the user replies **in a session's project thread**, the gateway relays it to that session automatically (`_handle_managed_thread_reply` matches the thread to the registry row and drives the agent under a per-session lock). You do not need to do anything special — just keep the conversation in the thread. If the session is paused at a handoff, the reply resumes it; otherwise it is delivered as the next message.

### 5. Stop / restart

- `/so-stop task_id=<id>` — kill the session's tmux process and mark the row terminal (`DONE`). Use when the task is finished or the user wants to abandon it.
- `/so-restart task_id=<id>` — kill and re-spawn. Note: restart re-spawns with a placeholder prompt (`"continue"`) because the original prompt is not persisted, so prefer a fresh `/so-spawn` when the user wants a specific new prompt.

Both enqueue a terminate intent that the watcher applies on its next tick (single-writer discipline — the watcher is the only mutator of session state).

### Reaping (current behavior — know the limits)

Terminal sessions (`DONE` / `ERROR`) automatically drop out of the watcher's active set, so they stop being polled — no zombie processing. But reaping is **soft, not complete**:

- There is no proactive dead-tmux detection. A session whose tmux dies *without* writing a `done` marker stays `RUNNING` until the hang ladder slowly walks it to `ERROR`. If a user reports a session that "died but still shows running", `/so-stop task_id=<id>` is the deterministic fix.
- Terminal rows are not garbage-collected; they persist in the registry.
- The feed is an append-only transition log, not a self-pruning "currently needs action" view.

Treat these as known limits, not failures.

## Decision guide (when the user talks to you)

| User intent | Do this |
|-------------|---------|
| "run / fix / build X on \<repo\>" (new work) | Spawn (omp unless they name claude). Confirm task_id + thread. |
| "what's running / status?" | `/control-status` (`/cs`). |
| reply to the agent's question | Tell them to reply **in the project thread**; the relay forwards it. |
| "stop / cancel that" | `/so-stop task_id=<id>`. |
| "restart it" / "it's stuck" | `/so-restart task_id=<id>` (or fresh `/so-spawn` for a new prompt). |
| agent asked a question / needs a decision | The watcher already posted it to the feed + thread (and pinged via DM). Relay the user's answer into the thread. |
| nothing is updating | Check the watcher cron is running AND `feed_channel_id` is set (see Precondition). |

## Hard rules

- Never `tmux send-keys`, capture panes, or mutate the registry directly — the relay and watcher own all session interaction.
- Never poll in a loop yourself; the watcher cron is the polling engine.
- Default to **omp**; only use claude/claude-code when the user asks.
- Keep each session's conversation in its own project thread so the relay can route replies.
- Everything is config-gated; if `session_orchestration.enabled` is false, say so rather than improvising.

## Where the pieces live (reference)

| Piece | Location |
|-------|----------|
| Spawn (command + tool) | `session_orchestration/spawn.py` (`handle_spawn_command`), `session_orchestration/spawn_tool.py` (`session_spawn`) |
| Stop / restart | `/so-stop`, `/so-restart` → `spawn.py` `handle_stop_command` / `handle_restart_command` → `registry.enqueue_terminate` |
| Status | `/control-status` (`/cs`) → `session_orchestration/status.py` `build_snapshot` |
| Watcher (poll/check-in engine) | `session_orchestration/watcher.py`, run by `scripts/session-orchestration-watch.sh` (cron, `ensure_watcher_cron`) |
| Markers (authoritative state) | `session_orchestration/markers.py`; agent writes `HERMES_MARKER_FILE` |
| Feed / DMs / icons | `session_orchestration/feed.py`, `session_orchestration/dm_transport.py` |
| Adapters | `session_orchestration/adapters/{claude_code,omp}.py` |
| Repo name → workdir | `session_orchestration/repo_registry.py` |

z-harness, when run under a managed session, emits the markers this skill relies on (`scripts/emit-hermes-marker.sh`, gated on `HERMES_MARKER_FILE`). The only cross-repo coupling is that shared marker schema.
