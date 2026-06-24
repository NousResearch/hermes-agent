---
name: session-orchestration
description: "Manage long-running claude-code and omp agent sessions — spawn, drive, watch, and feed."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [session-orchestration, tmux, registry, watcher, feed, claude-code, omp, Discord]
    related_skills: [claude-code, omp, hermes-agent]
---

# session-orchestration — Architecture and Usage Guide

The `session_orchestration/` package teaches Hermes to manage external coding-agent CLI sessions (claude-code and omp in v1) on the user's behalf. The user `@`s Hermes in Discord with a prompt, target agent, and optional z-command; Hermes spawns a tmux session, drives it, and reports back through a unified feed channel.

## Config Gate

All session-orchestration behavior is gated behind:

```yaml
# ~/.hermes/config.yaml
session_orchestration:
  enabled: true
```

When `enabled` is absent or falsy: byte-identical to pre-feature behavior — no new tmux sessions, no registry writes, no network calls, no feed pushes.

## Core Concepts

### Registry (SQLite, single-writer)

The registry is a `session_orchestration` table in Hermes's `state.db` (WAL mode). It tracks every managed session:

| Column | Purpose |
|--------|---------|
| `task_id` | Primary key (UUID from the adapter's `SessionHandle`) |
| `agent` | `"claude"` or `"omp"` |
| `tmux_session` | tmux session name (e.g. `hermes-cc-a1b2c3d4`) |
| `project` | Human label for the project |
| `repo` | Canonical 12-char hex key — SHA-256 of the normalized git remote URL (stable across machines for the same repo) |
| `run_id` | External run correlation key (from z-harness webhook) |
| `state` | Current `SessionLifecycle` value |
| `discord_thread_id` | Discord thread for this task's conversation |
| `lock_holder` | Current lock holder (`"relay"` / `"watcher"`) or NULL |
| `lock_ts` | Float expiry epoch (`str(time.time() + ttl)`) |
| `heartbeat_counter` | Incremented atomically on each watcher tick |
| `idle_ticks` | Consecutive ticks with unchanged pane hash |

**UNIQUE(run_id, repo)** — prevents duplicate rows for the same external run + repository pair.

**Single-writer discipline:** The **cron watcher is the sole mutator** of registry rows and derived counters. All other paths (webhook-adopt, Discord-drive) MUST enqueue intents to the `session_orchestration_queue` table; the cron watcher drains and applies them in a single serialized transaction. This eliminates lost-update races without requiring optimistic locking.

**Counter writes are atomic:** `SET col = col + 1` (not read-modify-write in Python), so a mid-increment crash leaves no stale value.

**Lock TTL:** `lock_ts` stores the float expiry epoch. `acquire_lock` refuses if a non-expired row exists. Default TTL = 300 s (5× cron interval). Stale lock reclaim uses DB wallclock comparison (`time.time()` vs stored epoch) — avoids `datetime.fromisoformat` timezone-parsing ambiguity in Python 3.11.

### Session Lifecycle

```
          spawn / adopt
               │
           RUNNING   ◄──────────────────────┐
            │   │                            │
    ● active  ❯ prompt             drive / resume
            │   │                            │
       WAITING_USER ──── HERMES_HANDOFF ──► PAUSED_HANDOFF
            │                            │
      (watcher: idle_ticks > N)          │
            │                            │
         STALLED                      (relay: /clear + re-inject)
            │
      (watcher: state==RUNNING,
       pane hash unchanged, no ●)
            │
          hang → nudge → escalate
            │
          DONE  /  ERROR
```

States returned by `AgentAdapter.detect()`:

| State | Meaning |
|-------|---------|
| `RUNNING` | Agent is active (● indicator or no clear signal) |
| `WAITING_USER` | Agent is at the `❯` prompt, waiting for a message |
| `PAUSED_HANDOFF` | Agent emitted `HERMES_HANDOFF` — ready for `/clear` + next prompt |
| `STALLED` | Watcher-assigned: pane hash unchanged for N ticks (not a `detect()` return) |
| `DONE` | Session exited cleanly |
| `ERROR` | pane unreachable / session dead |

`STALLED` and `DONE` are watcher-assigned from pane-hash staleness; `detect()` only returns the first four.

### AgentAdapter ABC

All adapters implement:

```python
class AgentAdapter:
    def capabilities(self) -> Capabilities: ...
    def launch(self, workdir: str, prompt: str) -> SessionHandle: ...
    def drive(self, handle: SessionHandle, message: str) -> None: ...
    def detect(self, handle: SessionHandle) -> SessionLifecycle: ...
    def resume(self, handle: SessionHandle, prompt: str) -> None: ...
```

`SessionHandle(session_id, tmux_session, pane, launch_ts)` is returned by `launch()` and stored in the registry row.

Capabilities are asserted at watcher startup via `verify_adapters()` — a mismatch hard-fails that adapter with a logged error without crashing the watcher.

## When to Spawn vs Drive

**Spawn** — create a new managed session:
- The user is starting a **new task** in a project (new feature, new bug fix, new exploration).
- No existing managed session is active for this project + agent combination.
- Via Discord: `/so-spawn agent:<claude|omp> prompt:"…" [workdir:"…"]`
- Via code: `session_orchestration.spawn.handle_spawn_command(...)`
- Spawn creates the initial registry row and seeds the first prompt via `SessionRelay.send_message`.

**Drive** — send a follow-up message to an existing managed session:
- The user **replies in the project's Discord thread** — the gateway's `_handle_managed_thread_reply` intercepts it.
- The session is already registered (`discord_thread_id` maps to a registry row).
- Drive acquires the per-session lock → detects `PAUSED_HANDOFF` (if so, calls `resume()`) → otherwise calls `adapter.drive()` → releases lock.
- Do NOT call `tmux send-keys` directly — the relay owns all tmux interaction.

**Check `/control-status`** to see all active sessions before deciding to spawn.

## Watcher (Cron, Three Layers)

The watcher runs as `hermes --no-agent session_orchestration_watch` at a 1–2 minute cadence (registered in `session_orchestration/cron_registration.py`). It is the authoritative state writer.

### Layer 1: Turn-Change (Push Once)

When a session transitions into a user-attention state (`WAITING_USER` or `PAUSED_HANDOFF`):
- Push ONCE to the unified **feed channel** (one message per transition, debounced).
- Push to the **task's project thread** (so the user sees it in context).
- Non-transition ticks produce no push.

### Layer 2: Heartbeat (Edit In-Place, ~5 min)

Every ~5 ticks (configurable), the watcher **edits** the existing per-task status message in the feed channel (no new notification). Shows: current state, elapsed time since last change, active agent, pane snippet. This is a positive-liveness signal, not a notification.

### Layer 3: Hang Detection (Nudge Once, Then Escalate)

Hang is declared ONLY when ALL of:
- `state == RUNNING` (never fires on `WAITING_USER` or `PAUSED_HANDOFF`)
- Pane hash unchanged for N consecutive ticks
- No active-tool indicator (`●` / spinner) visible
- `last_output_ts` older than the static threshold

On hang: notify in feed + exactly one auto-nudge. On the next tick if still hung: escalate to user (no second nudge, no auto-kill).

**`--hook` / webhook accelerant** acts only as a **positive-liveness reset** of the heartbeat counter when it carries fresh activity (with its own freshness TTL). It cannot gate or suppress hang detection. If the accelerant never fires, cron-only detection still eventually marks hang — graceful degradation guaranteed.

## Relay (Per-Session Lock)

`SessionRelay.send_message(task_id, handle, message)` is the ONLY safe path for driving a managed session. Sequence:

1. `registry.acquire_lock(task_id, holder="relay", ttl_seconds=300.0)` — BEGIN IMMEDIATE; returns `False` if lock held
2. `adapter.detect()` — if `PAUSED_HANDOFF`, call `adapter.resume()` and return
3. `adapter.drive(handle, message)` — load-buffer / paste-buffer sequence
4. `registry.release_lock(task_id, holder)` in `finally` — crash self-heals after TTL

The watcher acquires the SAME lock around `capture-pane`. If the lock is held by the relay, the watcher **skips this row for this tick** — no interleaving, no race between send and capture.

## Feed Channel

A single unified Discord channel receives EVERY state transition + watchdog alert across all managed tasks:
- Each feed message is deep-linked to its project thread.
- Status messages are **edited in-place** (heartbeat layer) — no notification flood.
- Per-project threads carry only the conversation (replies to Claude/omp).

## Webhook Ingest (z-harness → Hermes)

When z-harness has `notify.hermes_webhook_url` configured, it POSTs platform-neutral payloads signed with HMAC-sha256. Hermes's `session_orchestration/ingest.py`:
1. Validates HMAC (`X-Z-Harness-Signature: sha256=<hex>`)
2. Rate-limits per source (60 s window)
3. Deduplicates by `event_id` (persistent `session_orchestration_event_dedup` table, survives restart)
4. Correlates `(run_id, canonical_repo_id)` to an existing registry row → enqueues an `update` intent
5. If no match: adopts to an `external_runs_thread_id` channel

## `/control-status` Command

Lists all active sessions on demand:
- Count per state (`RUNNING`, `WAITING_USER`, etc.)
- Per-task: state, agent, elapsed, project
- Oldest `WAITING_USER` task highlighted (backstop — the user may have missed the feed push)

Aliases: `/so-status`, `/cs`.

## Two-Repo Split

The session-orchestration subsystem spans two repositories:

| Component | Repo |
|-----------|------|
| `session_orchestration/` package | `hermes-agent` |
| Discord gateway wiring | `hermes-agent/gateway/` |
| cron registration | `hermes-agent/scripts/session-orchestration-watch.sh` |
| Webhook ingest | `hermes-agent/session_orchestration/ingest.py` |
| Opt-in webhook sender | `z-harness/scripts/notify-watchdog.sh` |
| Opt-in config keys | `z-harness` (`notify.hermes_webhook_url`, `notify.hermes_webhook_secret`) |

z-harness contains NO Discord-specific code. The coupling is one-directional: z-harness → Hermes webhook endpoint.

## Adapter Quick Reference

| Adapter | Module | Drive method | Dialog handling | Detect signals |
|---------|--------|-------------|-----------------|----------------|
| `ClaudeCodeAdapter` | `adapters/claude_code.py` | load-buffer + paste-buffer | Trust dialog (Enter) + bypass-permissions (Down+Enter) | `❯` → WAITING_USER; `●` → RUNNING; `HERMES_HANDOFF` → PAUSED_HANDOFF |
| `OmpAdapter` | `adapters/omp.py` | load-buffer + paste-buffer | None (`--auto-approve`) | `>` or `❯` → WAITING_USER; spinner/`Running tool` → RUNNING; `HERMES_HANDOFF` → PAUSED_HANDOFF |
