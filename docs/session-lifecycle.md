# Session Lifecycle

> **Audience:** Gateway developers and maintainers
> **Source files:** `gateway/session.py` (~1444 lines), `gateway/run.py` (~16800 lines), `gateway/config.py`
> **Last updated:** 2026-06-16

## Overview

A **session** represents a continuous conversation between the agent and one or more users on a
messaging platform. The session lifecycle governs when conversations persist, when they reset,
how they survive gateway restarts, and how messages queue during concurrent operations.

The session system lives primarily in two modules:

- `gateway/session.py` — Data model (`SessionSource`, `SessionEntry`, `SessionContext`),
  key generation (`build_session_key`), and the main store (`SessionStore`).
- `gateway/run.py` — Gateway runner (`GatewayRunner`) that wires sessions into the message
  processing pipeline: session expiry watching, agent caching, restart recovery, and message
  queuing.

---

## 1. SessionSource — Message Origin Descriptor

`SessionSource` is a frozen record of *where a message came from*. It is attached to every
incoming `MessageEvent` and used for routing, isolation, and context injection.

### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `platform` | `Platform` | *(required)* | Enum identifying the messaging platform (telegram, discord, slack, signal, whatsapp, matrix, local, etc.). |
| `chat_id` | `str` | *(required)* | Platform-level chat/group/channel identifier. Routed through the adapter's `chat_id_key` transform. |
| `chat_name` | `Optional[str]` | `None` | Human-readable name of the chat or group. |
| `chat_type` | `str` | `"dm"` | One of `"dm"`, `"group"`, `"channel"`, `"thread"`. Controls session key generation and isolation. |
| `user_id` | `Optional[str]` | `None` | Platform-specific user identifier. Used for authorization and per-user session isolation. |
| `user_name` | `Optional[str]` | `None` | Display name of the message author. Injected into system prompt. |
| `thread_id` | `Optional[str]` | `None` | Forum topic / Discord thread / Slack thread identifier. Differentiates threaded conversations. |
| `chat_topic` | `Optional[str]` | `None` | Channel topic or description (Discord channel topic, Slack channel purpose). |
| `user_id_alt` | `Optional[str]` | `None` | Platform-specific stable alternative ID (Signal UUID, Feishu union_id). Used when `user_id` is ephemeral. |
| `chat_id_alt` | `Optional[str]` | `None` | Signal group internal ID — maps a Signal group V2 identifier to its canonical form. |
| `is_bot` | `bool` | `False` | True when the message author is a bot or webhook (Discord bots). |
| `guild_id` | `Optional[str]` | `None` | Discord guild / Slack workspace / Matrix server scope identifier. |
| `parent_chat_id` | `Optional[str]` | `None` | Parent channel when `chat_id` refers to a thread. |
| `message_id` | `Optional[str]` | `None` | ID of the triggering message. Used for pin/reply/react operations and Discord ID injection. |
| `role_authorized` | `bool` | `False` | True when adapter granted access via a platform role (not individual user ID). |

### Key Methods

- **`description`** (property: `str`) — Human-readable summary e.g. `"DM with Alice"`,
  `"group: My Group, thread: 12345"`.
- **`to_dict()` / `from_dict()`** — Serialization round-trip for persistence in `sessions.json`.

---

## 2. SessionEntry — Active Session Record

`SessionEntry` is the per-session metadata record stored in memory and persisted to
`{sessions_dir}/sessions.json`. Each entry maps a `session_key` to its current `session_id`.

### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `session_key` | `str` | *(required)* | Deterministic key identifying the conversation lane (see §4). |
| `session_id` | `str` | *(required)* | Unique identifier for this specific conversation incarnation. Format: `YYYYMMDD_HHMMSS_<8hex>`. |
| `created_at` | `datetime` | *(required)* | When this session incarnation was created. |
| `updated_at` | `datetime` | *(required)* | Last activity timestamp. Used for idle timeout and expiry checks. |
| `origin` | `Optional[SessionSource]` | `None` | The source that created this session, used for delivery routing. |
| `display_name` | `Optional[str]` | `None` | Chat display name (sourced from `SessionSource.chat_name`). |
| `platform` | `Optional[Platform]` | `None` | Platform enum, persisted for expiry policy lookup across restarts. |
| `chat_type` | `str` | `"dm"` | Chat type, also persisted for policy lookup. |
| `input_tokens` | `int` | `0` | Cumulative LLM input (prompt) tokens consumed. |
| `output_tokens` | `int` | `0` | Cumulative LLM output (completion) tokens consumed. |
| `cache_read_tokens` | `int` | `0` | Cumulative prompt cache read tokens. |
| `cache_write_tokens` | `int` | `0` | Cumulative prompt cache write tokens. |
| `total_tokens` | `int` | `0` | Total token count across all turns. |
| `estimated_cost_usd` | `float` | `0.0` | Estimated cumulative USD cost. |
| `cost_status` | `str` | `"unknown"` | Cost tracking status label. |
| `last_prompt_tokens` | `int` | `0` | Last API-reported prompt token count. Used for accurate compression pre-check. |

### Boolean Flags (State Machine)

SessionEntry has several boolean flags that form a simple state machine governing session
behavior on the next access.

| Flag | Type | Default | Description |
|---|---|---|---|
| `was_auto_reset` | `bool` | `False` | Set when a session was auto-reset due to policy expiry (idle/daily). Consumed once to inject a context notice. |
| `auto_reset_reason` | `Optional[str]` | `None` | `"idle"` or `"daily"` — why the previous session was auto-reset. |
| `reset_had_activity` | `bool` | `False` | Whether the expired session had any messages (`total_tokens > 0`). |
| `is_fresh_reset` | `bool` | `False` | Set by explicit `/new` or `/reset`. Triggers topic/channel skill re-injection on first message. Distinguished from `was_auto_reset` to avoid misleading "session expired" notices. |
| `expiry_finalized` | `bool` | `False` | Set by background expiry watcher after invoking `on_session_finalize` hooks, cleaning tool resources, and evicting the cached agent. Prevents redundant finalization across restarts. |
| `suspended` | `bool` | `False` | Hard force-wipe signal. Set by `/stop` or stuck-loop escalation (3+ consecutive restart failures). On next `get_or_create_session()`, forces a new `session_id` regardless of `resume_pending`. |
| `resume_pending` | `bool` | `False` | Soft recovery marker. Set by `suspend_recently_active()` (crash recovery) or drain timeout. On next access, preserves the existing `session_id` — the user continues on the same transcript. Cleared after the next successful turn completes. |
| `resume_reason` | `Optional[str]` | `None` | Why resume was marked: `"restart_timeout"`, `"shutdown_timeout"`, `"restart_interrupted"`. |
| `last_resume_marked_at` | `Optional[datetime]` | `None` | Timestamp of the last resume-pending marking. |

### State Transition Logic (get_or_create_session)

```
                    ┌──────────┐
                    │  Incoming │
                    │  Message  │
                    └────┬─────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  session_key exists  │──── No ──► Create fresh SessionEntry
              │  AND !force_new      │
              └──────────┬───────────┘
                         │ Yes
                         ▼
              ┌──────────────────────┐
              │  entry.suspended?    │──── Yes ──► Auto-reset: new session_id
              └──────────┬───────────┘           (reason="suspended")
                         │ No
                         ▼
              ┌──────────────────────┐
              │ entry.resume_pending?│──── Yes ──► Return existing entry
              └──────────┬───────────┘           (preserve session_id)
                         │ No                     Clear flag on next successful turn
                         ▼
              ┌──────────────────────┐
              │   Policy says reset? │──── Yes ──► Auto-reset: new session_id
              └──────────┬───────────┘           (reason="idle"/"daily")
                         │ No
                         ▼
              ┌──────────────────────┐
              │  Return existing     │
              │  entry, bump         │
              │  updated_at          │
              └──────────────────────┘
```

**Priority order in `get_or_create_session()`:**
1. `suspended=True` → always force-reset (hard wipe)
2. `resume_pending=True` → preserve session_id (soft recovery)
3. Policy expiry (idle/daily) → auto-reset
4. No trigger → return existing entry (bump `updated_at`)

---

## 3. SessionStore — Storage and Operations

`SessionStore` is the main storage layer. It maintains an in-memory dict (`_entries`) persisted
to `sessions.json`, with SQLite (`SessionDB`) as the canonical store for session metadata and
message transcripts.

### Constructor

```python
SessionStore(sessions_dir: Path, config: GatewayConfig, has_active_processes_fn=None)
```

- `sessions_dir` — Directory where `sessions.json` lives.
- `config` — `GatewayConfig` instance for reset policy lookups.
- `has_active_processes_fn` — Optional callback keyed by `session_key` to check for running
  background processes. Sessions with active processes are never expired or pruned.

### Operations (Methods)

| Method | Description |
|---|---|
| `get_or_create_session(source, force_new=False)` | Core entry point. Returns existing or creates new `SessionEntry`. Evaluates `suspended`, `resume_pending`, and reset policy. Creates/ends SQLite records. |
| `update_session(session_key, last_prompt_tokens=None)` | Lightweight metadata update after an interaction. Bumps `updated_at`, optionally records `last_prompt_tokens`. |
| `reset_session(session_key, display_name=None)` | Explicit reset (from `/new` or `/reset`). Creates new `session_id`, sets `is_fresh_reset=True`. Ends old SQLite session, creates new one. |
| `switch_session(session_key, target_session_id)` | Switch to a different existing session ID (from `/resume`). Ends current SQLite session, reopens target. |
| `suspend_session(session_key)` | Mark session as `suspended=True` (from `/stop`). Forces auto-reset on next access. |
| `mark_resume_pending(session_key, reason)` | Mark session as `resume_pending=True` (from drain timeout). Preserves session_id on next access. Will NOT override `suspended=True`. |
| `clear_resume_pending(session_key)` | Clear `resume_pending` after a successful resumed turn. Called from gateway after `run_conversation()` returns. |
| `suspend_recently_active(max_age_seconds=120)` | Crash recovery: mark recently-active sessions as `resume_pending=True`. Skips already-pending and already-suspended entries. Called on startup after unclean shutdown. |
| `prune_old_entries(max_age_days)` | Drop entries older than `max_age_days` (based on `updated_at`). Skips `suspended` entries and sessions with active processes. |
| `list_sessions(active_minutes=None)` | Return all sessions, optionally filtered by recent activity. Sorted by `updated_at` descending. |
| `lookup_by_session_id(session_id)` | Find the active `SessionEntry` for a persisted session ID. |
| `has_any_sessions()` | Check if any sessions have ever been created (uses SQLite for history, not just in-memory dict). |
| `append_to_transcript(session_id, message, skip_db=False)` | Append a message to SQLite transcript. `skip_db=True` prevents duplicate writes when the agent already persisted. |
| `rewrite_transcript(session_id, messages)` | Full replacement of session transcript (used by `/retry`, `/undo`, `/compress`). |
| `load_transcript(session_id)` | Load all messages from a session's SQLite transcript. |
| `rewind_session(session_id, n=1)` | Back up `n` user turns via soft-delete (keeps audit trail). Returns `{rewound_count, turns_undone, target_text}`. |

### Internal Helpers

- `_ensure_loaded()` / `_ensure_loaded_locked()` — Load `sessions.json` into `_entries` dict.
- `_save()` — Atomic write to `sessions.json` via temp file + `atomic_replace`.
- `_generate_session_key(source)` — Delegates to `build_session_key()` with config params.
- `_is_session_expired(entry)` — Policy check from entry alone (no source needed). Used by
  background expiry watcher.
- `_should_reset(entry, source)` — Policy check returning `"idle"`, `"daily"`, or `None`.

### Storage Layout

```
{sessions_dir}/
  sessions.json          # In-memory _entries dict, persisted as JSON
                           Maps session_key → SessionEntry (metadata only)
  {session_id}.jsonl     # (Legacy, removed in spec 002)
```

The canonical transcript store is SQLite via `SessionDB` (from `hermes_state`). The
`sessions.json` file persists the `session_key → session_id` mapping and entry metadata
(flags, timestamps, token counts). If SQLite is unavailable, the store falls back to
JSONL, but this is a degradation path.

---

## 4. SessionKey Generation Rules

Session keys are deterministic strings that identify a conversation lane. They are generated
by `build_session_key(source, group_sessions_per_user, thread_sessions_per_user)`.

### Key Format

```
agent:main:{platform}:{chat_type}[:{chat_id}][:{thread_id}][:{participant_id}]
```

### DM Rules

| Scenario | Key |
|---|---|
| DM with chat_id | `agent:main:telegram:dm:12345` |
| DM with chat_id + thread | `agent:main:telegram:dm:12345:thread_678` |
| DM without chat_id, with participant_id | `agent:main:signal:dm:user_abc` |
| DM without chat_id or participant_id | `agent:main:telegram:dm` |
| WhatsApp DM (canonicalized) | `agent:main:whatsapp:dm:{canonical_number}` |

- DMs always include `chat_id` when present, isolating each private conversation.
- `thread_id` further differentiates threaded DMs within the same DM chat.
- Without `chat_id`, falls back to `user_id_alt` or `user_id` as participant_id.
- Without any identifier, all DMs on that platform collapse to one shared session.

### Group/Channel Rules

| Scenario | Key |
|---|---|
| Group chat | `agent:main:telegram:group:-10012345` |
| Group chat, per-user isolation | `agent:main:telegram:group:-10012345:user_abc` |
| Thread in group, shared | `agent:main:discord:group:12345:thread_678` |
| Thread in group, per-user | `agent:main:discord:group:12345:thread_678:user_abc` |
| Channel | `agent:main:slack:channel:C12345` |
| WhatsApp group (canonicalized) | `agent:main:whatsapp:group:{canonical_id}:{participant}` |

- `chat_id` identifies the parent group/channel.
- `thread_id` differentiates threads within that parent.
- **Per-user isolation** (append `participant_id`) is controlled by:
  - `group_sessions_per_user` (default: `True`) — group/channel sessions are isolated.
  - `thread_sessions_per_user` (default: `False`) — threads are **shared** by default
    (Telegram forum topics, Discord threads, Slack threads all share one session per thread).
- `participant_id` = `user_id_alt` or `user_id` (in that priority).
- WhatsApp identifiers are canonicalized to handle JID/LID alias flips.

### Special Case: WhatApp

WhatsApp phone numbers go through `canonical_whatsapp_identifier()` which strips the
`@s.whatsapp.net` suffix and normalizes to E.164 format. This prevents session fragmentation
when the bridge returns different alias forms of the same phone number.

---

## 5. Multi-User Isolation Strategy

Multi-user isolation determines whether multiple users in the same chat share a conversation
or each get their own private session.

### Decision Logic (`is_shared_multi_user_session`)

```python
def is_shared_multi_user_session(source, *, group_sessions_per_user, thread_sessions_per_user):
    if source.chat_type == "dm":
        return False  # DMs are always private
    if source.thread_id:
        return not thread_sessions_per_user  # Threads: shared unless per-user
    return not group_sessions_per_user       # Groups: isolated unless shared
```

### Summary

| Chat Type | Default | Config Control |
|---|---|---|
| DM | Private (never shared) | N/A |
| Group/Channel | Per-user isolation | `group_sessions_per_user` (default: True) |
| Thread (forum, discord) | Shared (all participants see same context) | `thread_sessions_per_user` (default: False) |

### Impact on System Prompt

When `shared_multi_user_session=True`, the system prompt omits a fixed user name and instead
states: *"Multi-user {thread|session} — messages are prefixed with [sender name]. Multiple
users may participate."* Individual sender names are prefixed on each user message by the
gateway at runtime, preserving prompt caching (the system prompt doesn't change per-turn).

---

## 6. Reset Policy

Reset policies control when a session automatically loses context (gets a new `session_id`).

### Policy Modes (`SessionResetPolicy`)

| Mode | Behavior | Default Config |
|---|---|---|
| `"none"` | Never auto-reset. Context managed only by compression. | — |
| `"idle"` | Reset after N minutes of inactivity from `updated_at`. | `idle_minutes: 1440` (24h) |
| `"daily"` | Reset at a specific hour each day (local time). | `at_hour: 4` (4 AM) |
| `"both"` | Whichever triggers first — daily boundary OR idle timeout. | **(default)** |

### Policy Evaluation

```python
# Idle check
idle_deadline = entry.updated_at + timedelta(minutes=policy.idle_minutes)
if now > idle_deadline: return "idle"

# Daily check
today_reset = now.replace(hour=policy.at_hour, minute=0, second=0, microsecond=0)
if now.hour < policy.at_hour:
    today_reset -= timedelta(days=1)  # Reset hasn't happened yet today
if entry.updated_at < today_reset: return "daily"
```

### Per-Platform/Per-Type Policies

Reset policies are configurable per platform and session type via `config.get_reset_policy()`.
This allows different platforms to have different expiry rules (e.g., Telegram DMs reset
after 24h idle, but Slack groups persist indefinitely).

### Exclusions

Sessions with active background processes are **never** expired or reset. The
`has_active_processes_fn` callback checks for running processes when evaluating policies.

### Reset Effects

When a reset triggers:

1. Old session is ended in SQLite (with reason `"session_reset"`).
2. New `session_id` is generated (`YYYYMMDD_HHMMSS_<8hex>`).
3. New `SessionEntry` is created with `was_auto_reset=True` and the reset reason.
4. `reset_had_activity` is set if the old session had any turns (`total_tokens > 0`).
5. The old AIAgent cache entry is evicted on the next expiry watcher pass.
6. On the first message after reset, a context notice is injected: "Session expired due to inactivity / daily reset."

---

## 7. Restart Recovery Flow

The restart recovery system ensures that in-flight sessions are preserved across gateway
restarts, crashes, and drain timeouts. It is the solution to issue #7536.

### Startup Recovery Sequence

```
Gateway starts
       │
       ▼
┌───────────────────────────────┐
│ Check for .clean_shutdown     │── Exists? ──► Skip suspension (clean exit)
│ marker                        │
└───────────────────────────────┘
       │ Missing
       ▼
┌───────────────────────────────┐
│ session_store                 │── Marks sessions updated within
│ .suspend_recently_active()    │   last 120 seconds as resume_pending
└───────────────────────────────┘
       │
       ▼
┌───────────────────────────────┐
│ _suspend_stuck_loop_sessions()│── Suspends sessions that have been
│                               │   active across 3+ restarts
└───────────────────────────────┘
       │
       ▼
┌───────────────────────────────┐
│ Queue inbound messages while  │
│ startup restore runs          │
│ (_startup_restore_in_progress)│
└───────────────────────────────┘
       │
       ▼
┌───────────────────────────────┐
│ For each adapter, find        │
│ resume_pending sessions →     │
│ synthesize MessageEvent and   │
│ run _handle_message to let    │
│ the agent auto-continue       │
└───────────────────────────────┘
```

### suspend_recently_active(max_age_seconds=120)

Called on gateway startup when no `.clean_shutdown` marker exists (indicating a crash or
unexpected exit). For each session updated within the last 120 seconds:

- Sets `resume_pending=True`, `resume_reason="restart_interrupted"`,
  `last_resume_marked_at=now`.
- Skips entries already `resume_pending=True` (no double-mark).
- Skips entries explicitly `suspended=True` (hard wipe should stay).

### Stuck-Loop Detection (`_suspend_stuck_loop_sessions`)

Counts consecutive restarts via a JSON file (`{HERMES_HOME}/restart_counts.json`). If a
session has been active across 3+ consecutive restarts, it's auto-suspended so the user
gets a clean slate.

### Drain-Timeout Marking

On graceful shutdown/restart, the drain system calls `mark_resume_pending()` for any
session that was mid-turn when the drain timeout fired. Reasons:

- `"restart_timeout"` — killed during restart drain
- `"shutdown_timeout"` — killed during shutdown drain
- `"restart_interrupted"` — crash recovery (from `suspend_recently_active`)

All three reasons are in `_AUTO_RESUME_REASONS` and eligible for startup auto-resume.

### Auto-Resume on Next Access

When `get_or_create_session()` encounters `resume_pending=True`:

1. It returns the existing entry **without** creating a new `session_id`.
2. The existing transcript is loaded intact.
3. The marking is not cleared here — it survives until the next successful turn
   completes (`clear_resume_pending()` is called from the gateway after
   `run_conversation()` returns a real response).
4. If the resumed turn is interrupted again, the `resume_pending` flag remains set,
   and the next restart will retry. The stuck-loop counter handles terminal escalation
   (3 retries → suspended).

### Clean Shutdown Marker (`.clean_shutdown`)

Written at the end of a graceful shutdown. On next startup:

- If present: skip `suspend_recently_active()` entirely. Active agents were already
  drained, so no sessions are stuck.
- Then delete the marker.

This prevents unwanted auto-resets after `hermes update`, `hermes gateway restart`,
or `/restart`.

---

## 8. Message Queuing Flow

The message queuing system handles two scenarios:

1. **Interrupt follow-ups** — When a user sends multiple messages while the agent is
   processing, subsequent messages are queued as single-slot pending messages.
2. **`/queue` FIFO** — Explicit `/queue` commands that must each produce their own full
   agent turn, in order, without merging.

### Data Structures

```
adapter._pending_messages: Dict[session_key, MessageEvent]
    └── Single "next-up" slot per session. Overwritten on repeat sends
        (burst collapse). Shared with photo-burst follow-ups.

self._queued_events: Dict[session_key, List[MessageEvent]]
    └── Overflow buffer. Each /queue invocation appends here when the
        slot is occupied. Promoted one-at-a-time after each drain.
```

### Enqueue (`_enqueue_fifo`)

```
_enqueue_fifo(session_key, event, adapter)
       │
       ▼
┌───────────────────────────────────────┐
│ Is slot free?                         │
│ (session_key NOT in _pending_messages)│── Yes ──► Place event in slot
└───────────────────────────────────────┘
       │ No
       ▼
Append to _queued_events[session_key] (overflow tail)
```

### Dequeue / Promotion (`_promote_queued_event`)

Called at the drain site after the slot was consumed. If there's an overflow item:

- When `pending_event is None` (slot was empty), return overflow head as the new event.
- When `pending_event` exists, stage overflow head in the slot for the next recursion.
- If no adapter available, push back to `_queued_events` (don't silently drop).

### Queue Depth

`_queue_depth(session_key, adapter)` returns `len(overflow) + (1 if slot occupied else 0)`.

### Clearing

Queued events for a session are cleared on `/new` and `/reset` (via `_handle_reset_command`).

### FIFO Invariant

Each `/queue` invocation produces exactly one full agent turn, in FIFO order, with no
merging. The single-slot `_pending_messages` + overflow `_queued_events` design ensures
that repeated sends during an active turn don't cause out-of-order processing.

---

## 9. Session Context Injection

`SessionContext` is built from a `SessionSource` and `GatewayConfig` and injected into the
agent's system prompt. It tells the agent:

- Where the current message came from
- What platforms are connected
- Where it can deliver scheduled task outputs
- Whether this is a shared multi-user session

### Construction (`build_session_context`)

```python
def build_session_context(source, config, session_entry=None) -> SessionContext
```

1. Collects connected platforms from config.
2. Collects home channels for each platform.
3. Determines `shared_multi_user_session` via `is_shared_multi_user_session()`.
4. Attaches session metadata (key, id, timestamps) if `session_entry` is provided.

### PII Redaction (`build_session_context_prompt`)

The dynamic system prompt section (`## Current Session Context`) can optionally redact
personally identifiable information before sending to the LLM:

- User IDs → `user_<12hex>` (SHA-256 prefix)
- Chat IDs → `<platform>:<12hex>` or just `<12hex>`
- Platforms excluded from redaction: Discord (needs raw IDs for `@mentions`),
  and any plugin-registered platform not marked `pii_safe`.

Redaction applies only to the system prompt text. Routing, session keys, and adapter
operations always use the original values.

---

## 10. Background Expiry Watcher

The `_session_expiry_watcher` task runs in the gateway event loop every 300 seconds (5 min).

### Responsibilities

1. **Finalize expired sessions** — For each entry where `_is_session_expired()` returns
   True and `expiry_finalized` is False:
   - Invoke `on_session_finalize` plugin hooks (cleanup, notifications).
   - Clean up cached AIAgent resources (close tool resources, shut down memory provider).
   - Evict the cached agent entry.
   - Clear per-session overrides (`_session_model_overrides`, reasoning overrides, etc.).
   - Mark `expiry_finalized=True` and persist (sessions.json + state.db).
   - Promote the state.db session row to `end_reason='session_reset'` via
     `promote_to_session_reset()` — conditional: only live rows or rows ended with a
     recoverable accidental reason (`agent_close`, `ws_orphan_reap`) are promoted, so
     explicit boundaries (`compression`, `session_switch`, …) are never overwritten. This
     durably records the reset so stale-route recovery cannot resurrect the expired
     session with its full history (#61220, #61993, #63539).

2. **Sweep idle cached agents** — Calls `_sweep_idle_cached_agents()` to evict agents that
   have been idle beyond `_AGENT_CACHE_IDLE_TTL_SECS` (3600s / 1h), regardless of session
   reset policy. This prevents unbounded memory growth in gateways with long-lived sessions.

3. **Prune stale entries** — Calls `session_store.prune_old_entries()` hourly based on
   `config.session_store_max_age_days`. Prevents `sessions.json` from growing unbounded.

### Failure Handling

- Per-session retry count: each failed finalize is retried up to 3 consecutive times.
- After 3 failures, the entry is force-marked `expiry_finalized=True` to prevent infinite
  retry loops.

---

## 11. Agent Cache

The gateway maintains an LRU cache of `AIAgent` instances keyed by `session_key` to
preserve prompt caching across turns.

### Cache Properties

- **Max size:** 128 entries (`_AGENT_CACHE_MAX_SIZE`).
- **Eviction policy:** Least-recently-used (LRU via `OrderedDict`).
- **Idle TTL:** 3600s (1h) — enforced by `_session_expiry_watcher`.
- **Lock:** `_agent_cache_lock` (threading) for thread safety.

### Cache Lifecycle

```
Message arrives
    │
    ▼
get_or_create_session()  →  session_key obtained
    │
    ▼
Lookup _agent_cache[session_key]
    │
    ├── Hit → move_to_end(), reuse AIAgent (preserves prompt cache)
    │
    └── Miss → create new AIAgent, store in cache
                (if at capacity, popitem(last=False) evicts LRU entry)
    │
    ▼
run_conversation()  →  agent processes message
    │
    ▼
Session expiry watcher evicts agent when session finalizes
```

### Cleanup Flow

When a session expires:
1. `_cleanup_agent_resources(agent)` — shuts down memory provider, closes tool resources.
2. `_evict_cached_agent(key)` — removes from `_agent_cache` so the agent can be GC'd.

---

## Appendix: Key Configuration

| Config Key | Type | Default | Description |
|---|---|---|---|
| `group_sessions_per_user` | `bool` | `true` | Isolate group/channel sessions per user |
| `thread_sessions_per_user` | `bool` | `false` | Isolate thread sessions per user |
| `session_store_max_age_days` | `int` | `0` | Prune sessions older than N days (0=disabled) |
| `agent.gateway_auto_continue_freshness` | `int` | `3600` | Seconds for resume freshness window |
| `agent.gateway_timeout` | `int` | `1800` | Agent turn timeout (30 min default) |

### Reset Policy (per-platform/type, in config.yaml)

```yaml
session_reset:
  mode: none            # none (default) | idle | daily | both
  at_hour: 4            # daily reset hour (local time)
  idle_minutes: 1440    # idle timeout (24h)
  notify: true          # notify user on auto-reset
```

Platform-specific overrides can be set under `platforms.<name>.session_reset`.
