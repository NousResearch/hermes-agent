# Nudge System Implementation Plan

**Feature:** Agent Self-Nudge System for Persistent Wake-ups  
**Date:** 2026-05-02  
**Status:** ✅ IMPLEMENTED (with RECURRING support)  
**Author:** AI Assistant

---

## Implementation Summary

The Nudge System allows agents to schedule recurring wake-up reminders for themselves that persist across gateway restarts. Unlike cron jobs, nudges don't spawn sub-agents—they simply trigger the main agent to wake up and continue processing with its full context intact.

**RECURRING nudges are recommended** for robustness: if a container restarts while the agent is working, the recurring nudge will still fire at the next interval.

### Quick Start

```python
# RECOMMENDED: Recurring nudge - fires every 5 minutes
schedule_nudge("every 5m", context="Check if deployment is ready")

# One-time nudge - fires once at specific time
schedule_nudge("2025-06-02T14:30:00", name="afternoon-checkin")

# List scheduled nudges
list_nudges()

# Cancel a nudge (stops recurring nudges)
cancel_nudge("abc123")
```

### Usage Example: Infrastructure Deployment Loop (RECURRING)

```
User: "Deploy the new cluster and wait for it to be ready"

Agent: 
1. Triggers deployment via IaC
2. schedule_nudge("every 5m", context="Check if cluster deployment is ready")
3. "Deployment triggered. I'll check every 5 minutes..."

[Container may restart here - no problem! Recurring nudge persists]

[5 minutes later - Nudge fires]

Agent receives: "[Scheduled reminder 'nudge_abc123']: Check if cluster deployment is ready"

Agent:
1. Checks deployment status - NOT READY
2. "Not ready yet. I'll check again in 5 minutes..."
3. (Recurring nudge automatically reschedules - nothing to do!)

[5 minutes later - Nudge fires again]

Agent:
1. Checks deployment status - READY!
2. cancel_nudge("abc123")  # Stop the recurring nudge
3. Continues with configuration tasks
4. "The cluster is ready! Now I'll configure..."
```

**Key advantage of recurring nudges:** Even if the container restarts while the agent is processing step 3, the nudge will still fire at the next 5-minute interval. The agent doesn't need to explicitly reschedule.

---

## Architecture

### Design Philosophy

**Simple and Lightweight:** Nudges are just wake-up calls. No work delegation, no sub-agents, no complex delivery mechanisms.

**Persistent:** Nudges survive container restarts through JSON file storage.

**Agent-Centric:** The agent retains full context. When it wakes up, it remembers everything from its previous turn.

---

## Components

### 1. Nudge Storage (`nudges.py`)

**Location:** `/Users/shahmeershahid/workspace/hermes-agent/nudges.py`

**Responsibilities:**
- Store nudges in `~/.hermes/nudges.json`
- Parse schedule strings ("5m", "every 5m", "2h", "1d", ISO datetime)
- CRUD operations for nudges
- Get due nudges and mark them as fired
- **Auto-reschedule recurring nudges**
- Cleanup old fired nudges

**Key Functions:**
```python
def create_nudge(session_id, session_key, schedule, context=None, name=None)
def get_due_nudges(now=None)
def fire_nudge(nudge_id)  # Auto-reschedules recurring nudges
def cleanup_old_nudges(max_age_hours=24)
```

**Schedule Formats:**
- One-time: `"5m"`, `"30s"`, `"2h"`, `"1d"`, `"2025-06-02T14:30:00"`
- Recurring: `"every 5m"`, `"every 30s"`, `"every 2h"`, `"every 1d"`

**Nudge Schema:**
```python
{
    "id": "abc123def456",
    "name": "deployment-check",
    "session_id": "20250602_120000_abc12345",
    "session_key": "agent:main:telegram:dm:1234567890",
    "fire_at": "2025-06-02T12:05:00",
    "context": "Check if deployment is ready",
    "created_at": "2025-06-02T12:00:00",
    "fired": False,
    "fired_at": None,
    # Recurring nudge fields
    "is_recurring": True,
    "interval_seconds": 300,  # 5 minutes
    "schedule": "every 5m",
    "fire_count": 0,  # How many times fired (for recurring)
}
```

---

### 2. Agent Tools (`tools/schedule_nudge_tool.py`)

**Location:** `/Users/shahmeershahid/workspace/hermes-agent/tools/schedule_nudge_tool.py`

**Tools Registered:**
- `schedule_nudge` - Create a new nudge
- `list_nudges` - List scheduled nudges
- `cancel_nudge` - Cancel a nudge

**Toolset:** `nudge`

**Example Usage:**
```python
# Inside an agent turn
result = schedule_nudge(
    schedule="5m",
    context="Check deployment status",
    name="deployment-poll"
)
# Returns: {"success": True, "nudge_id": "abc123", "fire_at": "..."}
```

---

### 3. Gateway Integration (`gateway/run.py`)

**Modified Functions:**

#### `_start_cron_ticker()`
- Added `runner` parameter to access gateway state
- Calls `_process_due_nudges()` on each tick
- Periodic cleanup of old nudges

#### `_process_due_nudges()`
- Gets due nudges from `nudges.get_due_nudges()`
- Marks them as fired
- Calls `_trigger_nudge_wake()` for each

#### `_trigger_nudge_wake()`
- Verifies session still exists and isn't suspended
- Creates synthetic `MessageEvent` with nudge context
- Schedules `_handle_message()` on the event loop

**Wake Message Format:**
```
[Scheduled reminder 'deployment-poll']: Check if deployment is ready
```

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ AGENT TURN                                                  │
│ 1. Agent schedules nudge:                                   │
│    schedule_nudge("5m", context="Check deployment")        │
│ 2. Nudge stored in ~/.hermes/nudges.json                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ [Time passes - container may restart]                       │
│ Nudge persists in JSON file                                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ CRON TICKER (every 60s)                                     │
│ 1. Calls _process_due_nudges()                             │
│ 2. Finds due nudges                                         │
│ 3. Marks them as fired                                      │
│ 4. Triggers agent wake-ups                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ AGENT WAKE-UP                                               │
│ 1. Receives synthetic message with nudge context           │
│ 2. Full context preserved from previous turn                │
│ 3. Agent continues workflow                                 │
│ 4. Can schedule another nudge if needed                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Persistence Model

### What Survives Container Restart

| File | Location | Purpose |
|------|----------|---------|
| Nudges | `~/.hermes/nudges.json` | Scheduled nudges |
| Session DB | `~/.hermes/sessions.db` | Conversation history |
| Session Store | `~/.hermes/gateway_sessions.json` | Session metadata |

### Nudge Lifecycle

**One-Time Nudge:**
```
CREATED → DUE → FIRED → CLEANED_UP
   │       │      │         │
   │       │      │         └── After 24 hours
   │       │      └──────────── Marked fired, agent woken
   │       └─────────────────── Time reached
   └─────────────────────────── schedule_nudge() called
```

**Recurring Nudge:**
```
CREATED → DUE → FIRED → DUE → FIRED → DUE → ... → CANCELLED
   │       │      │       │      │       │           │
   │       │      │       │      │       │           └── cancel_nudge() called
   │       │      │       │      │       └────────────── Next interval reached
   │       │      │       │      └────────────────────── Agent woken, auto-rescheduled
   │       │      │       └───────────────────────────── Time reached
   │       │      └───────────────────────────────────── Agent woken, rescheduled
   │       └──────────────────────────────────────────── Time reached
   └──────────────────────────────────────────────────── schedule_nudge("every 5m") called
```

---

## Comparison: Nudges vs Cron Jobs

| Feature | Nudges | Cron Jobs |
|---------|--------|-----------|
| **Work delegation** | None - main agent wakes up | Spawns sub-agent |
| **Context** | Full context preserved | Isolated context |
| **Recurrence** | Built-in ("every 5m") | Full cron expressions |
| **Use case** | Wait-and-continue loops | Periodic background tasks |
| **Persistence** | JSON file | JSON file |
| **Complexity** | Simple wake-up call | Full job execution |
| **Output delivery** | Direct to waking agent | Configurable (chat, file, etc.) |

### When to Use Nudges

- **Waiting for infrastructure to be ready** (use "every 5m" recurring)
- **Polling with full context awareness** (you remember what you're waiting for)
- **Implementing "loops" that need to survive restarts** (recurring handles this)
- **Simple delays in workflow** (one-time ok if restart unlikely)

### When to Use Cron Jobs

- Periodic background tasks independent of conversation
- Data collection and monitoring
- Scheduled reports or notifications
- Tasks that don't need conversation context

### One-Time vs Recurring Nudges

| Type | Syntax | Use When | Risk |
|------|--------|----------|------|
| **One-time** | "5m", "1h", ISO datetime | Short delays (< 1 min), or when restart is impossible | If container restarts before you reschedule, you lose the nudge |
| **Recurring** | "every 5m", "every 1h" | **Always recommended** for robustness | None - automatically reschedules |

**Best Practice:** Always use recurring nudges ("every 5m") unless you have a specific reason not to. Cancel them when done with `cancel_nudge()`.

---

## Configuration

### Environment Variables (Optional)

None required - nudges work out of the box.

### Config Options (config.yaml)

```yaml
# Optional: Nudge cleanup settings
nudge:
  cleanup_after_hours: 24  # How long to keep fired nudges
```

---

## Files Modified

| File | Changes |
|------|---------|
| `nudges.py` | **NEW** - Nudge storage and management |
| `tools/schedule_nudge_tool.py` | **NEW** - Agent tools for scheduling nudges |
| `cron/scheduler.py` | Added `get_and_fire_due_nudges()` export |
| `gateway/run.py` | Added nudge processing to cron ticker |

---

## Testing

### Manual Test

1. Start gateway
2. In a chat, have agent call: `schedule_nudge("1m", context="Test nudge")`
3. Wait 1 minute
4. Agent should receive wake-up message and respond

### Container Restart Test

1. Schedule nudge for 5 minutes
2. Stop gateway
3. Wait 5 minutes
4. Start gateway
5. Agent should receive wake-up message immediately

---

## Future Enhancements

- **Recurring nudges:** Support for "every 5m" style scheduling
- **Nudge history:** Keep history of fired nudges for debugging
- **Nudge retry:** If agent wake-up fails, retry with backoff
- **Web UI:** Visual nudge management in dashboard

---

## Summary

The Nudge System provides a lightweight, persistent mechanism for agents to schedule their own wake-ups. It's simpler than cron jobs (no sub-agents), maintains full context, and survives container restarts through JSON file persistence.

**Key Benefits:**
- ✅ Simple mental model - just a wake-up call
- ✅ No context loss - agent remembers everything
- ✅ Survives restarts - works with containerized deployments
- ✅ Minimal resource usage - no sub-agent spawning
- ✅ Easy to use - single function call
