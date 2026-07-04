"""
A2A protocol helpers — Agent Card construction, JSON-RPC framing, and
disk-backed conversation persistence.

Wire shape follows the A2A spec (JSON-RPC 2.0 over HTTP):
  - Agent Card served at GET /.well-known/agent.json
  - Tasks via POST {jsonrpc:"2.0", method:"message/send", params:{...}}
  - Streaming via POST {jsonrpc:"2.0", method:"message/stream", params:{...}}
    → SSE response with task state transitions and artifact deltas
  - Push notifications via POST {jsonrpc:"2.0", method:"tasks/pushNotification/set"}
  - Methods handled inbound: message/send, message/stream, tasks/get,
    tasks/pushNotification/set, tasks/cancel

We deliberately implement the subset of A2A needed for text task exchange with
stdlib only (no a2a-sdk). If a2a-sdk is later added as an optional extra, the
client can upgrade transparently — the wire format is identical.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Optional

# A2A task lifecycle states (subset we use).
STATE_SUBMITTED = "submitted"
STATE_WORKING = "working"
STATE_INPUT_REQUIRED = "input-required"
STATE_COMPLETED = "completed"
STATE_FAILED = "failed"
STATE_CANCELED = "canceled"

# Maximum turns an A2A conversation can have before anti-loop kicks in.
# Default 5, configurable via A2A_MAX_PINGPONG_TURNS env (max 20).
_DEFAULT_MAX_PINGPONG = 5
_HARD_MAX_PINGPONG = 20


def max_pingpong_turns() -> int:
    try:
        v = int(os.getenv("A2A_MAX_PINGPONG_TURNS", str(_DEFAULT_MAX_PINGPONG)))
        return max(1, min(v, _HARD_MAX_PINGPONG))
    except (ValueError, TypeError):
        return _DEFAULT_MAX_PINGPONG


# --------------------------------------------------------------------------
# Agent Card
# --------------------------------------------------------------------------

def build_agent_card(
    *,
    name: str,
    url: str,
    description: str,
    skills: Optional[list[dict]] = None,
    streaming: bool = False,
    push_notifications: bool = False,
    auth_required: bool = False,
) -> dict:
    """Construct an A2A Agent Card document (the /.well-known/agent.json body)."""
    card: dict[str, Any] = {
        "name": name,
        "description": description,
        "url": url,
        "version": "0.2.0",
        "protocolVersion": "0.3",
        "capabilities": {
            "streaming": streaming,
            "pushNotifications": push_notifications,
            "stateTransitionHistory": False,
        },
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": skills or [],
    }
    if auth_required:
        card["securitySchemes"] = {
            "bearer": {"type": "http", "scheme": "bearer"}
        }
        card["security"] = [{"bearer": []}]
    return card


def skills_from_toolsets(toolset_names: list[str]) -> list[dict]:
    """Derive A2A skill descriptors from the agent's enabled toolsets.

    A2A 'skills' are coarse capability advertisements, not tool schemas. We map
    each enabled toolset to one skill entry so peers can match tasks to us.
    """
    skills = []
    for ts in sorted(set(toolset_names or [])):
        skills.append({
            "id": f"toolset.{ts}",
            "name": ts,
            "description": f"Hermes '{ts}' capabilities",
            "tags": [ts],
        })
    if not skills:
        skills.append({
            "id": "general",
            "name": "general",
            "description": "General-purpose conversational agent",
            "tags": ["general"],
        })
    return skills


def skills_from_real_toolsets(toolset_registry: dict) -> list[dict]:
    """Build A2A skill descriptors from the real toolset registry.

    Unlike ``skills_from_toolsets`` which takes a list of names, this accepts
    the actual toolset registry dict (toolset_name → {tools: [...], description: ...})
    and produces richer skill cards with per-tool descriptions.

    This enables Dynamic Agent Cards: the card we serve reflects what the agent
    can *actually do* right now, not a static list.
    """
    skills = []
    if toolset_registry and isinstance(toolset_registry, dict):
        for ts_name in sorted(toolset_registry.keys()):
            ts_info = toolset_registry[ts_name] or {}
            tools_list = ts_info.get("tools", []) if isinstance(ts_info, dict) else []
            tool_names = [t.get("name", str(t)) if isinstance(t, dict) else str(t) for t in (tools_list or [])]
            desc = ts_info.get("description", f"Hermes '{ts_name}' capabilities") if isinstance(ts_info, dict) else f"Hermes '{ts_name}' capabilities"
            skills.append({
                "id": f"toolset.{ts_name}",
                "name": ts_name,
                "description": desc,
                "tags": [ts_name],
                # Include tool names as tags for capability matching
                "tags": [ts_name] + tool_names[:10],
            })
    if not skills:
        skills.append({
            "id": "general",
            "name": "general",
            "description": "General-purpose conversational agent",
            "tags": ["general"],
        })
    return skills


# --------------------------------------------------------------------------
# JSON-RPC framing
# --------------------------------------------------------------------------

def jsonrpc_result(req_id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def jsonrpc_error(req_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def new_task_id() -> str:
    return "task-" + uuid.uuid4().hex[:16]


def new_context_id() -> str:
    return "ctx-" + uuid.uuid4().hex[:16]


def text_message(role: str, text: str) -> dict:
    """Build an A2A Message with a single text Part."""
    return {
        "role": role,  # "user" | "agent"
        "parts": [{"kind": "text", "text": text}],
        "messageId": uuid.uuid4().hex,
    }


def extract_text(message_or_params: dict) -> str:
    """Pull concatenated text from an A2A Message / params payload.

    Tolerant of both ``{"message": {...}}`` params and a bare message dict, and
    of both ``kind`` and legacy ``type`` part discriminators.
    """
    msg = message_or_params.get("message", message_or_params)
    parts = msg.get("parts", []) if isinstance(msg, dict) else []
    chunks = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if part.get("kind") in (None, "text") or part.get("type") == "text":
            txt = part.get("text")
            if isinstance(txt, str):
                chunks.append(txt)
    return "\n".join(chunks).strip()


def build_task(task_id: str, context_id: str, state: str, agent_text: str = "") -> dict:
    """Build an A2A Task object for a message/send result."""
    task: dict[str, Any] = {
        "id": task_id,
        "contextId": context_id,
        "status": {"state": state, "timestamp": _now_iso()},
        "kind": "task",
    }
    if agent_text:
        task["status"]["message"] = text_message("agent", agent_text)
        task["artifacts"] = [{
            "artifactId": uuid.uuid4().hex,
            "parts": [{"kind": "text", "text": agent_text}],
        }]
    return task


def build_streaming_event(event_type: str, task_id: str, context_id: str, data: dict | None = None) -> str:
    """Build a single SSE event for message/stream responses.

    Event types following A2A spec:
    - ``task``: full task state transition (submitted → working → completed)
    - ``artifact``: incremental artifact delta
    - ``status``: status update (state + optional message)
    - ``done``: stream complete marker
    """
    payload: dict[str, Any] = {"taskId": task_id, "contextId": context_id}
    if data:
        payload.update(data)
    return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --------------------------------------------------------------------------
# Anti-loop ping-pong protection
# --------------------------------------------------------------------------

# Track turns per context_id to prevent infinite agent-to-agent loops.
# A "turn" is one inbound message/send from a peer. When the count exceeds
# max_pingpong_turns(), we reject further messages for that context.
# OpenClaw pattern: maxPingPongTurns, default 5, max 20.

_turn_counts: dict[str, int] = defaultdict(int)
_turn_timestamps: dict[str, float] = {}

# Clean up turn tracking for contexts older than 1 hour.
_TURN_TTL = 3600


def track_turn(context_id: str) -> int:
    """Increment and return the turn count for this context.

    Returns the *new* count. Caller should reject if > max_pingpong_turns().
    Also prunes stale entries to prevent unbounded growth.
    """
    now = time.time()
    # Prune stale entries
    stale = [cid for cid, ts in _turn_timestamps.items() if now - ts > _TURN_TTL]
    for cid in stale:
        _turn_counts.pop(cid, None)
        _turn_timestamps.pop(cid, None)

    _turn_counts[context_id] += 1
    _turn_timestamps[context_id] = now
    return _turn_counts[context_id]


def turn_count(context_id: str) -> int:
    """Return current turn count for a context (0 if unknown)."""
    return _turn_counts.get(context_id, 0)


def reset_turns(context_id: str) -> None:
    """Reset turn count for a context (e.g. after explicit cancel)."""
    _turn_counts.pop(context_id, None)
    _turn_timestamps.pop(context_id, None)


# --------------------------------------------------------------------------
# Metrics collection
# --------------------------------------------------------------------------

# Lightweight in-memory metrics. Not persisted — resets on restart.
# For a real deployment, export these via the /metrics endpoint (adapter.py).
class Metrics:
    """Simple counters for A2A operations."""

    def __init__(self) -> None:
        self.inbound_total = 0
        self.outbound_total = 0
        self.streams_started = 0
        self.push_sent = 0
        self.push_failed = 0
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.anti_loop_triggers = 0
        self.rate_limit_triggers = 0
        self._start_time = time.time()
        # Rolling latency tracking (last 100 requests)
        self._latencies: deque[float] = deque(maxlen=100)

    def record_latency(self, seconds: float) -> None:
        self._latencies.append(seconds)

    def avg_latency(self) -> float:
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)

    def snapshot(self) -> dict[str, Any]:
        uptime = time.time() - self._start_time
        return {
            "uptime_seconds": round(uptime, 1),
            "inbound_total": self.inbound_total,
            "outbound_total": self.outbound_total,
            "streams_started": self.streams_started,
            "push_sent": self.push_sent,
            "push_failed": self.push_failed,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "anti_loop_triggers": self.anti_loop_triggers,
            "rate_limit_triggers": self.rate_limit_triggers,
            "avg_latency_ms": round(self.avg_latency() * 1000, 1),
        }


metrics = Metrics()


# --------------------------------------------------------------------------
# Conversation persistence (outside the context-compaction pipeline)
# --------------------------------------------------------------------------

def _conv_dir() -> Path:
    try:
        from hermes_constants import get_hermes_home
        base = Path(get_hermes_home())
    except Exception:
        base = Path(os.path.expanduser("~/.hermes"))
    return base / "a2a_conversations"


def _safe_name(context_id: str) -> str:
    return "".join(c for c in (context_id or "default") if c.isalnum() or c in "-_") or "default"


def persist_message(context_id: str, role: str, text: str, task_id: str = "") -> None:
    """Append one message to the context's on-disk conversation log."""
    try:
        d = _conv_dir()
        d.mkdir(parents=True, exist_ok=True)
        rec = {"ts": time.time(), "role": role, "text": text, "task_id": task_id}
        with (d / f"{_safe_name(context_id)}.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def load_conversation(context_id: str, limit: int = 50) -> list[dict]:
    """Load the last *limit* messages for a context (empty list if none)."""
    path = _conv_dir() / f"{_safe_name(context_id)}.jsonl"
    if not path.exists():
        return []
    out: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        return []
    return out[-limit:]


def list_conversations() -> list[str]:
    """Return known context-ids that have persisted conversations."""
    d = _conv_dir()
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.jsonl"))


# --------------------------------------------------------------------------
# Rate limiting (token bucket per peer)
# --------------------------------------------------------------------------

# Simple token-bucket rate limiter. Each peer gets a bucket.
# Configurable via A2A_RATE_LIMIT (requests per minute, default 60).
# OpenClaw pattern: rate limiting per agent identity.

_RATE_LIMIT_DEFAULT = 60  # requests per minute
_rate_buckets: dict[str, deque[float]] = defaultdict(deque)
_RATE_WINDOW = 60.0  # seconds


def _rate_limit_per_minute() -> int:
    try:
        return max(1, int(os.getenv("A2A_RATE_LIMIT", str(_RATE_LIMIT_DEFAULT))))
    except (ValueError, TypeError):
        return _RATE_LIMIT_DEFAULT


def rate_limit_allow(peer: str) -> bool:
    """Check if peer is within rate limit. Returns True if allowed."""
    limit = _rate_limit_per_minute()
    now = time.time()
    bucket = _rate_buckets[peer]
    # Expire old entries
    while bucket and now - bucket[0] > _RATE_WINDOW:
        bucket.popleft()
    if len(bucket) >= limit:
        return False
    bucket.append(now)
    return True


def rate_limit_status(peer: str) -> dict[str, Any]:
    """Return rate limit status for a peer."""
    limit = _rate_limit_per_minute()
    now = time.time()
    bucket = _rate_buckets[peer]
    # Count active entries
    active = sum(1 for ts in bucket if now - ts <= _RATE_WINDOW)
    return {
        "peer": peer,
        "limit_per_minute": limit,
        "used": active,
        "remaining": max(0, limit - active),
    }


# --------------------------------------------------------------------------
# Pending task registry (for async durable messaging)
# --------------------------------------------------------------------------

# Tracks tasks that are in-flight (submitted/working state) so we can
# support async completion notifications and orphaned task cleanup.
# OpenClaw pattern: sessions_send (async durable messaging).

_pending_tasks: dict[str, dict[str, Any]] = {}
_pending_lock = None  # Will be set by adapter on init


def register_pending_task(task_id: str, context_id: str, peer: str, callback_url: str = "") -> None:
    """Register a task as pending (in-flight)."""
    import threading
    global _pending_lock
    if _pending_lock is None:
        _pending_lock = threading.Lock()
    with _pending_lock:
        _pending_tasks[task_id] = {
            "context_id": context_id,
            "peer": peer,
            "callback_url": callback_url,
            "started_at": time.time(),
            "state": STATE_WORKING,
        }


def complete_pending_task(task_id: str, state: str, reply: str = "") -> dict | None:
    """Mark a pending task as complete. Returns the task info if found."""
    import threading
    global _pending_lock
    if _pending_lock is None:
        _pending_lock = threading.Lock()
    with _pending_lock:
        info = _pending_tasks.pop(task_id, None)
    if info:
        info["state"] = state
        info["reply"] = reply
        info["completed_at"] = time.time()
    return info


def pending_task_info(task_id: str) -> dict | None:
    """Get info about a pending task (for tasks/get)."""
    import threading
    global _pending_lock
    if _pending_lock is None:
        _pending_lock = threading.Lock()
    with _pending_lock:
        return _pending_tasks.get(task_id)


def orphaned_tasks(timeout_seconds: int = 300) -> list[dict]:
    """Find tasks that have been pending longer than timeout.

    Used by the orphaned task watchdog to clean up stale tasks.
    """
    import threading
    global _pending_lock
    if _pending_lock is None:
        _pending_lock = threading.Lock()
    now = time.time()
    with _pending_lock:
        return [
            {"task_id": tid, **info}
            for tid, info in _pending_tasks.items()
            if now - info.get("started_at", now) > timeout_seconds
        ]


def clear_orphaned_tasks(timeout_seconds: int = 300) -> list[str]:
    """Remove and return task_ids of tasks pending longer than timeout."""
    import threading
    global _pending_lock
    if _pending_lock is None:
        _pending_lock = threading.Lock()
    now = time.time()
    cleared = []
    with _pending_lock:
        for tid in list(_pending_tasks.keys()):
            info = _pending_tasks[tid]
            if now - info.get("started_at", now) > timeout_seconds:
                _pending_tasks.pop(tid, None)
                cleared.append(tid)
    return cleared