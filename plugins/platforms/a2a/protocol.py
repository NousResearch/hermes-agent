"""
A2A protocol helpers — Agent Card construction, JSON-RPC framing, task store,
and disk-backed conversation persistence.

Wire shape follows A2A Protocol v1.0 (JSON-RPC 2.0 binding over HTTP):
  - Agent Card served at GET /.well-known/agent.json (and agent-card.json)
  - Tasks via POST {jsonrpc:"2.0", method:"message/send", params:{...}}
  - Streaming via ``message/stream`` → SSE; events are StreamResponse objects
    discriminated by member presence (``statusUpdate`` / ``artifactUpdate``),
    stream closure signals the terminal state (no ``final`` field in v1.0)
  - Task states / message roles are v1.0 SCREAMING_SNAKE_CASE enums
  - Parts are the v1.0 unified shape ({"text": ..., "mediaType": ...}),
    discriminated by member presence (no ``kind`` field)
  - Push notification configs carry ``configId`` + ``createdAt`` and can be
    passed inline in ``message/send`` via configuration.taskPushNotificationConfig

We deliberately implement the subset of A2A needed for text task exchange with
stdlib only (no a2a-sdk). ``extract_text`` stays tolerant of v0.3 peers.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import Future
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

PROTOCOL_VERSION = "1.0"

# A2A v1.0 task lifecycle states.
STATE_SUBMITTED = "TASK_STATE_SUBMITTED"
STATE_WORKING = "TASK_STATE_WORKING"
STATE_INPUT_REQUIRED = "TASK_STATE_INPUT_REQUIRED"
STATE_AUTH_REQUIRED = "TASK_STATE_AUTH_REQUIRED"
STATE_COMPLETED = "TASK_STATE_COMPLETED"
STATE_FAILED = "TASK_STATE_FAILED"
STATE_CANCELED = "TASK_STATE_CANCELED"
STATE_REJECTED = "TASK_STATE_REJECTED"

TERMINAL_STATES = frozenset({STATE_COMPLETED, STATE_FAILED, STATE_CANCELED, STATE_REJECTED})

# A2A v1.0 message roles.
ROLE_USER = "ROLE_USER"
ROLE_AGENT = "ROLE_AGENT"

# The agent starts its reply with this marker when it needs clarification from
# the peer before it can complete the task; the adapter maps such replies to
# TASK_STATE_INPUT_REQUIRED (marker stripped, text in status.message).
INPUT_REQUIRED_MARKER = "[INPUT_REQUIRED]"

# JSON-RPC / A2A error codes.
# -32001..-32003 are A2A spec-defined and used only with their spec semantics.
# Custom errors live at -32050..-32059 (JSON-RPC implementation-defined server
# error space, clear of the A2A-reserved block).
ERR_PARSE = -32700
ERR_INVALID_PARAMS = -32602
ERR_METHOD_NOT_FOUND = -32601
ERR_TASK_NOT_FOUND = -32001        # A2A spec: TaskNotFoundError
ERR_TASK_NOT_CANCELABLE = -32002   # A2A spec: TaskNotCancelableError
ERR_PUSH_NOT_SUPPORTED = -32003    # A2A spec: PushNotificationNotSupportedError
ERR_UNAUTHORIZED = -32050
ERR_RATE_LIMITED = -32051
ERR_UNTRUSTED_PEER = -32052

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


def now_iso() -> str:
    """ISO 8601 UTC timestamp with millisecond precision (A2A v1.0)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


# --------------------------------------------------------------------------
# Agent Card (v1.0)
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
    """Construct an A2A v1.0 Agent Card document."""
    card: dict[str, Any] = {
        "name": name,
        "description": description,
        "url": url,  # convenience for pre-1.0 clients; canonical is supportedInterfaces
        "version": "1.0.0",
        "provider": {
            "organization": os.getenv("A2A_PROVIDER_ORG", "Hermes Agent"),
            "url": os.getenv("A2A_PROVIDER_URL", "") or url,
        },
        "supportedInterfaces": [
            {
                "url": url,
                "protocolBinding": "JSONRPC",
                "protocolVersion": PROTOCOL_VERSION,
            },
        ],
        "capabilities": {
            "streaming": streaming,
            "pushNotifications": push_notifications,
            "stateTransitionHistory": False,
            "extendedAgentCard": False,
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


def skills_from_toolsets(toolsets: "list[str] | dict[str, list[str]] | None") -> list[dict]:
    """Derive A2A skill descriptors from the agent's toolsets.

    Accepts either a plain list of toolset names, or a mapping of toolset name
    → tool names (built from the live tool registry for dynamic Agent Cards —
    tool names become tags so peers can match tasks to us).
    """
    skills = []
    if isinstance(toolsets, dict):
        for ts_name in sorted(toolsets.keys()):
            tool_names = [str(t) for t in (toolsets[ts_name] or [])]
            skills.append({
                "id": f"toolset.{ts_name}",
                "name": ts_name,
                "description": f"Hermes '{ts_name}' capabilities",
                "tags": [ts_name] + tool_names[:10],
            })
    else:
        for ts in sorted(set(toolsets or [])):
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


def text_part(text: str) -> dict:
    """Build a v1.0 text Part (member-presence discriminated, no ``kind``)."""
    return {"text": text, "mediaType": "text/plain"}


def text_message(role: str, text: str, context_id: str = "") -> dict:
    """Build an A2A v1.0 Message with a single text Part."""
    msg: dict[str, Any] = {
        "role": role,  # ROLE_USER | ROLE_AGENT
        "parts": [text_part(text)],
        "messageId": uuid.uuid4().hex,
    }
    if context_id:
        msg["contextId"] = context_id
    return msg


def extract_text(message_or_params: dict) -> str:
    """Pull concatenated text from an A2A Message / Task-result / params payload.

    v1.0 Parts carry a ``text`` member directly; v0.3 used ``kind: "text"``
    and some pre-0.3 peers used ``type``. All three shapes put the payload in
    ``part["text"]``, so presence of a string ``text`` member is the test.
    """
    msg = message_or_params.get("message", message_or_params)
    parts = msg.get("parts", []) if isinstance(msg, dict) else []
    chunks = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        txt = part.get("text")
        if isinstance(txt, str):
            chunks.append(txt)
    return "\n".join(chunks).strip()


def extract_context_id(params: dict) -> str:
    """v1.0 puts contextId inside the Message; tolerate legacy top-level."""
    msg = params.get("message") or {}
    ctx = ""
    if isinstance(msg, dict):
        ctx = str(msg.get("contextId") or "")
    return ctx or str(params.get("contextId") or "")


def build_task(
    task_id: str,
    context_id: str,
    state: str,
    agent_text: str = "",
    *,
    created_at: str = "",
) -> dict:
    """Build an A2A v1.0 Task object for a message/send result."""
    now = now_iso()
    task: dict[str, Any] = {
        "id": task_id,
        "contextId": context_id,
        "status": {"state": state, "timestamp": now},
        "createdAt": created_at or now,
        "lastModified": now,
    }
    if agent_text:
        task["status"]["message"] = text_message(ROLE_AGENT, agent_text, context_id)
        if state == STATE_COMPLETED:
            task["artifacts"] = [{
                "artifactId": uuid.uuid4().hex,
                "parts": [text_part(agent_text)],
            }]
    return task


# --------------------------------------------------------------------------
# Streaming (v1.0 StreamResponse events)
# --------------------------------------------------------------------------

def status_update(task_id: str, context_id: str, state: str, text: str = "") -> dict:
    """v1.0 StreamResponse with a statusUpdate member."""
    status: dict[str, Any] = {"state": state, "timestamp": now_iso()}
    if text:
        status["message"] = text_message(ROLE_AGENT, text, context_id)
    return {"statusUpdate": {"taskId": task_id, "contextId": context_id, "status": status}}


def artifact_update(task_id: str, context_id: str, text: str) -> dict:
    """v1.0 StreamResponse with an artifactUpdate member."""
    return {
        "artifactUpdate": {
            "taskId": task_id,
            "contextId": context_id,
            "artifact": {
                "artifactId": uuid.uuid4().hex,
                "parts": [text_part(text)],
            },
        }
    }


def sse_data(payload: dict) -> str:
    """Encode one StreamResponse as an SSE data frame (member-name discriminated)."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def sse_done() -> str:
    """SSE stream-closure marker — terminal state is implied by closure in v1.0."""
    return "event: done\ndata: {}\n\n"


# --------------------------------------------------------------------------
# Anti-loop ping-pong protection (per-adapter instance)
# --------------------------------------------------------------------------

class TurnTracker:
    """Counts inbound turns per context_id to stop infinite agent↔agent loops.

    A "turn" is one inbound message/send from a peer. When the count exceeds
    max_pingpong_turns(), the adapter rejects further messages for that context.
    """

    _TTL = 3600  # prune contexts idle longer than 1 hour

    def __init__(self) -> None:
        self._counts: dict[str, int] = defaultdict(int)
        self._timestamps: dict[str, float] = {}
        self._lock = threading.Lock()

    def track(self, context_id: str) -> int:
        """Increment and return the turn count; prunes stale contexts."""
        with self._lock:
            now = time.time()
            stale = [cid for cid, ts in self._timestamps.items() if now - ts > self._TTL]
            for cid in stale:
                self._counts.pop(cid, None)
                self._timestamps.pop(cid, None)
            self._counts[context_id] += 1
            self._timestamps[context_id] = now
            return self._counts[context_id]

    def reset(self, context_id: str) -> None:
        """Reset turn count for a context (e.g. after explicit cancel)."""
        with self._lock:
            self._counts.pop(context_id, None)
            self._timestamps.pop(context_id, None)


# --------------------------------------------------------------------------
# Rate limiting (sliding window per authenticated peer identity)
# --------------------------------------------------------------------------

_RATE_LIMIT_DEFAULT = 60  # requests per minute
_RATE_WINDOW = 60.0  # seconds


def _rate_limit_per_minute() -> int:
    try:
        return max(1, int(os.getenv("A2A_RATE_LIMIT", str(_RATE_LIMIT_DEFAULT))))
    except (ValueError, TypeError):
        return _RATE_LIMIT_DEFAULT


class RateLimiter:
    """Sliding-window request limiter, one bucket per authenticated identity."""

    def __init__(self) -> None:
        self._buckets: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, identity: str) -> bool:
        with self._lock:
            limit = _rate_limit_per_minute()
            now = time.time()
            bucket = self._buckets[identity]
            while bucket and now - bucket[0] > _RATE_WINDOW:
                bucket.popleft()
            if len(bucket) >= limit:
                return False
            bucket.append(now)
            return True


# --------------------------------------------------------------------------
# Metrics collection
# --------------------------------------------------------------------------

# Module-level singleton shared by the inbound adapter and the outbound client
# tools so /metrics and a2a_list report both directions. Not persisted.
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
        # Rolling latency tracking (last 100 completed inbound tasks)
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
# Task store — pending AND completed tasks (queryable via tasks/get, tasks/list)
# --------------------------------------------------------------------------

class TaskStore:
    """In-memory store of A2A tasks, kept after completion for tasks/get.

    Terminal tasks are retained (bounded to _MAX_TERMINAL, oldest evicted) so
    peers can poll tasks/get, page tasks/list, and reconnect via
    tasks/subscribe after the fact. Watchers registered via ``watch()`` are
    resolved with (state, reply) when the task reaches a terminal state.
    """

    _MAX_TERMINAL = 500

    def __init__(self) -> None:
        self._tasks: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        self._watchers: dict[str, list[Future]] = {}
        self._lock = threading.Lock()

    def create(self, task_id: str, context_id: str, peer: str) -> dict:
        rec = {
            "task_id": task_id,
            "context_id": context_id,
            "peer": peer,
            "state": STATE_SUBMITTED,
            "reply": "",
            "created_at": time.time(),
            "created_iso": now_iso(),
            "push_url": "",
            "push_config_id": "",
        }
        with self._lock:
            self._tasks[task_id] = rec
        return dict(rec)

    def set_state(self, task_id: str, state: str) -> None:
        with self._lock:
            rec = self._tasks.get(task_id)
            if rec and rec["state"] not in TERMINAL_STATES:
                rec["state"] = state

    def set_push_config(self, task_id: str, url: str) -> Optional[dict]:
        """Attach a push notification config; returns the stored config or None."""
        with self._lock:
            rec = self._tasks.get(task_id)
            if not rec:
                return None
            rec["push_url"] = url
            rec["push_config_id"] = "cfg-" + uuid.uuid4().hex[:12]
            return {
                "configId": rec["push_config_id"],
                "taskId": task_id,
                "createdAt": now_iso(),
                "pushNotificationConfig": {"url": url},
            }

    def pop_push_url(self, task_id: str) -> str:
        with self._lock:
            rec = self._tasks.get(task_id)
            if not rec:
                return ""
            url, rec["push_url"] = rec["push_url"], ""
            return url

    def get(self, task_id: str) -> Optional[dict]:
        with self._lock:
            rec = self._tasks.get(task_id)
            return dict(rec) if rec else None

    def complete(self, task_id: str, state: str, reply: str = "") -> Optional[dict]:
        """Transition a task to a terminal state. Idempotent: returns None if
        the task is unknown or already terminal (prevents double-counting)."""
        watchers: list[Future] = []
        with self._lock:
            rec = self._tasks.get(task_id)
            if not rec or rec["state"] in TERMINAL_STATES:
                return None
            rec["state"] = state
            rec["reply"] = reply
            rec["completed_at"] = time.time()
            watchers = self._watchers.pop(task_id, [])
            self._trim_locked()
            out = dict(rec)
        for fut in watchers:
            if not fut.done():
                fut.set_result((state, reply))
        return out

    def watch(self, task_id: str) -> Optional[Future]:
        """Future resolved with (state, reply) at terminal transition.

        Resolved immediately if the task is already terminal; None if unknown.
        """
        with self._lock:
            rec = self._tasks.get(task_id)
            if not rec:
                return None
            fut: Future = Future()
            if rec["state"] in TERMINAL_STATES:
                fut.set_result((rec["state"], rec.get("reply", "")))
            else:
                self._watchers.setdefault(task_id, []).append(fut)
            return fut

    def list(
        self,
        context_id: str = "",
        state: str = "",
        page_size: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        """Filtered task page (newest first). Returns (records, next_offset);
        next_offset is 0 when there are no more results."""
        page_size = max(1, min(int(page_size or 50), 200))
        with self._lock:
            recs = [dict(r) for r in reversed(self._tasks.values())]
        if context_id:
            recs = [r for r in recs if r["context_id"] == context_id]
        if state:
            recs = [r for r in recs if r["state"] == state]
        page = recs[offset:offset + page_size]
        next_offset = offset + page_size if offset + page_size < len(recs) else 0
        return page, next_offset

    def fail_orphans(self, timeout_seconds: int = 300) -> list[str]:
        """Mark non-terminal tasks older than timeout as FAILED; return their ids."""
        with self._lock:
            now = time.time()
            stale = [
                tid for tid, rec in self._tasks.items()
                if rec["state"] not in TERMINAL_STATES
                and now - rec["created_at"] > timeout_seconds
            ]
        failed = []
        for tid in stale:
            if self.complete(tid, STATE_FAILED, "[task orphaned — no reply produced]"):
                failed.append(tid)
        return failed

    def _trim_locked(self) -> None:
        terminal = [tid for tid, rec in self._tasks.items() if rec["state"] in TERMINAL_STATES]
        excess = len(terminal) - self._MAX_TERMINAL
        for tid in terminal[:max(0, excess)]:
            self._tasks.pop(tid, None)

    @staticmethod
    def to_task(rec: dict) -> dict:
        """Render a stored record as an A2A v1.0 Task object."""
        return build_task(
            rec["task_id"],
            rec["context_id"],
            rec["state"],
            rec.get("reply", ""),
            created_at=rec.get("created_iso", ""),
        )


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
