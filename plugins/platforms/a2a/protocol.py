"""A2A protocol helpers — Agent Card, JSON-RPC framing, task store, conversation persistence.
Wire shape is A2A v1.0: SCREAMING_SNAKE_CASE states/roles; Parts and StreamResponse events are
discriminated by member presence (no ``kind``/``final``); SSE closure signals the terminal state.
Stdlib only. ``extract_text`` stays tolerant of v0.3 peers."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import Future
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from gateway.platforms._shared import coerce_port as _coerce_int

PROTOCOL_VERSION = "1.0"

# A2A v1.0 task lifecycle states + message roles.
STATE_SUBMITTED, STATE_WORKING, STATE_INPUT_REQUIRED = "TASK_STATE_SUBMITTED", "TASK_STATE_WORKING", "TASK_STATE_INPUT_REQUIRED"
STATE_COMPLETED, STATE_FAILED = "TASK_STATE_COMPLETED", "TASK_STATE_FAILED"
STATE_CANCELED, STATE_REJECTED = "TASK_STATE_CANCELED", "TASK_STATE_REJECTED"
TERMINAL_STATES = frozenset({STATE_COMPLETED, STATE_FAILED, STATE_CANCELED, STATE_REJECTED})
ROLE_USER, ROLE_AGENT = "ROLE_USER", "ROLE_AGENT"

# A reply starting with this marker is a clarification request -> TASK_STATE_INPUT_REQUIRED (marker stripped).
INPUT_REQUIRED_MARKER = "[INPUT_REQUIRED]"

# JSON-RPC / A2A error codes. -32001..-32003 are A2A spec-defined; custom errors
# live at -32050..-32059 (implementation-defined space, clear of the A2A block).
ERR_PARSE, ERR_INVALID_PARAMS, ERR_METHOD_NOT_FOUND = -32700, -32602, -32601
ERR_TASK_NOT_FOUND, ERR_TASK_NOT_CANCELABLE = -32001, -32002  # A2A spec: TaskNotFoundError / TaskNotCancelableError
ERR_UNSUPPORTED_OPERATION = -32004  # A2A spec: UnsupportedOperationError
ERR_UNAUTHORIZED, ERR_RATE_LIMITED, ERR_UNTRUSTED_PEER = -32050, -32051, -32052

# Anti-loop: max inbound turns per context. A2A_MAX_PINGPONG_TURNS env, capped at 20.
_DEFAULT_MAX_PINGPONG, _HARD_MAX_PINGPONG = 5, 20
_RATE_LIMIT_DEFAULT, _RATE_WINDOW = 60, 60.0  # requests per minute, window seconds


def _env_int(name: str, default: int) -> int:
    return _coerce_int(os.getenv(name, default), default)


def max_pingpong_turns() -> int:
    v = _env_int("A2A_MAX_PINGPONG_TURNS", _DEFAULT_MAX_PINGPONG)
    return max(1, min(v, _HARD_MAX_PINGPONG))


def now_iso() -> str:
    """ISO 8601 UTC timestamp with millisecond precision (A2A v1.0)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except Exception:
        return Path(os.path.expanduser("~/.hermes"))


def build_agent_card(*, name: str, url: str, description: str, skills: Optional[list[dict]] = None,
                     streaming: bool = False, push_notifications: bool = False, auth_required: bool = False,
                     tenant: str = "") -> dict:
    """A2A v1.0 Agent Card. ``tenant`` is the optional multi-tenancy routing key on
    AgentInterface; when present, clients MUST echo it in request params."""
    iface: dict[str, Any] = {"url": url, "protocolBinding": "JSONRPC", "protocolVersion": PROTOCOL_VERSION, **({"tenant": tenant} if tenant else {})}
    card: dict[str, Any] = {
        "name": name,
        "description": description,
        "url": url,  # convenience for pre-1.0 clients; canonical is supportedInterfaces
        "version": "1.0.0",
        "provider": {"organization": os.getenv("A2A_PROVIDER_ORG", "Hermes Agent"), "url": os.getenv("A2A_PROVIDER_URL", "") or url},
        "supportedInterfaces": [iface],
        "capabilities": {"streaming": streaming, "pushNotifications": push_notifications,
                         "stateTransitionHistory": False, "extendedAgentCard": False},
        "defaultInputModes": ["text/plain"], "defaultOutputModes": ["text/plain"], "skills": skills or [],
    }
    if auth_required:
        card["securitySchemes"] = {"bearer": {"type": "http", "scheme": "bearer"}}
        card["security"] = [{"bearer": []}]
    return card


def skills_from_toolsets(toolsets: "list[str] | dict[str, list[str]] | None") -> list[dict]:
    """A2A skill descriptors from toolset names or a toolset -> tool-names mapping (tool names
    become tags, max 10)."""
    if not isinstance(toolsets, dict):
        toolsets = {ts: [] for ts in set(toolsets or [])}
    skills = [{"id": f"toolset.{name}", "name": name, "description": f"Hermes '{name}' capabilities",
               "tags": [name] + [str(t) for t in (toolsets[name] or [])][:10]} for name in sorted(toolsets)]
    return skills or [{"id": "general", "name": "general", "description": "General-purpose conversational agent", "tags": ["general"]}]


def jsonrpc_result(req_id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def jsonrpc_error(req_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def send_message_response(payload: dict) -> dict:
    """v1.0 SendMessageResponse oneof: exactly one of ``task`` / ``message``."""
    if isinstance(payload, dict) and payload.get("status") and payload.get("id"):
        return {"task": payload}
    return {"message": payload}


def unwrap_send_message_response(result: Any) -> Any:
    """Task/Message inside a v1.0 response; legacy bare payloads pass through."""
    if isinstance(result, dict):
        if isinstance(result.get("task"), dict):
            return result["task"]
        if isinstance(result.get("message"), dict):
            return result["message"]
    return result


def stream_task(task: dict) -> dict:
    """v1.0 StreamResponse with a task member."""
    return {"task": task}


def new_task_id() -> str:
    return "task-" + uuid.uuid4().hex[:16]


def new_context_id() -> str:
    return "ctx-" + uuid.uuid4().hex[:16]


def text_part(text: str) -> dict:
    """v1.0 text Part (member-presence discriminated, no ``kind``)."""
    return {"text": text, "mediaType": "text/plain"}


def text_message(role: str, text: str, context_id: str = "", message_id: str = "") -> dict:
    """A2A v1.0 Message with a single text Part."""
    msg: dict[str, Any] = {"role": role, "parts": [text_part(text)],
                           "messageId": message_id or uuid.uuid4().hex}
    if context_id:
        msg["contextId"] = context_id
    return msg


def _file_note(fname: str, body: str, mtype: str) -> str:
    label = f"[file: {fname}]" if fname else "[file]"
    return f"{label} {body}" + (f" ({mtype})" if mtype else "")


def _json_or_str(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(data)


def extract_text(message_or_params: dict) -> str:
    """Concatenated text from an A2A Message / Task-result / params payload. v1.0, v0.3
    (``kind``) and pre-0.3 (``type``) Parts all carry ``text``; file Parts render as
    URL/filename (raw base64 noted, not decoded); data Parts render their JSON."""
    msg = message_or_params.get("message", message_or_params)
    chunks = []
    for part in msg.get("parts", []) if isinstance(msg, dict) else []:
        if not isinstance(part, dict):
            continue
        if isinstance(txt := part.get("text"), str):
            chunks.append(txt)
        elif isinstance(url := part.get("url"), str) and url:
            chunks.append(_file_note(part.get("filename") or part.get("name") or "", url,
                                     part.get("mediaType") or part.get("mimeType") or ""))
        elif isinstance(v03 := part.get("file"), dict) and isinstance(v03.get("fileWithUri"), str):
            chunks.append(_file_note(v03.get("name") or "", v03["fileWithUri"], v03.get("mimeType") or ""))
        elif isinstance(part.get("raw"), str):
            chunks.append(_file_note(part.get("filename") or "", f"{len(part['raw'])} bytes base64-encoded",
                                     part.get("mediaType") or ""))
        elif (data := part.get("data")) is not None:
            chunks.append(f"[data ({part.get('mediaType') or 'application/json'})]\n{_json_or_str(data)}")
    return "\n".join(chunks).strip()


def extract_context_id(params: dict) -> str:
    """v1.0 puts contextId inside the Message; tolerate legacy top-level."""
    msg = params.get("message") or {}
    return (str(msg.get("contextId") or "") if isinstance(msg, dict) else "") or str(params.get("contextId") or "")


def build_task(task_id: str, context_id: str, state: str, agent_text: str = "", *,
               created_at: str = "", status_timestamp: str = "", artifact_id: str = "",
               status_message_id: str = "") -> dict:
    """A2A v1.0 Task. ``created_at`` is accepted but NOT serialized: the v1.0 Task proto has no
    createdAt and strict ProtoJSON parsers (a2a-sdk) reject unknown fields. Stored identifiers
    keep repeated GetTask rendering stable."""
    task: dict[str, Any] = {"id": task_id, "contextId": context_id,
                            "status": {"state": state, "timestamp": status_timestamp or now_iso()}}
    if agent_text:
        task["status"]["message"] = text_message(ROLE_AGENT, agent_text, context_id,
                                                  message_id=status_message_id)
        if state == STATE_COMPLETED:
            task["artifacts"] = [{"artifactId": artifact_id or uuid.uuid4().hex,
                                  "parts": [text_part(agent_text)]}]
    return task


def status_update(task_id: str, context_id: str, state: str, text: str = "") -> dict:
    """v1.0 StreamResponse with a statusUpdate member."""
    status: dict[str, Any] = {"state": state, "timestamp": now_iso()}
    if text:
        status["message"] = text_message(ROLE_AGENT, text, context_id)
    return {"statusUpdate": {"taskId": task_id, "contextId": context_id, "status": status}}


def artifact_update(task_id: str, context_id: str, text: str) -> dict:
    """v1.0 StreamResponse with an artifactUpdate member."""
    artifact = {"artifactId": uuid.uuid4().hex, "parts": [text_part(text)]}
    return {"artifactUpdate": {"taskId": task_id, "contextId": context_id, "artifact": artifact}}


def sse_data(payload: dict, req_id: Any = None) -> str:
    """One StreamResponse as an SSE data frame. §9.4 requires a full JSON-RPC envelope (a2a-sdk
    breaks on bare StreamResponses); ``req_id=None`` is the legacy no-envelope fallback."""
    envelope = jsonrpc_result(req_id, payload) if req_id is not None else payload
    return f"data: {json.dumps(envelope, ensure_ascii=False)}\n\n"


def sse_done() -> str:
    """Stream-closure marker as an SSE *comment* — ``data: {}`` would make JSON-RPC clients parse."""
    return ": done\n\n"


class TurnTracker:
    """Counts inbound turns per context_id; beyond max_pingpong_turns() the adapter rejects."""

    _TTL = 3600  # prune contexts idle longer than 1 hour

    def __init__(self) -> None:
        self._turns: dict[str, tuple[int, float]] = {}  # context_id -> (count, last_seen)
        self._lock = threading.Lock()

    def track(self, context_id: str) -> int:
        """Increment and return the turn count; prunes stale contexts."""
        with self._lock:
            now = time.time()
            self._turns = {cid: v for cid, v in self._turns.items() if now - v[1] <= self._TTL}
            count = self._turns.get(context_id, (0, now))[0] + 1
            self._turns[context_id] = (count, now)
            return count

    def reset(self, context_id: str) -> None:
        with self._lock:
            self._turns.pop(context_id, None)


class RateLimiter:
    """Sliding-window request limiter, one bucket per authenticated identity."""

    def __init__(self) -> None:
        self._buckets: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, identity: str) -> bool:
        with self._lock:
            limit = max(1, _env_int("A2A_RATE_LIMIT", _RATE_LIMIT_DEFAULT))
            now = time.time()
            bucket = self._buckets[identity]
            while bucket and now - bucket[0] > _RATE_WINDOW:
                bucket.popleft()
            if len(bucket) >= limit:
                return False
            bucket.append(now)
            return True


class Metrics:
    """Counters for A2A operations (module singleton ``metrics`` shared by the inbound adapter
    and outbound tools; not persisted)."""

    _COUNTERS = ("inbound_total", "outbound_total", "streams_started", "push_sent", "push_failed",
                 "tasks_completed", "tasks_failed", "anti_loop_triggers", "rate_limit_triggers")

    def __init__(self) -> None:
        for name in self._COUNTERS:
            setattr(self, name, 0)
        self._start_time = time.time()
        self._latencies: deque[float] = deque(maxlen=100)  # last 100 completed inbound tasks

    def record_latency(self, seconds: float) -> None:
        self._latencies.append(seconds)

    def avg_latency(self) -> float:
        return sum(self._latencies) / len(self._latencies) if self._latencies else 0.0

    def snapshot(self) -> dict[str, Any]:
        return {"uptime_seconds": round(time.time() - self._start_time, 1), **{n: getattr(self, n) for n in self._COUNTERS},
                "avg_latency_ms": round(self.avg_latency() * 1000, 1)}


metrics = Metrics()


class TaskStore:
    """In-memory A2A tasks, kept after completion for tasks/get. Records carry agent slug +
    tenant; readers pass a scope and get not-found outside it (spec authz rule)."""

    _MAX_TERMINAL = 500

    def __init__(self) -> None:
        self._tasks: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        self._watchers: dict[str, list[Future]] = {}
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._completion_sequence = 0

    @staticmethod
    def _in_scope(rec: dict, agent_slug: str = "", tenant: str = "") -> bool:
        return not ((agent_slug and rec.get("agent_slug", "") != agent_slug) or (tenant and rec.get("tenant", "") != tenant))

    def _scoped(self, task_id: str, agent_slug: str = "", tenant: str = "") -> Optional[dict]:
        """Live record if visible in scope. Caller holds the lock."""
        rec = self._tasks.get(task_id)
        return rec if rec and self._in_scope(rec, agent_slug, tenant) else None

    def _push_rec(self, task_id: str, config_id: str = "", agent_slug: str = "", tenant: str = "") -> Optional[dict]:
        """Scoped record that has a push config (matching ``config_id`` if given). Caller holds the lock."""
        rec = self._scoped(task_id, agent_slug, tenant)
        if rec and rec.get("push_url") and (not config_id or rec.get("push_config_id") == config_id):
            return rec
        return None

    @staticmethod
    def _push_config_view(rec: dict) -> dict:
        return {"configId": rec.get("push_config_id") or "", "taskId": rec["task_id"],
                "createdAt": rec.get("created_iso", ""), "pushNotificationConfig": {"url": rec.get("push_url") or ""}}

    def create(self, task_id: str, context_id: str, peer: str, agent_slug: str = "", tenant: str = "") -> dict:
        created_iso = now_iso()
        rec = {"task_id": task_id, "context_id": context_id, "peer": peer, "agent_slug": agent_slug or "", "tenant": tenant or "",
               "state": STATE_SUBMITTED, "reply": "", "progress": "", "created_at": time.time(),
               "created_iso": created_iso, "status_iso": created_iso, "revision": 0,
               "artifact_id": uuid.uuid4().hex, "status_message_id": uuid.uuid4().hex,
               "push_url": "", "push_config_id": ""}
        with self._lock:
            self._tasks[task_id] = rec
        return dict(rec)

    def set_state(self, task_id: str, state: str) -> None:
        with self._lock:
            if (rec := self._tasks.get(task_id)) and rec.get("completed_at") is None:
                rec["state"] = state
                rec["status_iso"] = now_iso()
                rec["revision"] = int(rec.get("revision", 0)) + 1
                self._condition.notify_all()

    def set_progress(self, task_id: str, text: str) -> bool:
        """Replace the latest non-final progress snapshot for a task."""
        with self._lock:
            rec = self._tasks.get(task_id)
            if not rec or rec.get("completed_at") is not None:
                return False
            rec["progress"] = text
            rec["status_iso"] = now_iso()
            rec["revision"] = int(rec.get("revision", 0)) + 1
            self._condition.notify_all()
            return True

    def set_push_config(self, task_id: str, url: str, agent_slug: str = "", tenant: str = "") -> Optional[dict]:
        """Attach a push notification config; returns the stored config or None."""
        with self._lock:
            if not (rec := self._scoped(task_id, agent_slug, tenant)):
                return None
            rec["push_url"], rec["push_config_id"] = url, "cfg-" + uuid.uuid4().hex[:12]
            return self._push_config_view(rec)

    def get_push_config(self, task_id: str, config_id: str = "", agent_slug: str = "", tenant: str = "") -> Optional[dict]:
        with self._lock:
            return self._push_config_view(rec) if (rec := self._push_rec(task_id, config_id, agent_slug, tenant)) else None

    def list_push_configs(self, task_id: str, agent_slug: str = "", tenant: str = "") -> list[dict]:
        cfg = self.get_push_config(task_id, "", agent_slug, tenant)
        return [cfg] if cfg else []

    def delete_push_config(self, task_id: str, config_id: str = "", agent_slug: str = "", tenant: str = "") -> bool:
        with self._lock:
            rec = self._push_rec(task_id, config_id, agent_slug, tenant)
            if rec:
                rec["push_url"] = rec["push_config_id"] = ""
            return rec is not None

    def pop_push_url(self, task_id: str) -> str:
        with self._lock:
            rec = self._tasks.get(task_id)
            if rec:
                url, rec["push_url"] = rec["push_url"], ""
            return url if rec else ""

    def get(self, task_id: str, agent_slug: str = "", tenant: str = "") -> Optional[dict]:
        with self._lock:
            return dict(rec) if (rec := self._scoped(task_id, agent_slug, tenant)) else None

    def complete(self, task_id: str, state: str, reply: str = "") -> Optional[dict]:
        """Finalize one dispatch cycle. Idempotent, including input-required state."""
        with self._lock:
            rec = self._tasks.get(task_id)
            if not rec:
                return None
            if rec.get("completed_at") is not None:
                if not (rec.get("state") == STATE_INPUT_REQUIRED and state == STATE_CANCELED):
                    return None
            rec.update(state=state, reply=reply, completed_at=time.time(), status_iso=now_iso(),
                       revision=int(rec.get("revision", 0)) + 1)
            self._completion_sequence += 1
            rec["completion_seq"] = self._completion_sequence
            watchers = self._watchers.pop(task_id, [])
            self._trim_locked()
            self._condition.notify_all()
            out = dict(rec)
        for fut in watchers:
            if not fut.done():
                fut.set_result((state, reply))
        return out

    def watch(self, task_id: str, agent_slug: str = "", tenant: str = "") -> Optional[Future]:
        with self._lock:
            if not (rec := self._scoped(task_id, agent_slug, tenant)):
                return None
            fut: Future = Future()
            if rec.get("completed_at") is not None:
                fut.set_result((rec["state"], rec.get("reply", "")))
            else:
                self._watchers.setdefault(task_id, []).append(fut)
            return fut

    def wait_for_update(self, task_id: str, revision: int, timeout: float,
                        agent_slug: str = "", tenant: str = "") -> Optional[dict]:
        """Wait for a task snapshot change; timeout returns the unchanged record."""
        with self._condition:
            def changed() -> bool:
                current = self._tasks.get(task_id)
                return (current is None or not self._in_scope(current, agent_slug, tenant)
                        or int(current.get("revision", 0)) != revision)

            self._condition.wait_for(changed, timeout=max(0.0, timeout))
            rec = self._tasks.get(task_id)
            return dict(rec) if rec and self._in_scope(rec, agent_slug, tenant) else None

    def list(self, context_id: str = "", state: str = "", page_size: int = 50, offset: int = 0,
             agent_slug: str = "", tenant: str = "", with_total: bool = False):
        """Filtered task page (newest first) as ``(records, next_offset)``, or
        ``(records, next_offset, total)`` with ``with_total`` (v1.0 ListTasks totalSize)."""
        page_size = max(1, min(int(page_size or 50), 100))
        with self._lock:
            recs = [dict(r) for r in reversed(self._tasks.values())
                    if self._in_scope(r, agent_slug, tenant)
                    and (not context_id or r["context_id"] == context_id) and (not state or r["state"] == state)]
        total = len(recs)
        page = recs[offset:offset + page_size]
        next_offset = offset + page_size if offset + page_size < total else 0
        return (page, next_offset, total) if with_total else (page, next_offset)

    def orphan_ids(self, timeout_seconds: int = 300, skip: Optional[set[str]] = None) -> List[str]:
        skip = skip or set()
        with self._lock:
            stale = [tid for tid, rec in self._tasks.items() if tid not in skip
                     and rec.get("completed_at") is None
                     and time.time() - rec["created_at"] > timeout_seconds]
        return stale

    def fail_orphans(self, timeout_seconds: int = 300, skip: Optional[set[str]] = None) -> List[str]:
        stale = self.orphan_ids(timeout_seconds, skip)
        return [tid for tid in stale if self.complete(tid, STATE_FAILED, "[task orphaned — no reply produced]")]

    def _trim_locked(self) -> None:
        terminal = sorted(((tid, int(rec.get("completion_seq") or 0)) for tid, rec in self._tasks.items()
                           if rec.get("completed_at") is not None), key=lambda item: item[1])
        for tid, _sequence in terminal[:max(0, len(terminal) - self._MAX_TERMINAL)]:
            self._tasks.pop(tid, None)

    @staticmethod
    def to_task(rec: dict, include_artifacts: bool = True) -> dict:
        """Render a stored record as an A2A v1.0 Task."""
        text = (rec.get("progress", "") if rec.get("state") in (STATE_SUBMITTED, STATE_WORKING)
                else rec.get("reply", ""))
        task = build_task(rec["task_id"], rec["context_id"], rec["state"], text,
                          created_at=rec.get("created_iso", ""), status_timestamp=rec.get("status_iso", ""),
                          artifact_id=rec.get("artifact_id", ""),
                          status_message_id=rec.get("status_message_id", ""))
        if not include_artifacts:
            task.pop("artifacts", None)
        return task


def _conv_dir() -> Path:
    return _hermes_home() / "a2a_conversations"


def _conv_path(context_id: str) -> Path:
    safe = "".join(c for c in (context_id or "default") if c.isalnum() or c in "-_") or "default"
    return _conv_dir() / f"{safe}.jsonl"


_conversation_lock = threading.Lock()


@contextmanager
def _conversation_file_lock(path: Path):
    """Serialize a conversation read-modify-append across Hermes processes."""
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def persist_message(context_id: str, role: str, text: str, task_id: str = "") -> None:
    """Append one message to the context's on-disk conversation log. Never raises."""
    try:
        path = _conv_path(context_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _conversation_lock:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"ts": time.time(), "role": role, "text": text,
                                     "task_id": task_id}, ensure_ascii=False) + "\n")
    except Exception:
        pass


def persist_message_once(context_id: str, role: str, text: str, task_id: str) -> bool:
    """Append a task result once across threads and processes."""
    if not task_id:
        return False
    try:
        path = _conv_path(context_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _conversation_lock:
            with _conversation_file_lock(path):
                if path.exists():
                    with path.open("r", encoding="utf-8") as handle:
                        for line in handle:
                            try:
                                existing = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if existing.get("role") == role and existing.get("task_id") == task_id:
                                return False
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"ts": time.time(), "role": role, "text": text,
                                             "task_id": task_id}, ensure_ascii=False) + "\n")
        return True
    except Exception:
        return False


def load_conversation(context_id: str, limit: int = 50) -> list[dict]:
    """Last *limit* messages for a context (empty list if none / unreadable)."""
    try:
        lines = _conv_path(context_id).read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    out: list[dict] = []
    for line in lines:
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out[-limit:]


def list_conversations() -> list[str]:
    """Context-ids that have persisted conversations."""
    return sorted(p.stem for p in _conv_dir().glob("*.jsonl"))


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import copy  # noqa: F401,E402

ERR_PUSH_NOT_SUPPORTED = -32003    # A2A spec: PushNotificationNotSupportedError

STATE_AUTH_REQUIRED = "TASK_STATE_AUTH_REQUIRED"

def data_part(data: Any, media_type: str = "application/json") -> dict:
    """Build a v1.0 data Part (structured data, no ``kind`` field)."""
    return {"data": data, "mediaType": media_type}

def file_part(url: str = "", raw: str = "", filename: str = "",
              media_type: str = "application/octet-stream") -> dict:
    """Build a v1.0 file Part.

    Either ``url`` (file reference) or ``raw`` (base64-encoded bytes) must be
    provided. Discrimination is by member presence — no ``kind`` field.
    """
    part: dict[str, Any] = {"mediaType": media_type}
    if filename:
        part["filename"] = filename
    if url:
        part["url"] = url
    elif raw:
        part["raw"] = raw
    return part

def message_with_parts(role: str, parts: list[dict], context_id: str = "") -> dict:
    """Build an A2A v1.0 Message with arbitrary Parts (text, file, data)."""
    msg: dict[str, Any] = {
        "role": role,
        "parts": parts,
        "messageId": uuid.uuid4().hex,
    }
    if context_id:
        msg["contextId"] = context_id
    return msg

def stream_message(message: dict) -> dict:
    """v1.0 StreamResponse with a message member."""
    return {"message": message}
# ---- END PLUGIN-COMPAT ----
