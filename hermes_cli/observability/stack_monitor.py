"""Local stack-monitor producer for Hermes lifecycle observations.

This is an adapter only: stack-monitor owns the wire contract, collector,
SQLite store, retention, and projections. Hermes contributes bounded,
metadata-only lifecycle observations through the existing first-party hook
seam. The caller path only enqueues; socket I/O happens on a daemon thread.
"""

from __future__ import annotations

import atexit
import datetime as _datetime
import json
import logging
import os
import queue
import socket
import struct
import threading
import uuid
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

_MAX_QUEUE = 2048
_MAX_FIELD = 256
_DEFAULT_CAPACITY = _MAX_QUEUE

_HANDLED_HOOKS = frozenset(
    {
        "on_session_start",
        "on_session_end",
        "on_session_finalize",
        "on_session_reset",
        "pre_llm_call",
        "pre_api_request",
        "post_api_request",
        "api_request_error",
        "pre_tool_call",
        "post_tool_call",
        "post_approval_response",
        "subagent_start",
        "subagent_stop",
        "on_skill_lifecycle",
    }
)


def _utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def _bounded(value: Any, limit: int = _MAX_FIELD) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:limit]


def _socket_path(config: Mapping[str, Any]) -> Path:
    raw = config.get("socket_path")
    if raw:
        return Path(os.path.expandvars(os.path.expanduser(str(raw))))
    override = os.environ.get("ARES_STACK_OBSERVATION_SOCKET")
    if override:
        return Path(override)
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"  # windows-footgun: ok — XDG runtime dir is POSIX-only by definition
    return Path(runtime_dir) / "ares-observatory" / "collector.sock"


def _config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        monitoring = (load_config() or {}).get("monitoring") or {}
        raw = monitoring.get("stack_observation") or {}
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _enabled(config: Mapping[str, Any]) -> bool:
    raw = os.environ.get("ARES_STACK_OBSERVATION_ENABLED")
    if raw is not None:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(config.get("enabled", True))


def _kind_status(hook_name: str, kwargs: Mapping[str, Any]) -> tuple[str, str]:
    if hook_name == "terminal_observation_gap":
        return str(kwargs.get("gap_kind") or "health"), "cancelled"
    if hook_name in {"pre_api_request", "pre_llm_call"}:
        return "llm_call", "started"
    if hook_name in {"post_api_request", "on_session_end"}:
        return "llm_call" if hook_name == "post_api_request" else "health", "completed"
    if hook_name == "api_request_error":
        return "llm_call", "failed"
    if hook_name in {"pre_tool_call"}:
        return "tool", "started"
    if hook_name == "post_tool_call":
        status = str(kwargs.get("status") or "completed").lower()
        if status not in {"completed", "failed", "cancelled", "started", "streaming", "retried"}:
            status = "completed"
        return "tool", status
    if hook_name == "post_approval_response":
        return "receipt", "completed"
    if hook_name in {"subagent_start"}:
        return "graph_run", "started"
    if hook_name in {"subagent_stop"}:
        return "graph_run", "completed" if kwargs.get("status") in {None, "completed", "ok", "success"} else "failed"
    if hook_name == "on_skill_lifecycle":
        return "receipt", "completed"
    return "health", "completed"


def _correlation(kwargs: Mapping[str, Any]) -> dict[str, str | None]:
    return {
        "session_id": _bounded(kwargs.get("session_id")),
        "run_id": _bounded(kwargs.get("run_id") or kwargs.get("task_id")),
        "trace_id": _bounded(kwargs.get("trace_id")),
        "span_id": _bounded(kwargs.get("span_id")),
        "parent_span_id": _bounded(kwargs.get("parent_span_id")),
        "node_id": _bounded(kwargs.get("node_id")),
        "attempt_id": _bounded(kwargs.get("attempt_id")),
        "trial_id": _bounded(kwargs.get("trial_id")),
        "request_id": _bounded(kwargs.get("api_request_id") or kwargs.get("tool_call_id")),
    }


def build_envelope(hook_name: str, kwargs: Mapping[str, Any], *, sequence: int) -> dict[str, Any]:
    """Build a content-free stack-observation envelope from one hook payload."""
    kind, status = _kind_status(hook_name, kwargs)
    payload: dict[str, Any] = {"hook": hook_name}
    for key in ("tool_name", "child_role", "platform", "approval_outcome", "error_type"):
        value = _bounded(kwargs.get(key))
        if value is not None:
            payload[key] = value
    if hook_name == "subagent_stop":
        payload["tool_call_count"] = int(kwargs.get("tool_call_count") or 0)
        payload["files_read_count"] = int(kwargs.get("files_read_count") or 0)
        payload["files_written_count"] = int(kwargs.get("files_written_count") or 0)
    if hook_name == "post_tool_call":
        payload["status"] = status
    if hook_name == "api_request_error":
        payload["retryable"] = bool(kwargs.get("retryable", False))
        payload["retry_count"] = int(kwargs.get("retry_count") or 0)
    if hook_name == "on_session_end" and isinstance(kwargs.get("coverage"), Mapping):
        payload["coverage"] = {
            str(key): int(value)
            for key, value in kwargs["coverage"].items()
            if isinstance(value, int) and value >= 0
        }
    if hook_name == "terminal_observation_gap":
        payload["reason"] = "session_end_without_terminal_hook"
        payload["missing_terminal_hook"] = _bounded(kwargs.get("missing_terminal_hook"))

    timing: dict[str, Any] = {}
    for source, target in (
        ("model", "model"),
        ("provider", "provider"),
        ("finish_reason", "error_category"),
    ):
        value = _bounded(kwargs.get(source))
        if value is not None:
            timing[target] = value
    for source, target in (
        ("approx_input_tokens", "prompt_tokens"),
        ("assistant_content_chars", "completion_tokens"),
        ("api_call_count", "total_tokens"),
    ):
        value = kwargs.get(source)
        if isinstance(value, int) and value >= 0:
            timing[target] = value
    duration = kwargs.get("api_duration") or kwargs.get("duration_ms")
    if isinstance(duration, (int, float)) and duration >= 0:
        timing["duration_ms"] = int(duration)

    return {
        "schema_version": 1,
        "event_id": str(uuid.uuid4()),
        "observed_at": _utc_now(),
        "producer_id": f"hermes-agent:{os.getpid()}",
        "process_id": os.getpid(),
        "source_crate": "hermes-agent",
        "adapter_id": "ares.lifecycle.v1",
        "provenance": "adapted",
        "correlation": _correlation(kwargs),
        "producer_sequence": sequence,
        "kind": kind,
        "status": status,
        "timing": timing,
        "privacy": {
            "tier": "MetadataOnly",
            "redaction": "ContentDisabled",
            "content_fields": 0,
        },
        "payload": payload,
    }


class _Producer:
    def __init__(self, path: Path, capacity: int) -> None:
        self.path = path
        self.queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max(1, capacity))
        self.stop = threading.Event()
        self.sequence = 0
        self.dropped = 0
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run,
            name="hermes-stack-observation-producer",
            daemon=True,
        )
        self._thread.start()

    def next_sequence(self) -> int:
        """Allocate the next producer sequence atomically.

        Hooks fire from multiple threads (gateway workers, tool executors);
        a bare ``+= 1`` across threads can assign the same sequence twice,
        which the collector then records as colliding producer identities.
        """
        with self._lock:
            self.sequence += 1
            return self.sequence

    def emit(self, event: dict[str, Any]) -> None:
        try:
            self.queue.put_nowait(event)
        except queue.Full:
            with self._lock:
                self.dropped += 1

    def flush(self, timeout: float = 1.0) -> None:
        if timeout <= 0:
            return
        finished = threading.Event()

        def wait_for_queue() -> None:
            self.queue.join()
            finished.set()

        waiter = threading.Thread(target=wait_for_queue, daemon=True)
        waiter.start()
        finished.wait(timeout)

    def _run(self) -> None:
        stream: socket.socket | None = None
        while not self.stop.is_set() or not self.queue.empty():
            try:
                event = self.queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                if stream is None:
                    stream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    stream.settimeout(0.5)
                    stream.connect(str(self.path))
                payload = json.dumps(event, separators=(",", ":"), ensure_ascii=True).encode()
                stream.sendall(struct.pack(">I", len(payload)) + payload)
            except Exception:
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
                stream = None
                with self._lock:
                    self.dropped += 1
            finally:
                self.queue.task_done()
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass

    def close(self) -> None:
        self.stop.set()
        self._thread.join(timeout=1.0)


_PRODUCER: _Producer | None = None
_PRODUCER_LOCK = threading.Lock()
_OPEN_EVENTS: dict[tuple[str, str, str], dict[str, Any]] = {}
_SESSION_COUNTS: dict[str, dict[str, int]] = {}


def _producer() -> _Producer | None:
    global _PRODUCER
    config = _config()
    if not _enabled(config):
        return None
    path = _socket_path(config)
    if not path.exists():
        return None
    if _PRODUCER is not None and _PRODUCER.path == path:
        return _PRODUCER
    with _PRODUCER_LOCK:
        if _PRODUCER is None or _PRODUCER.path != path:
            if _PRODUCER is not None:
                _PRODUCER.close()
            capacity = config.get("capacity", _DEFAULT_CAPACITY)
            try:
                capacity = int(capacity)
            except (TypeError, ValueError):
                capacity = _DEFAULT_CAPACITY
            _PRODUCER = _Producer(path, capacity)
    return _PRODUCER


def observe_lifecycle(hook_name: str, **kwargs: Any) -> None:
    if hook_name not in _HANDLED_HOOKS:
        return
    producer = _producer()
    if producer is None:
        return
    session_id = _bounded(kwargs.get("session_id")) or ""
    request_id = _bounded(kwargs.get("api_request_id") or kwargs.get("tool_call_id")) or ""
    if hook_name in {"pre_api_request", "pre_tool_call"} and request_id:
        kind, _ = _kind_status(hook_name, kwargs)
        counts = _SESSION_COUNTS.setdefault(
            session_id,
            {"started_llm": 0, "terminal_llm": 0, "started_tool": 0, "terminal_tool": 0},
        )
        counts[f"started_{'llm' if kind == 'llm_call' else 'tool'}"] += 1
        _OPEN_EVENTS[(session_id, request_id, kind)] = {
            "session_id": session_id,
            "request_id": request_id,
            "gap_kind": kind,
            "missing_terminal_hook": (
                "post_api_request" if kind == "llm_call" else "post_tool_call"
            ),
        }
    elif hook_name in {"post_api_request", "api_request_error", "post_tool_call"} and request_id:
        kind, _ = _kind_status(hook_name, kwargs)
        counts = _SESSION_COUNTS.setdefault(
            session_id,
            {"started_llm": 0, "terminal_llm": 0, "started_tool": 0, "terminal_tool": 0},
        )
        counts[f"terminal_{'llm' if kind == 'llm_call' else 'tool'}"] += 1
        _OPEN_EVENTS.pop((session_id, request_id, kind), None)
    if hook_name == "on_session_end":
        kwargs = dict(kwargs)
        kwargs["coverage"] = _SESSION_COUNTS.pop(
            session_id,
            {"started_llm": 0, "terminal_llm": 0, "started_tool": 0, "terminal_tool": 0},
        )
        pending_items = [
            (key, pending)
            for key, pending in _OPEN_EVENTS.items()
            if key[0] == session_id
        ]
        if not pending_items:
            # Some legacy/transport paths finalize without forwarding the
            # session identity on every hook. Do not lose a known open event;
            # close it as an explicitly unmatched terminal observation.
            pending_items = list(_OPEN_EVENTS.items())
        for key, pending in pending_items:
            producer.emit(
                build_envelope(
                    "terminal_observation_gap",
                    pending,
                    sequence=producer.next_sequence(),
                )
            )
            _OPEN_EVENTS.pop(key, None)
    producer.emit(
        build_envelope(
            hook_name,
            kwargs,
            sequence=producer.next_sequence(),
        )
    )
    if hook_name == "on_session_end":
        producer.flush()


def handles_hook(hook_name: str) -> bool:
    # Hook admission follows the enabled policy, not a point-in-time socket
    # probe. The collector may start/restart after the agent does; suppressing
    # the lifecycle hook when the socket is briefly absent loses the event
    # entirely instead of recording a bounded drop.
    return hook_name in _HANDLED_HOOKS and _enabled(_config())


def shutdown() -> None:
    global _PRODUCER
    with _PRODUCER_LOCK:
        if _PRODUCER is not None:
            for key, pending in list(_OPEN_EVENTS.items()):
                _PRODUCER.emit(
                    build_envelope(
                        "terminal_observation_gap",
                        pending,
                        sequence=_PRODUCER.next_sequence(),
                    )
                )
                _OPEN_EVENTS.pop(key, None)
            _PRODUCER.flush()
            _PRODUCER.close()
            _PRODUCER = None


atexit.register(shutdown)

__all__ = ["build_envelope", "handles_hook", "observe_lifecycle", "shutdown"]
