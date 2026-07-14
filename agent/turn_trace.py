"""Per-turn waterfall tracing.

Records timing spans for everything hermes does around the LLM call in a turn
(gateway ingress, prologue, context assembly, LLM calls, tool dispatch,
finalize, delivery) and emits ONE JSON line per turn to a rotating
``~/.hermes/logs/turn_traces.jsonl`` sink, so turns can be rendered as a
waterfall and hermes-added overhead separated from model inference time.

Activation: ``HERMES_TURN_TRACE=1`` (also ``true``/``on``). When disabled every
entry point is a cheap no-op. Sink override: ``HERMES_TURN_TRACE_FILE``.

Threading model: a turn crosses threads (gateway asyncio loop -> run_sync
worker -> per-LLM-call worker -> concurrent tool executors), so the trace is
carried explicitly — instrumentation sites bind the trace to the object they
already share (the agent instance) via :func:`bind` / :func:`get_bound`, with a
thread-local *current* as a convenience for same-thread nesting. Spans store
absolute wall-clock start/end; parent/child nesting is derived at render time
from interval containment, so cross-thread and overlapping (concurrent tool
batch) spans need no stack bookkeeping.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

_TRUTHY = ("1", "true", "on", "yes")

# Single rotation keeps the sink bounded without a logging-handler dependency.
_MAX_SINK_BYTES = 64 * 1024 * 1024

_write_lock = threading.Lock()
_tls = threading.local()


def enabled() -> bool:
    return os.environ.get("HERMES_TURN_TRACE", "").strip().lower() in _TRUTHY


def sink_path() -> str:
    override = os.environ.get("HERMES_TURN_TRACE_FILE", "").strip()
    if override:
        return os.path.expanduser(override)
    return os.path.join(os.path.expanduser("~"), ".hermes", "logs", "turn_traces.jsonl")


class Span:
    __slots__ = ("name", "start", "end", "thread", "tags")

    def __init__(self, name: str, start: float, end: float, thread: str, tags: Dict[str, Any]):
        self.name = name
        self.start = start
        self.end = end
        self.thread = thread
        self.tags = tags

    def to_wire(self, t0: float) -> Dict[str, Any]:
        wire: Dict[str, Any] = {
            "n": self.name,
            "t0": round((self.start - t0) * 1000.0, 2),
            "d": round((self.end - self.start) * 1000.0, 2),
            "th": self.thread,
        }
        if self.tags:
            wire["tags"] = self.tags
        return wire


class _SpanHandle:
    """Context manager for an in-flight span; ``tag()`` adds attributes late."""

    __slots__ = ("_trace", "name", "tags", "_start")

    def __init__(self, trace: "TurnTrace", name: str, tags: Dict[str, Any]):
        self._trace = trace
        self.name = name
        self.tags = tags
        self._start = time.time()

    def tag(self, **tags: Any) -> "_SpanHandle":
        self.tags.update(tags)
        return self

    def __enter__(self) -> "_SpanHandle":
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.tags.setdefault("error", exc_type.__name__)
        self._trace.add_span(self.name, self._start, time.time(), **self.tags)


class _NullSpan:
    __slots__ = ()

    def tag(self, **tags: Any) -> "_NullSpan":
        return self

    def __enter__(self) -> "_NullSpan":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


_NULL_SPAN = _NullSpan()


class TurnTrace:
    """Mutable collector for one turn; thread-safe span appends."""

    def __init__(self, key: Optional[str] = None, started_at: Optional[float] = None, **tags: Any):
        self.trace_id = uuid.uuid4().hex[:12]
        self.key = key or self.trace_id
        self.started_at = started_at if started_at is not None else time.time()
        self.tags: Dict[str, Any] = dict(tags)
        self._spans: List[Span] = []
        self._lock = threading.Lock()
        self._finished = False

    def span(self, name: str, **tags: Any) -> _SpanHandle:
        return _SpanHandle(self, name, tags)

    def add_span(self, name: str, start: float, end: float, **tags: Any) -> None:
        """Retrofit a span from timings measured elsewhere (e.g. api_duration)."""
        s = Span(name, start, end, threading.current_thread().name, tags)
        with self._lock:
            self._spans.append(s)

    def mark(self, name: str, at: Optional[float] = None, **tags: Any) -> None:
        """Point-in-time event (zero-duration span)."""
        ts = at if at is not None else time.time()
        self.add_span(name, ts, ts, **tags)

    def tag(self, **tags: Any) -> None:
        with self._lock:
            self.tags.update(tags)

    def finish(self, **tags: Any) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self._finished:
                return None
            self._finished = True
            self.tags.update(tags)
            spans = list(self._spans)
            # Snapshot: another thread (e.g. the run_sync executor after a
            # cancel) can still tag() while json.dumps iterates the record.
            final_tags = dict(self.tags)
        ended_at = time.time()
        record = {
            "schema": 1,
            "trace_id": self.trace_id,
            "key": self.key,
            "started_at": round(self.started_at, 3),
            "duration_ms": round((ended_at - self.started_at) * 1000.0, 2),
            "tags": final_tags,
            "spans": [s.to_wire(self.started_at) for s in sorted(spans, key=lambda s: s.start)],
        }
        _emit(record)
        return record


def _emit(record: Dict[str, Any]) -> None:
    try:
        path = sink_path()
        data = (json.dumps(record, ensure_ascii=False, default=str) + "\n").encode("utf-8")
        with _write_lock:
            parent = os.path.dirname(path)
            if parent:  # bare relative sink filename: dirname is "" — append to CWD
                os.makedirs(parent, exist_ok=True)
            try:
                if os.path.getsize(path) > _MAX_SINK_BYTES:
                    os.replace(path, path + ".1")
            except OSError:
                pass
            # Single O_APPEND write: atomic per record even when the gateway
            # daemon and a CLI run share the default sink.
            fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                os.write(fd, data)
            finally:
                os.close(fd)
    except Exception:
        # Tracing must never break a turn.
        pass


# --- carrying the trace ------------------------------------------------------

_BIND_ATTR = "_hermes_turn_trace"


def begin(key: Optional[str] = None, started_at: Optional[float] = None, **tags: Any) -> Optional[TurnTrace]:
    """Start a trace (returns None when tracing is disabled)."""
    if not enabled():
        return None
    trace = TurnTrace(key=key, started_at=started_at, **tags)
    adopt(trace)
    return trace


def adopt(trace: Optional[TurnTrace]) -> None:
    """Make ``trace`` current for THIS thread (worker-thread entry points)."""
    _tls.trace = trace


def current() -> Optional[TurnTrace]:
    return getattr(_tls, "trace", None)


def bind(obj: Any, trace: Optional[TurnTrace]) -> None:
    """Attach the trace to a shared object (typically the agent instance)."""
    try:
        setattr(obj, _BIND_ATTR, trace)
    except Exception:
        pass


def get_bound(obj: Any) -> Optional[TurnTrace]:
    return getattr(obj, _BIND_ATTR, None)


def span(name: str, trace: Optional[TurnTrace] = None, obj: Any = None, **tags: Any):
    """No-op-safe span: resolves trace from arg, bound object, or thread-local."""
    t = trace or (get_bound(obj) if obj is not None else None) or current()
    if t is None or t._finished:
        return _NULL_SPAN
    return t.span(name, **tags)


def mark(name: str, trace: Optional[TurnTrace] = None, obj: Any = None, **tags: Any) -> None:
    t = trace or (get_bound(obj) if obj is not None else None) or current()
    if t is not None and not t._finished:
        t.mark(name, **tags)
