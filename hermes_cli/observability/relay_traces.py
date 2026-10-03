"""Execution traces recorded from Hermes's NeMo Relay events.

Relay already sees every Hermes session, turn, LLM attempt and tool call as ATOF scope events. This
module is one more Relay subscriber: it attributes each event to the Hermes session whose
``hermes.session`` scope it descends from, appends it (payload bounded, never reshaped) to
``<profile home>/traces/<session_id>.jsonl``, and fans it out to in-process listeners so a live
client can follow the run. A delegated child's session-scope events are also written to its
parent's file, which is how a reader finds the child's own file.

Relay only emits LLM and tool events for calls routed through its managed pipeline, which Hermes
enables only when a Relay plugin or shared metrics needs it. For every other run this module fills
the gap from Hermes's observer hooks with Relay's own manual span API (``llm.call`` / ``tools.call``),
parented under the live turn scope, so the recorded stream has the same shape either way.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent import relay_runtime

logger = logging.getLogger(__name__)

SUBSCRIBER_NAME = "hermes.traces"
TRACES_DIRNAME = "traces"
TURN_INPUT_MARK = "hermes.turn.input"

# Shared-metrics projections of the same calls; recording them would draw every call twice.
_METRICS_SCOPES = frozenset({"hermes.model_call", "hermes.task_run", "hermes.tool_call"})
_OWNER_CAP = 100_000
_STRING_CAP = 4_000
_LIST_CAP = 100
_DEPTH_CAP = 8
_PREVIEW_CAP = 500
_DEFAULT_RETENTION_DAYS = 30
_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]")

TraceListener = Callable[[str, str, str, dict[str, Any]], None]
"""``(profile_key, root_session_id, session_id, event)`` for every recorded event."""


@dataclass(frozen=True)
class _Owner:
    home: Path
    profile_key: str
    session_id: str
    root_session_id: str
    parent_session_id: str = ""


# ── payload bounds ─────────────────────────────────────────────────────────────────────────────


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _bound(value: Any, depth: int = 0) -> Any:
    """Cap strings, lists and nesting so one huge tool result or prompt cannot bloat a trace."""
    if isinstance(value, str):
        return value if len(value) <= _STRING_CAP else f"{value[:_STRING_CAP]}… [{len(value)} chars]"
    if depth >= _DEPTH_CAP:
        return "…" if isinstance(value, (dict, list, tuple)) else value
    if isinstance(value, dict):
        return {str(k): _bound(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        items = [_bound(v, depth + 1) for v in list(value)[:_LIST_CAP]]
        return items + ([f"… [{len(value)} items]"] if len(value) > _LIST_CAP else [])
    return value


def _slim_llm_request(data: Any) -> Any:
    """A managed LLM start carries the whole request; keep its model and last message only, or every
    attempt would re-store the entire conversation."""
    if not isinstance(data, dict) or not isinstance(data.get("content"), dict):
        return data
    content = dict(data["content"])
    for key in ("messages", "input"):
        items = content.get(key)
        if isinstance(items, list):
            content[key] = items[-1:]
            content.setdefault("message_count", len(items))
    if isinstance(content.get("tools"), list):
        content["tool_count"] = len(content.pop("tools"))
    content.pop("system", None)
    content.pop("instructions", None)
    return {"content": content}


def slim_event(event: dict[str, Any]) -> dict[str, Any]:
    """The recorded form of one ATOF event: same fields, bounded payloads."""
    slim = dict(event)
    if slim.get("category") == "llm" and slim.get("scope_category") == "start":
        slim["data"] = _slim_llm_request(slim.get("data"))
    for key in ("data", "metadata", "category_profile"):
        if slim.get(key) is not None:
            slim[key] = _bound(slim[key])
    slim.pop("data_schema", None)
    return slim


# ── storage ───────────────────────────────────────────────────────────────────────────────────


def traces_dir(home: Path) -> Path:
    return Path(home) / TRACES_DIRNAME


def trace_path(home: Path, session_id: str) -> Path:
    return traces_dir(home) / f"{_SAFE_ID.sub('_', session_id)}.jsonl"


def _read_file(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return []
    events = []
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue  # a torn final line from a crash mid-append
        if isinstance(event, dict):
            events.append(event)
    return events


def _child_session_ids(events: Iterable[dict[str, Any]], session_id: str) -> list[str]:
    children: dict[str, None] = {}
    for event in events:
        metadata = event.get("metadata")
        if event.get("name") == relay_runtime.SESSION_SCOPE and isinstance(metadata, dict):
            child = str(metadata.get(relay_runtime.SESSION_ID_KEY) or "")
            if child and child != session_id:
                children[child] = None
    return list(children)


def read_session_events(home: Path, session_id: str, *, limit: int = 50_000) -> tuple[list[dict[str, Any]], bool]:
    """Recorded events for ``session_id`` and every delegated descendant, oldest first.

    Returns ``(events, truncated)``; past ``limit`` the newest events are kept.
    """
    seen: set[str] = set()
    events: dict[tuple[str, str], dict[str, Any]] = {}
    pending = [session_id]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        recorded = _read_file(trace_path(home, current))
        for event in recorded:
            # A child's session scope sits in both files: keep one copy per (uuid, phase).
            key = (str(event.get("uuid") or ""), str(event.get("scope_category") or event.get("kind") or ""))
            events.setdefault(key, event)
        pending.extend(_child_session_ids(recorded, current))
    ordered = sorted(events.values(), key=lambda e: str(e.get("timestamp") or ""))
    return (ordered[-limit:], True) if len(ordered) > limit else (ordered, False)


def _prune(home: Path, retention_days: float) -> None:
    if retention_days <= 0:
        return
    cutoff = time.time() - retention_days * 86_400
    try:
        entries = list(traces_dir(home).iterdir())
    except OSError:
        return
    for entry in entries:
        try:
            if entry.suffix == ".jsonl" and entry.stat().st_mtime < cutoff:
                entry.unlink()
        except OSError:
            logger.debug("Unable to prune trace file %s", entry, exc_info=True)


# ── the subscriber ────────────────────────────────────────────────────────────────────────────


class _Recorder:
    """Process-wide Relay subscriber; hosts (one per profile) attach as their first session opens."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._homes: dict[str, tuple[Path, str]] = {}  # runtime_id -> (home, profile_key)
        self._owners: OrderedDict[str, _Owner] = OrderedDict()
        self._listeners: list[TraceListener] = []
        self._registered_relays: dict[int, Any] = {}

    def attach(self, host: relay_runtime.RelayRuntime, retention_days: float) -> None:
        with self._lock:
            if host.runtime_id in self._homes:
                return
            home = Path(host.profile_key)
            self._homes[host.runtime_id] = (home, host.profile_key)
            if id(host.relay) not in self._registered_relays:
                host.relay.subscribers.register(SUBSCRIBER_NAME, self)
                self._registered_relays[id(host.relay)] = host.relay
        _prune(home, retention_days)

    def attached(self, host: Any) -> bool:
        return getattr(host, "runtime_id", None) in self._homes

    @property
    def recording(self) -> bool:
        """Any profile attached: the per-token stream path's cheap early out."""
        return bool(self._homes)

    def add_listener(self, listener: TraceListener) -> Callable[[], None]:
        with self._lock:
            self._listeners.append(listener)

        def remove() -> None:
            with self._lock:
                if listener in self._listeners:
                    self._listeners.remove(listener)

        return remove

    def _remember(self, uuid: str, owner: _Owner) -> None:
        self._owners[uuid] = owner
        self._owners.move_to_end(uuid)
        while len(self._owners) > _OWNER_CAP:
            self._owners.popitem(last=False)

    def _attribute(self, event: dict[str, Any]) -> tuple[_Owner, list[str]] | None:
        """The owning session and the session files this event belongs in."""
        uuid, parent_uuid = str(event.get("uuid") or ""), str(event.get("parent_uuid") or "")
        metadata = _as_dict(event.get("metadata"))
        is_start = event.get("kind") == "scope" and event.get("scope_category") == "start"
        is_end = event.get("kind") == "scope" and event.get("scope_category") == "end"
        with self._lock:
            parent = self._owners.get(parent_uuid)
            if event.get("name") == relay_runtime.SESSION_SCOPE and metadata.get(relay_runtime.SESSION_ID_KEY):
                if is_end and uuid in self._owners:
                    owner = self._owners[uuid]
                else:
                    home = self._homes.get(str(metadata.get(relay_runtime.RUNTIME_INSTANCE_KEY) or ""))
                    if home is None:
                        return None
                    session_id = str(metadata[relay_runtime.SESSION_ID_KEY])
                    owner = _Owner(
                        home[0], home[1], session_id, parent.root_session_id if parent else session_id,
                        str(metadata.get(relay_runtime.PARENT_SESSION_ID_KEY) or ""),
                    )
                    self._remember(uuid, owner)
                files = [owner.session_id]
                if parent is not None and parent.session_id != owner.session_id:
                    files.append(parent.session_id)
                return owner, files
            owner = self._owners.get(uuid) if is_end else parent
            if owner is None:
                return None
            if is_start:
                self._remember(uuid, owner)
            return owner, [owner.session_id]

    def __call__(self, relay_event: Any) -> None:
        try:
            event = relay_event.to_dict()
            if event.get("name") in _METRICS_SCOPES:
                return
            attributed = self._attribute(event)
            if attributed is None:
                return
            owner, files = attributed
            recorded = slim_event(event)
            line = json.dumps(recorded, ensure_ascii=False, separators=(",", ":"), default=str) + "\n"
            with self._write_lock:
                traces_dir(owner.home).mkdir(parents=True, exist_ok=True)
                for session_id in files:
                    with trace_path(owner.home, session_id).open("a", encoding="utf-8") as handle:
                        handle.write(line)
            with self._lock:
                listeners = list(self._listeners)
        except Exception:
            logger.debug("Hermes trace recording failed", exc_info=True)
            return
        for listener in listeners:
            try:
                listener(owner.profile_key, owner.root_session_id, owner.session_id, recorded)
            except Exception:
                logger.debug("Hermes trace listener failed", exc_info=True)

    def reset_for_tests(self) -> None:
        with self._lock:
            for relay in self._registered_relays.values():
                relay.subscribers.deregister(SUBSCRIBER_NAME)
            self._registered_relays.clear()
            self._homes.clear()
            self._owners.clear()
            self._listeners.clear()


RECORDER = _Recorder()


def add_listener(listener: TraceListener) -> Callable[[], None]:
    """Receive every recorded event live; returns the unsubscribe callable."""
    return RECORDER.add_listener(listener)


# ── policy ────────────────────────────────────────────────────────────────────────────────────


def policy() -> dict[str, Any]:
    """This profile's ``telemetry.traces`` block (read-only snapshot)."""
    from hermes_cli.config import read_raw_config_readonly

    config: Any = read_raw_config_readonly() or {}
    for key in ("telemetry", "traces"):
        config = config.get(key) if isinstance(config, dict) else None
    return config if isinstance(config, dict) else {}


def _prepare_session(host: relay_runtime.RelayRuntime, context: dict[str, Any]) -> None:
    """Attach the recorder before the coordinator pushes the session scope it must observe."""
    del context
    if host.profile_key != relay_runtime.current_profile_key():
        return
    config = policy()
    if config.get("enabled", True) is False:
        return
    try:
        retention = float(config.get("retention_days", _DEFAULT_RETENTION_DAYS))
    except (TypeError, ValueError):
        retention = float(_DEFAULT_RETENTION_DAYS)
    RECORDER.attach(host, retention)


relay_runtime.SESSION_COORDINATOR.register_session_initializer(SUBSCRIBER_NAME, _prepare_session)


# ── hook-driven spans for unmanaged runs ───────────────────────────────────────────────────────


def _timestamp(epoch: Any) -> datetime:
    value = epoch if isinstance(epoch, (int, float)) and not isinstance(epoch, bool) else time.time()
    return datetime.fromtimestamp(value, tz=timezone.utc)


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):  # content parts
        return " ".join(str(p.get("text") or "") for p in value if isinstance(p, dict)).strip()
    return ""


def _preview(value: Any) -> str:
    text = _text(value).strip()
    return text if len(text) <= _PREVIEW_CAP else f"{text[:_PREVIEW_CAP]}…"


def _jsonish(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    try:
        json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)
    return json.loads(json.dumps(value, default=str))


class _SpanBridge:
    """Open manual Relay spans keyed by the hook correlation ids that close them."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._open: dict[tuple[str, str, str], tuple[relay_runtime.RelayRuntime, Any, Any]] = {}

    @staticmethod
    def _host() -> relay_runtime.RelayRuntime | None:
        host = relay_runtime.get_runtime(create=False)
        if host is None or not RECORDER.attached(host) or host.managed_execution_enabled():
            return None
        return host

    @staticmethod
    def _parent(host: relay_runtime.RelayRuntime, session_id: str) -> tuple[Any, Any] | None:
        session = host.get_session(session_id)
        if session is None:
            return None
        turn = relay_runtime.active_turn(session_id)
        if turn is not None and turn.handle is not None and turn.lease.host is host:
            return session, turn.handle
        return session, session.handle

    def start(self, key: tuple[str, str, str], open_span: Callable[[Any, Any], Any]) -> None:
        host = self._host()
        parent = None if host is None or not key[2] else self._parent(host, key[0])
        if host is None or parent is None:
            return
        session, handle = parent
        span = host.run_in_session(session, open_span, host.relay, handle)
        with self._lock:
            self._open[key] = (host, session, span)

    def finish(self, key: tuple[str, str, str], close_span: Callable[[Any, Any], None]) -> None:
        with self._lock:
            opened = self._open.pop(key, None)
        if opened is not None:
            host, session, span = opened
            host.run_in_session(session, close_span, host.relay, span, allow_closing=True)

    def opened(self, key: tuple[str, str, str]) -> tuple[relay_runtime.RelayRuntime, Any, Any] | None:
        with self._lock:
            return self._open.get(key)

    def abandon(self, session_id: str) -> None:
        """Close spans whose closing hook never fired (a vetoed tool, an interrupted call)."""
        with self._lock:
            keys = [key for key in self._open if key[0] == session_id]
        for key in keys:
            if key[1] == "tool":
                self.finish(key, lambda relay, span: relay.tools.call_end(
                    span, relay.ToolExecutionResult(None), metadata={"hermes.status": "abandoned"}))
            else:
                self.finish(key, lambda relay, span: relay.llm.call_end(
                    span, None, metadata={"hermes.status": "abandoned"}))

    def reset_for_tests(self) -> None:
        with self._lock:
            self._open.clear()


SPANS = _SpanBridge()


# ── streaming ─────────────────────────────────────────────────────────────────────────────────

STREAM_FIRST_TOKEN_MARK = "hermes.llm.first_token"
STREAM_MARK = "hermes.llm.stream"
_STREAM_FLUSH_SECONDS = 0.5


@dataclass
class _Stream:
    started: bool = False
    kind: str = ""
    parts: list[str] = field(default_factory=list)
    flushed_at: float = 0.0


class _StreamMarks:
    """Relay has no in-call streaming events, so a model call's progress is recorded as Relay marks
    under its span: one ``hermes.llm.first_token``, then ``hermes.llm.stream`` chunks of the text or
    reasoning streamed since the last mark (at most every ``_STREAM_FLUSH_SECONDS``, and whenever
    the stream switches between reasoning and text)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._streams: dict[tuple[str, str], _Stream] = {}

    @staticmethod
    def _take(stream: _Stream) -> tuple[str, dict[str, Any]] | None:
        text, stream.parts = "".join(stream.parts), []
        stream.flushed_at = time.monotonic()
        return (STREAM_MARK, {"kind": stream.kind, "text": text}) if text else None

    def delta(self, session_id: str, request_id: str, kind: str, text: str) -> None:
        if not text or not request_id:
            return
        marks: list[tuple[str, dict[str, Any]]] = []
        with self._lock:
            stream = self._streams.setdefault((session_id, request_id), _Stream())
            if not stream.started:
                stream.started, stream.flushed_at = True, time.monotonic()
                marks.append((STREAM_FIRST_TOKEN_MARK, {"kind": kind}))
            if stream.kind and stream.kind != kind and (mark := self._take(stream)):
                marks.append(mark)
            stream.kind = kind
            stream.parts.append(text)
            if time.monotonic() - stream.flushed_at >= _STREAM_FLUSH_SECONDS and (mark := self._take(stream)):
                marks.append(mark)
        for name, data in marks:
            _mark_llm(session_id, request_id, name, data)

    def finish(self, session_id: str, request_id: str) -> None:
        with self._lock:
            stream = self._streams.pop((session_id, request_id), None)
            mark = self._take(stream) if stream is not None else None
        if mark is not None:
            _mark_llm(session_id, request_id, *mark)

    def abandon(self, session_id: str) -> None:
        with self._lock:
            for key in [key for key in self._streams if key[0] == session_id]:
                self._streams.pop(key, None)

    def reset_for_tests(self) -> None:
        with self._lock:
            self._streams.clear()


STREAMS = _StreamMarks()


def _mark_llm(session_id: str, request_id: str, name: str, data: dict[str, Any]) -> None:
    """Mark the model call ``request_id``. Relay marks attach only to scopes, never to an LLM call
    handle, so the mark sits on the scope the call runs in (the managed pipeline's logical call
    scope, else the turn) and names the call: ``llm_uuid`` for a manual span, ``api_request_id``
    always."""
    host = relay_runtime.get_runtime(create=False)
    turn = relay_runtime.active_turn(session_id)
    session = None if host is None else host.get_session(session_id)
    if host is None or session is None or turn is None or turn.lease.host is not host or not RECORDER.attached(host):
        return
    handle = turn.logical_llm_calls.get(request_id) or turn.handle
    if handle is None:
        return
    opened = SPANS.opened((session_id, "llm", request_id))
    call = {"api_request_id": request_id, **({"llm_uuid": str(opened[2].uuid)} if opened is not None else {})}
    host.run_in_session(session, host.relay.scope.event, name, handle=handle, data=_bound({**data, **call}))


def note_stream_delta(session_id: str, request_id: str, kind: str, text: str) -> None:
    """Record streamed model output for the trace; called from the agent's stream path per delta."""
    if RECORDER.recording:
        try:
            STREAMS.delta(session_id, request_id, kind, text)
        except Exception:
            logger.debug("Hermes trace stream mark failed", exc_info=True)


def _llm_key(kw: dict[str, Any]) -> tuple[str, str, str]:
    # An auxiliary attempt may carry no request id; its pre/post pair shares ``started_at``.
    request_id = str(kw.get("api_request_id") or "")
    if not request_id and kw.get("aux_task"):
        request_id = f"{kw['aux_task']}:{kw.get('started_at')}"
    return str(kw.get("session_id") or ""), "llm", request_id


def _tool_key(kw: dict[str, Any]) -> tuple[str, str, str]:
    call_id = str(kw.get("tool_call_id") or "")
    return str(kw.get("session_id") or ""), "tool", call_id or f"{kw.get('tool_name')}:{kw.get('api_request_id')}"


def _start_llm(kw: dict[str, Any]) -> None:
    from agent.relay_llm import _relay_operation_name

    provider, model = str(kw.get("provider") or "provider"), str(kw.get("model") or "")
    messages = _as_list(kw.get("request_messages"))
    body = {
        "model": model,
        "messages": _bound(_jsonish(messages[-1:])),
        "message_count": kw.get("message_count", len(messages)),
        "tool_count": kw.get("tool_count"),
        "approx_input_tokens": kw.get("approx_input_tokens"),
    }
    metadata = {
        "hermes.provider": provider, "hermes.api_mode": kw.get("api_mode"),
        "hermes.api_request_id": kw.get("api_request_id"), "hermes.aux_task": kw.get("aux_task"),
    }

    def open_span(relay: Any, parent: Any) -> Any:
        return relay.llm.call(
            _relay_operation_name(provider, {"api_mode": kw.get("api_mode")}), relay.LLMRequest({}, body),
            handle=parent, model_name=model, timestamp=_timestamp(kw.get("started_at")),
            metadata={k: v for k, v in metadata.items() if v not in (None, "")},
        )

    SPANS.start(_llm_key(kw), open_span)


_RELAY_FINISH_REASONS = {
    "stop": "complete", "end_turn": "complete", "completed": "complete", "length": "length",
    "max_tokens": "length", "tool_calls": "tool_use", "tool_use": "tool_use", "function_call": "tool_use",
    "content_filter": "content_filter",
}


def _relay_usage(usage: dict[str, Any] | None) -> dict[str, int]:
    """Hermes's canonical usage in Relay's normalized annotation keys."""
    if not usage:
        return {}
    prompt = usage.get("prompt_tokens") or usage.get("input_tokens")
    completion = usage.get("output_tokens") or usage.get("completion_tokens")
    mapped = {
        "prompt_tokens": prompt, "completion_tokens": completion,
        "total_tokens": usage.get("total_tokens"), "cache_read_tokens": usage.get("cache_read_tokens"),
    }
    return {k: int(v) for k, v in mapped.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def _finish_llm(kw: dict[str, Any]) -> None:
    error = kw.get("error")
    response = _as_dict(kw.get("response"))
    message = _as_dict(response.get("assistant_message"))
    tool_calls = [
        str(_as_dict(call.get("function")).get("name") or call.get("name") or "")
        for call in _as_list(message.get("tool_calls")) if isinstance(call, dict)
    ]
    usage = _as_dict(kw.get("usage")) or None
    data = _bound({
        "model": kw.get("response_model") or response.get("model"),
        "finish_reason": kw.get("finish_reason"),
        "content": _preview(message.get("content")),
        "tool_calls": [name for name in tool_calls if name],
        "usage": usage,
        "first_chunk_at": kw.get("first_chunk_at"),
        "error": _jsonish(error) if error else None,
    })
    status = {"otel.status_code": "ERROR" if error else "OK"}
    annotation: dict[str, Any] = {"message": data.get("content") or "", "usage": _relay_usage(usage)}
    if data.get("model"):
        annotation["model"] = str(data["model"])
    if data.get("finish_reason"):
        annotation["finish_reason"] = _RELAY_FINISH_REASONS.get(str(data["finish_reason"]), "unknown")

    def close_span(relay: Any, span: Any) -> None:
        response_data = {k: v for k, v in data.items() if v not in (None, "", [])}
        try:
            relay.llm.call_end(span, response_data, metadata=status, annotated_response=annotation,
                               timestamp=_timestamp(kw.get("ended_at")))
        except ValueError:  # Relay rejected the annotation; the span still has to close
            relay.llm.call_end(span, response_data, metadata=status, timestamp=_timestamp(kw.get("ended_at")))

    session_id, _kind, request_id = _llm_key(kw)
    STREAMS.finish(session_id, request_id)  # the streamed tail lands inside the call it belongs to
    SPANS.finish(_llm_key(kw), close_span)


def _start_tool(kw: dict[str, Any]) -> None:
    name, args = str(kw.get("tool_name") or "tool"), _bound(_jsonish(kw.get("args") or {}))
    call_id = str(kw.get("tool_call_id") or "") or None

    def open_span(relay: Any, parent: Any) -> Any:
        return relay.tools.call(name, args, handle=parent, tool_call_id=call_id, timestamp=_timestamp(None))

    SPANS.start(_tool_key(kw), open_span)


def _finish_tool(kw: dict[str, Any]) -> None:
    status = str(kw.get("status") or "")
    failed = status in {"error", "failed", "blocked"} or bool(kw.get("error_type"))
    result = _bound(_jsonish(kw.get("result")))
    metadata = {
        "otel.status_code": "ERROR" if failed else "OK", "hermes.status": status or None,
        "hermes.error_type": kw.get("error_type"), "hermes.error_message": _preview(kw.get("error_message")) or None,
    }

    def close_span(relay: Any, span: Any) -> None:
        relay.tools.call_end(
            span, relay.ToolExecutionResult(result), timestamp=_timestamp(None),
            metadata={k: v for k, v in metadata.items() if v is not None},
        )

    SPANS.finish(_tool_key(kw), close_span)


def _mark_turn_input(kw: dict[str, Any]) -> None:
    """Name the turn: Relay's turn scope carries no prompt, and an unmanaged run has no request body."""
    session_id, preview = str(kw.get("session_id") or ""), _preview(kw.get("user_message"))
    host = relay_runtime.get_runtime(create=False)
    turn = relay_runtime.active_turn(session_id)
    if not preview or host is None or not RECORDER.attached(host) or turn is None or turn.handle is None:
        return
    session = host.get_session(session_id)
    if session is not None:
        host.run_in_session(session, host.relay.scope.event, TURN_INPUT_MARK, handle=turn.handle,
                            data={"preview": preview})


_HOOK_HANDLERS: dict[str, Callable[[dict[str, Any]], None]] = {
    "pre_llm_call": _mark_turn_input,
    "pre_api_request": _start_llm,
    "post_api_request": _finish_llm,
    "api_request_error": _finish_llm,
    "pre_auxiliary_call": _start_llm,
    "post_auxiliary_call": _finish_llm,
    "pre_tool_call": _start_tool,
    "post_tool_call": _finish_tool,
    "on_session_end": lambda kw: _abandon(str(kw.get("session_id") or "")),
}
HANDLED_HOOKS = frozenset(_HOOK_HANDLERS)


def _abandon(session_id: str) -> None:
    STREAMS.abandon(session_id)
    SPANS.abandon(session_id)


def handles_hook(hook_name: str) -> bool:
    if hook_name not in HANDLED_HOOKS:
        return False
    host = relay_runtime.get_runtime(create=False)
    return host is not None and RECORDER.attached(host)


def observe_lifecycle(hook_name: str, **kwargs: Any) -> None:
    handler = _HOOK_HANDLERS.get(hook_name)
    if handler is None or not relay_runtime.relay_instrumentation_enabled():
        return
    try:
        handler(kwargs)
    except Exception:
        logger.debug("Hermes trace hook failed: %s", hook_name, exc_info=True)


def _reset_for_tests() -> None:
    RECORDER.reset_for_tests()
    SPANS.reset_for_tests()
    STREAMS.reset_for_tests()
