"""Derive OpenTelemetry-style traces from the Hermes session store.

Hermes already persists everything a trace needs: ``sessions`` rows carry
server-side ``started_at`` / ``ended_at`` and full token accounting, and
``messages`` rows carry a server-side ``timestamp`` plus ``tool_calls`` (the
OpenAI tool-call JSON) and ``tool_call_id`` so a tool call can be paired with
its result. Every subagent is itself a session linked by
``parent_session_id``. That means a complete, accurately-timed span tree can be
reconstructed for *any* session — historical or live — with zero extra
instrumentation.

This module is the read-side "derive-on-read" trace builder. It turns a session
(and its subagent descendants) into a provider-neutral :class:`Trace` of
:class:`Span` objects. ``agent/trace_export.py`` renders that into OTLP/JSON
(OpenInference conventions, ingestible by Arize Phoenix / any OTel backend) or
the Chrome Trace Event format (viewable in https://ui.perfetto.dev).

Accuracy note: the Hermes agent loop runs tool calls sequentially, so inferring
span durations from consecutive message timestamps matches real execution. The
only inferred link is a ``delegate_task`` tool call → its child session, matched
by start-time proximity; a future precision pass can persist the spawning
``tool_call_id`` to make that exact.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)

# OpenInference span kinds (what Phoenix and other OTel-GenAI viewers expect).
KIND_AGENT = "AGENT"
KIND_LLM = "LLM"
KIND_TOOL = "TOOL"
KIND_CHAIN = "CHAIN"

# Status codes, mirroring OTLP (1 = OK, 2 = ERROR).
STATUS_OK = "ok"
STATUS_ERROR = "error"
STATUS_UNSET = "unset"

# Tool names that spawn subagent sessions. A span with one of these names gets
# its matched child session's subtree nested underneath it.
_DELEGATE_TOOL_NAMES = frozenset({"delegate_task"})

# How long after the last message a session's ``ended_at`` may sit and still be
# trusted as real activity (vs a cleanup/orphan reaper firing much later).
_END_GRACE_SECONDS = 300.0


class _SessionStore(Protocol):
    """The slice of ``SessionDB`` the builder depends on (keeps it testable)."""

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]: ...

    def get_messages(
        self, session_id: str, include_inactive: bool = False
    ) -> List[Dict[str, Any]]: ...

    def get_child_session_ids(self, parent_session_id: str) -> List[str]: ...


@dataclass
class Span:
    """A single unit of work on the trace timeline.

    Times are epoch seconds (float) to match ``messages.timestamp``. Exporters
    convert to their own units (OTLP nanoseconds, Chrome microseconds).
    """

    span_id: str
    parent_id: Optional[str]
    name: str
    kind: str
    start: float
    end: float
    status: str = STATUS_UNSET
    session_id: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "kind": self.kind,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "status": self.status,
            "session_id": self.session_id,
            "attributes": {k: v for k, v in self.attributes.items() if v is not None},
        }


@dataclass
class Trace:
    """A full span tree rooted at one session (plus its subagent descendants)."""

    trace_id: str
    root_session_id: str
    spans: List[Span] = field(default_factory=list)
    root_span_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def start(self) -> float:
        return min((s.start for s in self.spans), default=0.0)

    @property
    def end(self) -> float:
        return max((s.end for s in self.spans), default=0.0)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "root_session_id": self.root_session_id,
            "root_span_id": self.root_span_id,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "metadata": {k: v for k, v in self.metadata.items() if v is not None},
            "spans": [s.to_dict() for s in self.spans],
        }


# ── tool-call shape helpers ──────────────────────────────────────────────────


def _tool_call_id(call: Dict[str, Any]) -> str:
    return str(call.get("id") or call.get("tool_call_id") or "")


def _tool_call_name(call: Dict[str, Any]) -> str:
    fn = call.get("function")
    if isinstance(fn, dict) and fn.get("name"):
        return str(fn["name"])
    return str(call.get("name") or "tool")


def _tool_call_args(call: Dict[str, Any]) -> Any:
    fn = call.get("function")
    raw = fn.get("arguments") if isinstance(fn, dict) else call.get("arguments")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw
    return raw


def _as_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(content)


def _looks_like_error(message: Dict[str, Any]) -> bool:
    """Best-effort error detection on a tool-result message."""
    text = _as_text(message.get("content")).lstrip()
    if not text:
        return False
    head = text[:400].lower()
    if text.startswith("{"):
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                if obj.get("error") or obj.get("success") is False:
                    return True
                status = str(obj.get("status", "")).lower()
                if status in {"error", "failed", "failure"}:
                    return True
        except (json.JSONDecodeError, TypeError):
            pass
    return any(
        marker in head
        for marker in ("traceback (most recent call last)", "error:", "exception:")
    )


# ── builder ──────────────────────────────────────────────────────────────────


def _short(value: str, limit: int = 120) -> str:
    flat = " ".join(value.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _llm_span_name() -> str:
    """Label for an LLM (assistant-turn) span. A plain structural "llm" (matching
    the OTel/Langfuse convention of short, uniform LLM labels) — the model and
    the response text live in the span's attributes / detail panel, not the row.
    """
    return "llm"


def _mk_span_id(prefix: str, session_id: str, key: Any) -> str:
    return f"{prefix}:{session_id}:{key}"


def _trace_id(session_id: str) -> str:
    return f"trace:{session_id}"


def build_trace(
    store: _SessionStore,
    session_id: str,
    *,
    include_subagents: bool = True,
    _depth: int = 0,
    _max_depth: int = 8,
) -> Optional[Trace]:
    """Reconstruct a :class:`Trace` for ``session_id`` from the session store.

    Returns ``None`` when the session does not exist. Walks delegate subagent
    descendants (``parent_session_id``) and nests each under the
    ``delegate_task`` tool span that spawned it.
    """
    session = store.get_session(session_id)
    if not session:
        return None

    trace = Trace(
        trace_id=_trace_id(session_id),
        root_session_id=session_id,
        metadata={
            "source": session.get("source"),
            "model": session.get("model"),
            "cwd": session.get("cwd"),
            "git_branch": session.get("git_branch"),
        },
    )

    root_span = _build_session_spans(store, session, parent_span_id=None, trace=trace)
    trace.root_span_id = root_span.span_id if root_span else None

    if include_subagents and root_span and _depth < _max_depth:
        _attach_subagents(store, session_id, trace, _depth=_depth, _max_depth=_max_depth)

    trace.spans.sort(key=lambda s: (s.start, s.span_id))
    return trace


# Synthetic re-injections that re-enter the conversation as ``user`` messages
# but are CONTINUATIONS of earlier work, not a fresh prompt: async-delegation
# completions (`[ASYNC DELEGATION …]`) and background-process notifications
# (`[IMPORTANT: …]`). They must not open a new turn — otherwise a background
# subagent dispatched in turn N shows up as its own orphan "[ASYNC DELEGATION]"
# turn when it finishes, instead of folding into the group that spawned it. This
# mirrors the desktop live view, which only resets the live turn on a real
# ``prompt.submit``.
_CONTINUATION_PREFIXES = (
    "[ASYNC DELEGATION",
    "[IMPORTANT:",
)


def _is_continuation(message: Dict[str, Any]) -> bool:
    """True for a synthetic re-injection that should merge into the current turn."""
    if message.get("role") != "user":
        return False
    return _as_text(message.get("content")).lstrip().startswith(_CONTINUATION_PREFIXES)


def _split_turns(messages: List[Dict[str, Any]]) -> List[tuple]:
    """Split messages into turns. A turn begins at each *real* ``user`` message and
    runs until the next one. Synthetic continuations (async-delegation /
    background-process re-injections) do NOT start a turn — they merge into the
    one that spawned the work. Leading non-user messages (e.g. system) join the
    first turn. Returns ``[(start_idx, end_idx), ...]`` index ranges.
    """
    bounds: List[tuple] = []
    start = 0
    for i, m in enumerate(messages):
        if i > 0 and m.get("role") == "user" and not _is_continuation(m):
            bounds.append((start, i))
            start = i
    bounds.append((start, len(messages)))
    return bounds


def build_session_turns(
    store: _SessionStore,
    session_id: str,
    *,
    include_subagents: bool = True,
) -> List[Trace]:
    """Build one :class:`Trace` per turn for a session.

    A turn (one user prompt → the agent's full response, subagents included) is
    the natural trace unit — it has no inter-turn idle gaps, so each renders as a
    tight waterfall. Returns traces in chronological order.
    """
    session = store.get_session(session_id)
    if not session:
        return []
    messages = store.get_messages(session_id)
    if not messages:
        return []

    meta = {
        "source": session.get("source"),
        "model": session.get("model"),
        "cwd": session.get("cwd"),
        "git_branch": session.get("git_branch"),
    }
    out: List[Trace] = []
    for ti, (a, b) in enumerate(_split_turns(messages)):
        slice_msgs = messages[a:b]
        if not slice_msgs:
            continue
        trace = Trace(
            trace_id=f"{_trace_id(session_id)}:t{ti}",
            root_session_id=session_id,
            metadata={**meta, "turn": ti},
        )
        root = _build_session_spans(
            store,
            session,
            parent_span_id=None,
            trace=trace,
            messages=slice_msgs,
            agent_key=f"turn{ti}",
        )
        if not root:
            continue
        trace.root_span_id = root.span_id
        if include_subagents:
            _attach_subagents(
                store,
                session_id,
                trace,
                agent_key=f"turn{ti}",
                window=(root.start, root.end),
                _depth=0,
                _max_depth=8,
            )
        trace.spans.sort(key=lambda s: (s.start, s.span_id))
        out.append(trace)

    return out


def _build_session_spans(
    store: _SessionStore,
    session: Dict[str, Any],
    *,
    parent_span_id: Optional[str],
    trace: Trace,
    messages: Optional[List[Dict[str, Any]]] = None,
    agent_key: str = "root",
) -> Optional[Span]:
    """Append the AGENT span for one session (or a turn slice of it) plus its
    LLM/TOOL child spans.

    ``messages`` lets a caller pass a turn-scoped slice; ``agent_key`` keeps the
    AGENT span id unique per turn. Returns the session's root (AGENT) span, or
    ``None`` for an empty session.
    """
    session_id = session["id"]
    if messages is None:
        messages = store.get_messages(session_id)
    if not messages:
        return None

    msg_start = min(float(m["timestamp"]) for m in messages)
    msg_end = max(float(m["timestamp"]) for m in messages)
    started_at = float(session.get("started_at") or msg_start)

    # Clamp the AGENT span to real message activity. Session ``started_at`` /
    # ``ended_at`` are unreliable for trace timing: ``ended_at`` can be a
    # cleanup/orphan reaper firing hours later (e.g. ``ws_orphan_reap``), and on
    # a turn slice ``started_at`` is the whole-session start, far before this
    # turn. Snap to them only when they sit right at the slice's edges, so a
    # span never balloons into inter-turn idle.
    activity_start = msg_start
    if msg_start - _END_GRACE_SECONDS <= started_at <= msg_start:
        activity_start = started_at
    activity_end = msg_end
    raw_ended = session.get("ended_at")
    if raw_ended is not None:
        ended_at = float(raw_ended)
        if msg_end <= ended_at <= msg_end + _END_GRACE_SECONDS:
            activity_end = ended_at

    goal = _session_goal(messages, session)
    agent_span = Span(
        span_id=_mk_span_id("agent", session_id, agent_key),
        parent_id=parent_span_id,
        name=goal,
        kind=KIND_AGENT,
        start=activity_start,
        end=activity_end,
        status=STATUS_ERROR if session.get("end_reason") in {"error", "failed"} else STATUS_OK,
        session_id=session_id,
        attributes={
            "session.id": session_id,
            "session.source": session.get("source"),
            "llm.model_name": session.get("model"),
            "llm.token_count.prompt": session.get("input_tokens"),
            "llm.token_count.completion": session.get("output_tokens"),
            "llm.token_count.reasoning": session.get("reasoning_tokens"),
            "session.message_count": session.get("message_count"),
            "session.tool_call_count": session.get("tool_call_count"),
            "session.end_reason": session.get("end_reason"),
        },
    )
    trace.spans.append(agent_span)

    # Pre-index tool results by tool_call_id so calls pair with their output.
    results_by_id: Dict[str, Dict[str, Any]] = {}
    for m in messages:
        if m.get("role") == "tool" and m.get("tool_call_id"):
            results_by_id[str(m["tool_call_id"])] = m

    # Walk the turn timeline. An assistant message closes the LLM span that
    # began at the previous boundary; each tool_call it carries becomes a TOOL
    # span ending at its paired result.
    prev_boundary = activity_start
    for m in messages:
        role = m.get("role")
        ts = float(m["timestamp"])

        if role == "assistant":
            llm_span = Span(
                span_id=_mk_span_id("llm", session_id, m["id"]),
                parent_id=agent_span.span_id,
                name=_llm_span_name(),
                kind=KIND_LLM,
                start=prev_boundary,
                end=ts,
                status=STATUS_OK,
                session_id=session_id,
                attributes={
                    "llm.model_name": session.get("model"),
                    "llm.token_count.completion": m.get("token_count"),
                    "output.value": _short(_as_text(m.get("content")), 2000),
                    "hermes.finish_reason": m.get("finish_reason"),
                    "hermes.has_reasoning": bool(
                        m.get("reasoning") or m.get("reasoning_content")
                    ),
                },
            )
            trace.spans.append(llm_span)

            for call in m.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                _append_tool_span(
                    trace=trace,
                    session=session,
                    parent_span_id=agent_span.span_id,
                    call=call,
                    call_ts=ts,
                    results_by_id=results_by_id,
                    fallback_end=activity_end,
                )

            prev_boundary = ts
        elif role in {"user", "tool"}:
            # User input and tool results define the next LLM span's start.
            prev_boundary = ts

    return agent_span


def _append_tool_span(
    *,
    trace: Trace,
    session: Dict[str, Any],
    parent_span_id: str,
    call: Dict[str, Any],
    call_ts: float,
    results_by_id: Dict[str, Dict[str, Any]],
    fallback_end: float,
) -> None:
    session_id = session["id"]
    call_id = _tool_call_id(call)
    name = _tool_call_name(call)
    result = results_by_id.get(call_id) if call_id else None
    end = float(result["timestamp"]) if result else fallback_end
    status = STATUS_OK
    if result and _looks_like_error(result):
        status = STATUS_ERROR
    elif not result:
        status = STATUS_UNSET

    args = _tool_call_args(call)
    span = Span(
        span_id=_mk_span_id("tool", session_id, call_id or f"{call_ts}:{name}"),
        parent_id=parent_span_id,
        name=name,
        kind=KIND_TOOL,
        start=call_ts,
        end=max(end, call_ts),
        status=status,
        session_id=session_id,
        attributes={
            "tool.name": name,
            "tool.call_id": call_id or None,
            "input.value": _short(_as_text(args), 2000),
            "output.value": _short(_as_text(result.get("content")), 2000) if result else None,
            "hermes.is_delegate": name in _DELEGATE_TOOL_NAMES,
        },
    )
    trace.spans.append(span)


def _attach_subagents(
    store: _SessionStore,
    session_id: str,
    trace: Trace,
    *,
    agent_key: str = "root",
    window: Optional[tuple] = None,
    _depth: int,
    _max_depth: int,
) -> None:
    """Nest each delegate child session under the tool span that spawned it.

    Children are matched to ``delegate_task`` tool spans by start-time proximity
    (each child consumed once). Unmatched children attach to the session's AGENT
    span so they are never dropped from the trace. ``window`` (start, end) limits
    attachment to children spawned during a turn slice.
    """
    child_ids = store.get_child_session_ids(session_id)
    if not child_ids:
        return

    delegate_spans = sorted(
        (
            s
            for s in trace.spans
            if s.session_id == session_id
            and s.kind == KIND_TOOL
            and s.attributes.get("hermes.is_delegate")
        ),
        key=lambda s: s.start,
    )
    agent_span_id = _mk_span_id("agent", session_id, agent_key)

    children = []
    for cid in child_ids:
        csess = store.get_session(cid)
        if not csess:
            continue
        if window is not None:
            cstart = float(csess.get("started_at") or 0.0)
            if not (window[0] - 1.0 <= cstart <= window[1] + 1.0):
                continue
        children.append(csess)
    children.sort(key=lambda c: float(c.get("started_at") or 0.0))

    used: set = set()
    for csess in children:
        cstart = float(csess.get("started_at") or 0.0)
        parent_span_id = agent_span_id
        best = None
        best_gap = None
        for ds in delegate_spans:
            if ds.span_id in used:
                continue
            gap = abs(ds.start - cstart)
            if best_gap is None or gap < best_gap:
                best, best_gap = ds, gap
        if best is not None:
            used.add(best.span_id)
            parent_span_id = best.span_id

        child_root = _build_session_spans(
            store, csess, parent_span_id=parent_span_id, trace=trace
        )
        if child_root and _depth + 1 < _max_depth:
            _attach_subagents(
                store, csess["id"], trace, _depth=_depth + 1, _max_depth=_max_depth
            )


def _session_goal(messages: List[Dict[str, Any]], session: Dict[str, Any]) -> str:
    """A human-readable label for an AGENT span.

    Prefer the first user message in the given slice so per-turn spans get their
    own prompt as a label (the session title is identical across every turn).
    Fall back to the session title, then a short id.
    """
    for m in messages:
        if m.get("role") == "user":
            text = _as_text(m.get("content")).strip()
            if text:
                return _short(text, 120)
    title = session.get("title")
    if title:
        return _short(str(title), 120)
    return f"session {str(session.get('id', ''))[:8]}"
