"""Session-scoped control surface for plugins: steer, interrupt, continue a turn.

A plugin hook receives a session id but no agent handle, so a plugin that decides
mid-turn that the agent is heading the wrong way has no way to say so before the
turn ends. This module is the single bridge for that decision:

* :func:`steer` queues text for the next safe boundary (after the current tool
  batch) or, with ``interrupt=True``, also aborts the in-flight model request so
  the text lands one iteration earlier. It reuses the host's ``/steer``
  machinery: one standalone user row, never a mutation of an already-persisted
  row.
* :func:`request_continuation` leaves text for the stop gate: when the model
  answers, the gate keeps the turn open, appends the text, and lets the model
  respond once more. Bounded by ``agent.max_plugin_continuations`` per turn.
* :func:`turn_state` reports what a plugin can safely know about the live turn.

Only the running turn is bound (``bind_turn`` in ``agent.conversation_loop``).
A call after the turn ended returns ``{"ok": False, "reason": "no_live_turn"}``;
the plugin must fall back to its own queue instead of assuming delivery.

Everything here is in-memory and thread-safe: a plugin's background worker may
call it while the turn loop runs. It never calls a model and never blocks.
"""

from __future__ import annotations

import logging
import threading
import time
import weakref
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

#: Bumped when the callable surface changes so a plugin can feature-detect.
#: v3 adds ``live_state`` (P-0023), same-name per-run tools (P-0021), the interrupt tree kill
#: (P-0022) and a caller-pinned cache scope (P-0024).
CAPABILITY_VERSION = "plugin_runtime_v3"
#: ``display_kind`` on the durable row a continuation is delivered as. Clients
#: that do not know it still render the text; the value keeps it distinguishable
#: from a human prompt in the transcript.
DISPLAY_KIND = "adviser"
#: A single steer/continuation text is a device-side message; cap it like a tool result.
MAX_TEXT_CHARS = 4000
DEFAULT_MAX_CONTINUATIONS = 2
#: Continuation/steer tokens are only queried within the turn and the one after it.
_TOKEN_TTL_SECONDS = 600.0

_LOCK = threading.RLock()
#: session id -> weakref to the agent whose turn is running.
_LIVE: dict[str, weakref.ReferenceType] = {}
#: session id -> read-only facts about the live turn.
_TURNS: dict[str, dict[str, Any]] = {}
#: session id -> the one continuation waiting for the stop gate.
_CONTINUATIONS: dict[str, dict[str, Any]] = {}
#: session id -> the one post-turn text the surface should run as a fresh turn.
_POST_TURN: dict[str, dict[str, Any]] = {}
#: session id -> cooperative flags published by plugins (plan_mode, ...); not turn-scoped.
_FLAGS: dict[str, dict[str, Any]] = {}
#: token -> {"state": pending|taken|expired, "session", "created_at", "taken_at"}
_TOKENS: dict[str, dict[str, Any]] = {}


def _clip(text: Any) -> str:
    cleaned = str(text or "").strip()
    if len(cleaned) <= MAX_TEXT_CHARS:
        return cleaned
    return cleaned[: MAX_TEXT_CHARS - 1].rstrip() + "…"


def _prune(now: Optional[float] = None) -> None:
    cutoff = (now if now is not None else time.time()) - _TOKEN_TTL_SECONDS
    for token, record in list(_TOKENS.items()):
        if float(record.get("created_at") or 0.0) < cutoff:
            _TOKENS.pop(token, None)


def _agent_for(session_id: str) -> Optional[Any]:
    ref = _LIVE.get(session_id)
    if ref is None:
        return None
    agent = ref()
    if agent is None:
        _LIVE.pop(session_id, None)
        _TURNS.pop(session_id, None)
    return agent


def bind_turn(agent: Any) -> bool:
    """Register the agent whose turn is starting. Fail-open; never raises."""
    try:
        session = str(getattr(agent, "session_id", "") or "")
        if not session:
            return False
        with _LOCK:
            _LIVE[session] = weakref.ref(agent)
            coding = getattr(agent, "_resolved_is_coding", None)
            _TURNS[session] = {
                "turn_id": str(getattr(agent, "_current_turn_id", "") or ""),
                "platform": str(getattr(agent, "platform", "") or ""),
                "coding": None if coding is None else bool(coding),
                "bound_at": time.time(),
            }
        return True
    except Exception:
        logger.debug("plugin_runtime.bind_turn failed", exc_info=True)
        return False


def unbind_turn(agent: Any) -> None:
    """Release the turn. A continuation nobody consumed is marked expired."""
    try:
        session = str(getattr(agent, "session_id", "") or "")
        if not session:
            return
        with _LOCK:
            ref = _LIVE.get(session)
            if ref is not None and ref() is not agent:
                return
            _LIVE.pop(session, None)
            _TURNS.pop(session, None)
            pending = _CONTINUATIONS.pop(session, None)
            if pending is not None:
                record = _TOKENS.get(pending["token"])
                if record is not None and record.get("state") == "pending":
                    record["state"] = "expired"
                    record["expired_at"] = time.time()
    except Exception:
        logger.debug("plugin_runtime.unbind_turn failed", exc_info=True)


def live_turn(session_id: str) -> bool:
    with _LOCK:
        return _agent_for(str(session_id or "")) is not None


def turn_state(session_id: str) -> dict[str, Any]:
    """Facts a plugin may branch on. Absent values are ``None``, never guessed."""
    session = str(session_id or "")
    now = time.time()
    with _LOCK:
        agent = _agent_for(session)
        facts = dict(_TURNS.get(session) or {})
        pending = _CONTINUATIONS.get(session)
        flags = dict(_FLAGS.get(session) or {})
    bound_at = float(facts.pop("bound_at", 0.0) or 0.0)
    return {
        "capability": CAPABILITY_VERSION,
        "live": agent is not None,
        "session_id": session,
        # Read live: bind_turn runs before the turn context stamps _current_turn_id.
        "turn_id": str(getattr(agent, "_current_turn_id", "") or facts.get("turn_id") or ""),
        "platform": facts.get("platform") or "",
        "coding": facts.get("coding"),
        "bound_seconds": round(now - bound_at, 3) if bound_at else 0.0,
        "pending_continuation": bool(pending),
        "flags": flags,
    }


def set_flag(session_id: str, key: str, value: Any) -> bool:
    """Publish a cooperative session flag (``plan_mode``, ``review_only``...).

    Flags are not turn-scoped: a plugin sets one when its mode changes and the adviser reads it
    through :func:`turn_state`. Only a live session can publish; a flag is advisory metadata,
    never an authorization.
    """
    session = str(session_id or "")
    key = str(key or "").strip()[:64]
    if not session or not key:
        return False
    with _LOCK:
        if _agent_for(session) is None:
            return False
        _FLAGS.setdefault(session, {})[key] = value
    return True


def get_flags(session_id: str) -> dict[str, Any]:
    with _LOCK:
        return dict(_FLAGS.get(str(session_id or "")) or {})


def request_interrupt(session_id: str, *, source: str = "plugin") -> dict[str, Any]:
    """Ask the live turn to stop treating the current tool batch as authoritative.

    Reuses the host's cross-thread ``interrupt``: the executor cancels queued tool futures and
    stops waiting for running ones; cooperative tools see ``is_interrupted()``. A tool that
    never polls is abandoned, not killed.
    """
    session = str(session_id or "")
    with _LOCK:
        agent = _agent_for(session)
    if agent is None:
        return {"ok": False, "reason": "no_live_turn"}
    interrupt_fn = getattr(agent, "interrupt", None)
    if not callable(interrupt_fn):
        return {"ok": False, "reason": "host_unsupported"}
    try:
        interrupt_fn(tool_reason=str(source or "plugin"))
    except Exception as exc:
        logger.warning("plugin_runtime.interrupt failed (%s): %s", source, exc)
        return {"ok": False, "reason": "host_error"}
    return {"ok": True, "reason": ""}


def live_state(session_id: str) -> dict[str, Any]:
    """Facts about the live turn a delivery channel may branch on, read from the live agent.

    * ``streaming``: a provider request is in flight for this turn (the host brackets every
      model call with ``_model_request_active``), so the answer is being produced right now.
    * ``aborting``: an interrupt is pending and has not been cleared.
    * ``tool_batch``: a tool batch is executing.
    * ``terminal_idle``: the turn is bound and none of the above is true.

    A turn that is not bound reports every flag false: a caller must fall back to its own queue
    rather than read ``terminal_idle`` as "safe to deliver".
    """
    session = str(session_id or "")
    with _LOCK:
        agent = _agent_for(session)
        facts = dict(_TURNS.get(session) or {})
    if agent is None:
        return {"capability": CAPABILITY_VERSION, "live": False, "session_id": session,
                "turn_id": "", "streaming": False, "aborting": False,
                "tool_batch": False, "terminal_idle": False}
    request_active = getattr(agent, "_model_request_active", None)
    streaming = bool(request_active.is_set()) if request_active is not None else False
    aborting = bool(getattr(agent, "_interrupt_requested", False))
    tool_batch = bool(getattr(agent, "_executing_tools", False))
    return {
        "capability": CAPABILITY_VERSION,
        "live": True,
        "session_id": session,
        "turn_id": str(getattr(agent, "_current_turn_id", "") or facts.get("turn_id") or ""),
        "streaming": streaming,
        "aborting": aborting,
        "tool_batch": tool_batch,
        "terminal_idle": not (streaming or aborting or tool_batch),
    }


def interrupt(session_id: str, text: str, *, source: str = "plugin") -> dict[str, Any]:
    """Queue advice and interrupt the running tool batch in one call.

    Steer first, interrupt second: an interrupt can clear transient state, and the queued row
    must survive it. When the interrupt is refused (race with turn end) the text is still
    queued, so the caller gets ``mode="steer"`` instead of losing it.
    """
    queued = steer(session_id, text, source=source)
    if not queued.get("ok"):
        return queued
    stopped = request_interrupt(session_id, source=source)
    return {"ok": True, "mode": "interrupt" if stopped.get("ok") else "steer",
            "reason": stopped.get("reason") or ""}


def queue_post_turn(session_id: str, text: str, *, source: str = "plugin",
                    kind: str = "adviser") -> dict[str, Any]:
    """Leave text for the surface to run as a fresh turn after this one ends.

    The surfaces already auto-submit a leftover ``pending_steer`` as the next user input
    (CLI/gateway/TUI), so this is a turn trigger, not a new API. Only a live turn may queue one;
    one per session, and the caller must enforce its own loop budget.
    """
    session = str(session_id or "")
    body = _clip(text)
    if not body:
        return {"ok": False, "reason": "empty_text"}
    with _LOCK:
        _prune()
        if _agent_for(session) is None:
            return {"ok": False, "reason": "no_live_turn"}
        if session in _POST_TURN:
            return {"ok": False, "reason": "already_pending"}
        _POST_TURN[session] = {"text": body, "source": str(source or ""),
                               "kind": str(kind or "adviser"), "created_at": time.time()}
    return {"ok": True, "reason": ""}


def take_post_turn(session_id: str) -> Optional[dict[str, Any]]:
    with _LOCK:
        return _POST_TURN.pop(str(session_id or ""), None)


def attach_post_turn(agent: Any, result: Any) -> None:
    """Merge a queued post-turn text into the conversation result the surface reads.

    The result's ``pending_steer`` is the contract all surfaces already honour; the extra
    ``pending_steer_kind`` lets a surface label an adviser-triggered turn instead of showing it
    as leftover operator input. Never raises into the turn path.
    """
    try:
        session = str(getattr(agent, "session_id", "") or "")
        pending = take_post_turn(session)
        if not pending or not isinstance(result, dict):
            return
        existing = str(result.get("pending_steer") or "")
        text = str(pending.get("text") or "")
        result["pending_steer"] = f"{existing}\n{text}" if existing else text
        result["pending_steer_kind"] = str(pending.get("kind") or "adviser")
        logger.debug("plugin post-turn queued (session=%s, source=%s)",
                     session, pending.get("source") or "")
    except Exception:
        logger.debug("plugin_runtime.attach_post_turn failed", exc_info=True)


def preserve_handle(session_id: str) -> Optional[Callable[..., dict[str, Any]]]:
    """Capture a turn-end card writer, or ``None`` when no turn is bound.

    A review that decides to *preserve* advice finishes after the turn, when the plugin no
    longer has an agent handle. Call this while the turn is live (the plugin does it in
    ``post_llm_call``) and keep the closure: it appends one adviser row at the end of the
    turn, flushes that row to the session DB, and notifies the surface. The row is a real
    transcript message (``display_kind=adviser``), so it hydrates as a card on resume.
    """
    agent = _agent_for(str(session_id or ""))
    if agent is None:
        return None
    session = str(getattr(agent, "session_id", "") or session_id)

    def _append(text: str, metadata: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        body = _clip(text)
        if not body:
            return {"ok": False, "reason": "empty_text"}
        if str(getattr(agent, "session_id", "") or "") != session:
            return {"ok": False, "reason": "session_changed"}
        try:
            from agent.plugin_events import notify_plugin_message
            from agent.prompt_builder import advisory_user_row
        except Exception:
            logger.debug("plugin_runtime.preserve: host API missing", exc_info=True)
            return {"ok": False, "reason": "host_api_missing"}
        row = advisory_user_row(body)
        if isinstance(metadata, dict) and metadata:
            row["display_metadata"] = dict(metadata)
        messages = getattr(agent, "messages", None)
        if not isinstance(messages, list):
            return {"ok": False, "reason": "no_transcript"}
        with _LOCK:
            messages.append(row)
        flush = getattr(agent, "_flush_messages_to_session_db", None)
        if callable(flush):
            try:
                flush(messages)
            except Exception:
                logger.debug("plugin_runtime.preserve: flush failed", exc_info=True)
        notify_plugin_message(agent, row)
        return {"ok": True, "id": str(row.get("plugin_message_id") or ""), "reason": ""}

    return _append


def preserve(session_id: str, text: str, *,
             metadata: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """One-shot form of :func:`preserve_handle`."""
    handle = preserve_handle(session_id)
    if handle is None:
        return {"ok": False, "reason": "no_live_turn"}
    return handle(text, metadata)


def steer(session_id: str, text: str, *, source: str = "") -> dict[str, Any]:
    """Queue plugin advice for the live turn.

    Always the host's queued-steer path, never ``redirect``: redirect persists a *user
    correction* row, and a reviewer's advice must not borrow the user's authority. The row is
    typed ``display_kind="adviser"`` (see ``agent.prompt_builder.advisory_user_row``), so the
    model is not told the text is a direct user message.

    Returns ``{"ok": bool, "mode": "steer"|"", "reason": str}``.
    """
    session = str(session_id or "")
    body = _clip(text)
    if not body:
        return {"ok": False, "mode": "", "reason": "empty_text"}
    with _LOCK:
        agent = _agent_for(session)
    if agent is None:
        return {"ok": False, "mode": "", "reason": "no_live_turn"}
    try:
        accepted = bool(agent.steer(body, kind="adviser"))
    except TypeError:
        # A host that predates the ``kind`` keyword still accepts the text; the row then
        # carries the operator marker, which is the pre-patch behaviour.
        try:
            accepted = bool(agent.steer(body))
        except Exception as exc:
            logger.warning("plugin_runtime.steer failed (%s): %s", source, exc)
            return {"ok": False, "mode": "", "reason": "host_error"}
    except Exception as exc:
        logger.warning("plugin_runtime.steer failed (%s): %s", source, exc)
        return {"ok": False, "mode": "", "reason": "host_error"}
    return {"ok": accepted, "mode": "steer" if accepted else "",
            "reason": "" if accepted else "host_refused"}


def request_continuation(session_id: str, text: str, *, source: str = "",
                         token: str = "") -> dict[str, Any]:
    """Leave one message for the stop gate of the live turn.

    At most one continuation may wait per session: the first wins so a noisy
    reviewer cannot stack a queue the model reads as several user turns.
    """
    session = str(session_id or "")
    body = _clip(text)
    if not body:
        return {"ok": False, "token": "", "reason": "empty_text"}
    now = time.time()
    with _LOCK:
        _prune(now)
        if _agent_for(session) is None:
            return {"ok": False, "token": "", "reason": "no_live_turn"}
        if session in _CONTINUATIONS:
            return {"ok": False, "token": "", "reason": "already_pending"}
        token = token or f"{session}:{time.time_ns()}"
        _CONTINUATIONS[session] = {
            "token": token, "text": body, "source": str(source or ""), "created_at": now,
        }
        _TOKENS[token] = {
            "state": "pending", "session": session, "created_at": now,
        }
    return {"ok": True, "token": token, "reason": ""}


def take_continuation(session_id: str) -> Optional[dict[str, Any]]:
    """The stop gate's drain: pop the pending continuation for this session."""
    session = str(session_id or "")
    with _LOCK:
        pending = _CONTINUATIONS.pop(session, None)
        if pending is None:
            return None
        record = _TOKENS.get(pending["token"])
        if record is not None:
            record["state"] = "taken"
            record["taken_at"] = time.time()
    return pending


def discard_continuation(session_id: str) -> Optional[dict[str, Any]]:
    """Drop a continuation nobody will consume (budget exhausted).

    Distinct from :func:`take_continuation` on purpose: the caller never appends
    the text, so the token must read ``expired`` and the plugin must return its
    advice rows to the pending queue instead of assuming delivery.
    """
    session = str(session_id or "")
    with _LOCK:
        pending = _CONTINUATIONS.pop(session, None)
        if pending is None:
            return None
        record = _TOKENS.get(pending["token"])
        if record is not None and record.get("state") == "pending":
            record["state"] = "expired"
            record["expired_at"] = time.time()
    return pending


def continuation_status(session_id: str, token: str) -> str:
    """``pending`` / ``taken`` / ``expired`` / ``unknown`` for one token."""
    if not token:
        return "unknown"
    with _LOCK:
        _prune()
        record = _TOKENS.get(str(token))
        if record is None:
            return "unknown"
        state = str(record.get("state") or "unknown")
        if state == "pending" and str(record.get("session") or "") != str(session_id or ""):
            return "unknown"
        return state


def max_continuations() -> int:
    """Per-turn budget for the stop gate (``agent.max_plugin_continuations``)."""
    try:
        from hermes_cli.config import load_config

        agent_cfg = (load_config() or {}).get("agent") or {}
        return max(0, int(agent_cfg.get("max_plugin_continuations", DEFAULT_MAX_CONTINUATIONS)))
    except (TypeError, ValueError):
        return DEFAULT_MAX_CONTINUATIONS
    except Exception:
        return DEFAULT_MAX_CONTINUATIONS


def clear_session(session_id: str) -> None:
    """Drop every record for a session (``/new``, teardown, tests)."""
    session = str(session_id or "")
    with _LOCK:
        _LIVE.pop(session, None)
        _TURNS.pop(session, None)
        _POST_TURN.pop(session, None)
        _FLAGS.pop(session, None)
        pending = _CONTINUATIONS.pop(session, None)
        if pending is not None:
            _TOKENS.pop(pending["token"], None)
        for token in [t for t, rec in _TOKENS.items() if rec.get("session") == session]:
            _TOKENS.pop(token, None)


__all__ = [
    "CAPABILITY_VERSION",
    "DISPLAY_KIND",
    "MAX_TEXT_CHARS",
    "attach_post_turn",
    "bind_turn",
    "clear_session",
    "continuation_status",
    "discard_continuation",
    "get_flags",
    "interrupt",
    "live_turn",
    "max_continuations",
    "preserve",
    "preserve_handle",
    "queue_post_turn",
    "request_continuation",
    "request_interrupt",
    "set_flag",
    "steer",
    "take_continuation",
    "take_post_turn",
    "turn_state",
    "unbind_turn",
]
