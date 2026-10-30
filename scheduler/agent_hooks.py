"""Agent-side adapters for the scheduling engine.

Every entry point is an observer: it never raises into a turn, and with no registered tasks
it returns after a single ``active`` check (no lock, allocation, or DB read).

Design: website/docs/developer-guide/scheduled-tasks.md.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("hermes.scheduler")


def _engine() -> Any:
    from scheduler.engine import get_engine

    return get_engine()


def _session_id(agent: Any) -> str:
    return str(getattr(agent, "session_id", "") or "")


def _profile_name(agent: Any) -> str:
    for attr in ("_profile_name", "profile_name", "_own_profile"):
        value = getattr(agent, attr, None)
        if isinstance(value, str) and value:
            return value
    return ""


def _agent_name(agent: Any) -> str:
    for attr in ("_agent_name", "agent_name"):
        value = getattr(agent, attr, None)
        if isinstance(value, str) and value:
            return value
    return ""


def _durable_seed(agent: Any) -> tuple[Optional[int], str]:
    """``(step_seq, profile_name)`` from the durable session row, when readable."""
    db = getattr(agent, "_session_db", None)
    getter = getattr(db, "get_session", None)
    session_id = _session_id(agent)
    if not session_id or not callable(getter):
        return None, ""
    try:
        row = getter(session_id)
        if not row:
            return None, ""
        return int(row.get("step_seq") or 0), str(row.get("profile_name") or "")
    except Exception:
        return None, ""


def _ensure(agent: Any, *, seed_steps: bool) -> Optional[str]:
    engine = _engine()
    session_id = _session_id(agent)
    if not session_id:
        return None
    step_seq: Optional[int] = None
    db_profile = ""
    if seed_steps:
        step_seq, db_profile = _durable_seed(agent)
    try:
        engine.ensure_session(
            session_id,
            turn_seq=max(0, int(getattr(agent, "_user_turn_count", 0) or 0) - 1),
            step_seq=step_seq,
            profile_name=_profile_name(agent) or db_profile,
            agent_name=_agent_name(agent),
        )
    except Exception:
        logger.debug("scheduler: ensure_session failed", exc_info=True)
        return None
    return session_id


def on_turn_start(agent: Any) -> None:
    """Fire pre-turn tasks for the turn that is starting."""
    engine = _engine()
    if not engine.active:
        return
    session_id = _ensure(agent, seed_steps=True)
    if not session_id:
        return
    try:
        engine.turn_pre(
            session_id,
            turn_seq=int(getattr(agent, "_user_turn_count", 0) or 0),
            turn_id=getattr(agent, "_current_turn_id", "") or None,
            profile_name=_profile_name(agent), agent_name=_agent_name(agent),
        )
    except Exception:
        logger.debug("scheduler: turn_pre failed", exc_info=True)


def on_step_pre(agent: Any, *, step_index: int) -> None:
    """Fire pre-step tasks for the request that is about to reach the provider."""
    engine = _engine()
    if not engine.active:
        return
    session_id = _session_id(agent)
    if not session_id:
        return
    try:
        if not engine.has_session(session_id):
            session_id = _ensure(agent, seed_steps=True)
            if not session_id:
                return
        engine.step_pre(
            session_id, step_index=step_index,
            turn_id=getattr(agent, "_current_turn_id", "") or None,
            turn_seq=int(getattr(agent, "_user_turn_count", 0) or 0),
        )
    except Exception:
        logger.debug("scheduler: step_pre failed", exc_info=True)


def on_step_committed(
    agent: Any, *, step_index: int, usage_status: str = "ok", status: str = "ok",
    total_tokens: int = 0, input_tokens: int = 0, output_tokens: int = 0,
    cache_read_tokens: int = 0, cache_write_tokens: int = 0, reasoning_tokens: int = 0,
    duration_ms: int = 0, finish_reason: Optional[str] = None,
) -> None:
    """Advance the session's step clock and fire post-step tasks."""
    engine = _engine()
    if not engine.active:
        return
    session_id = _session_id(agent)
    if not session_id:
        return
    try:
        if not engine.has_session(session_id):
            session_id = _ensure(agent, seed_steps=True)
            if not session_id:
                return
        engine.step_committed(
            session_id, step_index=step_index, usage_status=usage_status, status=status,
            total_tokens=total_tokens, input_tokens=input_tokens, output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens, cache_write_tokens=cache_write_tokens,
            reasoning_tokens=reasoning_tokens, duration_ms=duration_ms, finish_reason=finish_reason,
            turn_id=getattr(agent, "_current_turn_id", "") or None,
            turn_seq=int(getattr(agent, "_user_turn_count", 0) or 0),
        )
    except Exception:
        logger.debug("scheduler: step_committed failed", exc_info=True)


def on_turn_committed(agent: Any, result: Any) -> None:
    """Advance the turn clock only for a completed turn (``result["completed"]``)."""
    engine = _engine()
    if not engine.active:
        return
    if not isinstance(result, dict) or not result.get("completed"):
        return
    try:
        _ensure(agent, seed_steps=True)
        engine.turn_committed(
            _session_id(agent),
            turn_seq=int(getattr(agent, "_user_turn_count", 0) or 0),
            turn_id=getattr(agent, "_current_turn_id", "") or None,
            profile_name=_profile_name(agent), agent_name=_agent_name(agent),
        )
    except Exception:
        logger.debug("scheduler: turn_committed failed", exc_info=True)
