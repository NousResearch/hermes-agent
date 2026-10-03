"""One-shot session wake deadlines - the orchestrator's own alarm (#122444).

Session-scoped and in-process (the classic CLI must be running; durable cross-process
scheduling stays ``hermes cron``). A turn arms a wake with ``schedule_wake``; the CLI idle
hook injects the prompt into ``_pending_input`` once ``fires_at`` passes, so a long-running
orchestrator re-enters its loop with zero external input instead of sleeping forever.

Invariants (mirrors heartbeat.py): injection is a plain user message - no system-prompt
mutation or toolset swap, so prompt caching stays intact - the wake is one-shot (consumed
on fire, never re-armed by the driver), and a real user message always wins: a wake only
fires into an idle session with an empty input queue.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

_WAKE_PREFIX = "wake:"


@dataclass
class WakeState:
    """Serializable per-session one-shot wake deadline."""

    prompt: str
    fires_at: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "WakeState":
        d = json.loads(raw)
        return cls(prompt=str(d.get("prompt", "")), fires_at=float(d.get("fires_at", 0.0)))


def _get_session_db() -> Optional[Any]:
    """Persistence goes through the goals module's per-HERMES_HOME cached SessionDB (one shared connection)."""
    try:
        from hermes_cli.goals import _get_session_db as _goals_db

        return _goals_db()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("wake: SessionDB bootstrap failed (%s)", exc)
        return None


def load_wake(session_id: str) -> Optional[WakeState]:
    db = _get_session_db() if session_id else None
    if db is None:
        return None
    try:
        raw = db.get_meta(_WAKE_PREFIX + session_id)
    except Exception as exc:
        logger.debug("wake: get_meta failed (%s)", exc)
        return None
    try:
        return WakeState.from_json(raw) if raw else None
    except Exception as exc:
        logger.warning("wake: could not parse stored wake for %s: %s", session_id, exc)
        return None


def save_wake(session_id: str, state: WakeState) -> None:
    if not session_id:
        return
    db = _get_session_db()
    if db is None:
        from hermes_cli.goals import _warn_dropped_write

        _warn_dropped_write("wake", "wake", session_id)
        return
    try:
        db.set_meta(_WAKE_PREFIX + session_id, state.to_json())
    except Exception as exc:
        logger.debug("wake: set_meta failed (%s)", exc)


def clear_wake(session_id: str) -> None:
    if not session_id:
        return
    db = _get_session_db()
    if db is None:
        return
    try:
        db.set_meta(_WAKE_PREFIX + session_id, "")
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("wake: clear failed (%s)", exc)


def schedule_wake(session_id: str, prompt: str, fires_at: float) -> WakeState:
    """Arm a one-shot wake for ``session_id``; replaces any wake already armed (latest wins)."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("wake prompt is empty")
    state = WakeState(prompt=prompt, fires_at=float(fires_at))
    save_wake(session_id, state)
    return state


def due_wake_prompt(session_id: str, now: Optional[float] = None) -> Optional[str]:
    """Return the prompt if the wake deadline passed, consuming the wake (one-shot), else None.

    The wake is cleared BEFORE the caller queues the prompt: like heartbeat's claim-first
    ``due_prompt``, a consumed wake may be lost only on a crash, never double-fired into
    two turns. Throttled callers (the 10 Hz idle hook) bound the DB read rate.
    """
    state = load_wake(session_id)
    if state is None:
        return None
    if (now if now is not None else time.time()) < state.fires_at:
        return None
    clear_wake(session_id)
    return state.prompt


def migrate_wake_to_session(old_session_id: str, new_session_id: str) -> bool:
    """Carry an armed wake across a compression-driven session rotation (copy to child,
    clear parent, never raise). Same shape as ``heartbeat.migrate_heartbeat_to_session``:
    without this the deadline is stranded on the dead session id exactly when the issue's
    compaction case would need it."""
    if not old_session_id or not new_session_id or old_session_id == new_session_id:
        return False
    try:
        state = load_wake(old_session_id)
        if state is None or load_wake(new_session_id) is not None:
            return False
        save_wake(new_session_id, state)
        clear_wake(old_session_id)
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("wake: migrate failed (%s)", exc)
        return False


__all__ = [
    "WakeState",
    "schedule_wake",
    "load_wake",
    "clear_wake",
    "due_wake_prompt",
    "migrate_wake_to_session",
]
