"""Persistent session loops — timer-driven same-session continuation.

A loop is a prompt that repeats at a user-specified interval across turns.
After each turn completes, if the interval has elapsed and the loop is
active, the prompt is fed back into the same session as a normal user
message.  No judge, no system-prompt mutation — prompt caching stays intact.

State is persisted in SessionDB's ``state_meta`` table keyed by
``loop:<session_id>`` so ``/loop resume`` picks it up.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Constants & defaults
# ──────────────────────────────────────────────────────────────────────

DEFAULT_MAX_TURNS = 20


# ──────────────────────────────────────────────────────────────────────
# Dataclass
# ──────────────────────────────────────────────────────────────────────


@dataclass
class LoopState:
    """Serializable loop state stored per session."""

    prompt: str
    interval_seconds: int = 300       # default 5m
    status: str = "active"            # active | paused | done
    last_fired_at: float = 0.0
    created_at: float = 0.0
    turns_completed: int = 0          # turns that have finished while loop active

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "LoopState":
        data = json.loads(raw)
        return cls(
            prompt=data.get("prompt", ""),
            interval_seconds=int(data.get("interval_seconds", 300) or 300),
            status=data.get("status", "active"),
            last_fired_at=float(data.get("last_fired_at", 0.0) or 0.0),
            created_at=float(data.get("created_at", 0.0) or 0.0),
            turns_completed=int(data.get("turns_completed", 0) or 0),
        )


# ──────────────────────────────────────────────────────────────────────
# Persistence (SessionDB state_meta)
# ──────────────────────────────────────────────────────────────────────


def _meta_key(session_id: str) -> str:
    return f"loop:{session_id}"


_DB_CACHE: Dict[str, Any] = {}


def _get_session_db() -> Optional[Any]:
    """Return a SessionDB instance for the current HERMES_HOME.

    Cached per ``hermes_home`` path so profile switches still pick up
    the right DB.  Defensive against import/instantiation failures.
    """
    try:
        from hermes_constants import get_hermes_home
        from hermes_state import SessionDB

        home = str(get_hermes_home())
    except Exception as exc:  # pragma: no cover
        logger.debug("LoopManager: SessionDB bootstrap failed (%s)", exc)
        return None

    cached = _DB_CACHE.get(home)
    if cached is not None:
        return cached
    try:
        db = SessionDB()
    except Exception as exc:  # pragma: no cover
        logger.debug("LoopManager: SessionDB() raised (%s)", exc)
        return None
    _DB_CACHE[home] = db
    return db


def load_loop(session_id: str) -> Optional[LoopState]:
    """Load the loop for a session, or None if none exists."""
    if not session_id:
        return None
    db = _get_session_db()
    if db is None:
        return None
    try:
        raw = db.get_meta(_meta_key(session_id))
    except Exception as exc:
        logger.debug("LoopManager: get_meta failed: %s", exc)
        return None
    if not raw:
        return None
    try:
        return LoopState.from_json(raw)
    except Exception as exc:
        logger.warning("LoopManager: could not parse stored loop for %s: %s", session_id, exc)
        return None


def save_loop(session_id: str, state: LoopState) -> None:
    """Persist a loop to SessionDB.  No-op if DB unavailable."""
    if not session_id:
        return
    db = _get_session_db()
    if db is None:
        return
    try:
        db.set_meta(_meta_key(session_id), state.to_json())
    except Exception as exc:
        logger.debug("LoopManager: set_meta failed: %s", exc)


# ──────────────────────────────────────────────────────────────────────
# Interval parsing
# ──────────────────────────────────────────────────────────────────────


def _parse_interval(token: str) -> Optional[int]:
    """Parse a duration token into seconds.

    Examples:
        "5m" → 300
        "30m" → 1800
        "2h" → 7200
        "1d" → 86400
    """
    if not token:
        return None
    s = token.strip().lower()
    match = re.match(
        r"^(\d+)\s*(m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days)$",
        s,
    )
    if not match:
        return None
    value = int(match.group(1))
    unit = match.group(2)[0]  # First char: m, h, or d
    multipliers = {"m": 60, "h": 3600, "d": 86400}
    return value * multipliers[unit]


# ──────────────────────────────────────────────────────────────────────
# LoopManager
# ──────────────────────────────────────────────────────────────────────


class LoopManager:
    """Per-session loop state + timer-driven continuation decisions.

    The CLI and gateway each hold one ``LoopManager`` per live session.
    """

    def __init__(self, session_id: str, *, default_max_turns: int = DEFAULT_MAX_TURNS):
        self.session_id = session_id
        self.default_max_turns = int(default_max_turns or DEFAULT_MAX_TURNS)
        self._state: Optional[LoopState] = load_loop(session_id)

    # --- introspection ------------------------------------------------

    @property
    def state(self) -> Optional[LoopState]:
        return self._state

    def is_active(self) -> bool:
        return self._state is not None and self._state.status == "active"

    def status_line(self) -> str:
        s = self._state
        if s is None or s.status == "done":
            return "No active loop. Set one with /loop <prompt>."
        turns = f"{s.turns_completed}/{self.default_max_turns} turns"
        if s.status == "active":
            return f"⊙ Loop (active, {turns}): {s.prompt}"
        if s.status == "paused":
            return f"⏸ Loop (paused, {turns}): {s.prompt}"
        return f"Loop ({s.status}, {turns}): {s.prompt}"

    # --- mutation -----------------------------------------------------

    def set(self, prompt: str, *, interval_seconds: Optional[int] = None) -> LoopState:
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("loop prompt is empty")
        state = LoopState(
            prompt=prompt,
            interval_seconds=int(interval_seconds) if interval_seconds else 300,
            status="active",
            last_fired_at=0.0,
            created_at=time.time(),
            turns_completed=0,
        )
        self._state = state
        save_loop(self.session_id, state)
        return state

    def pause(self, reason: str = "user-paused") -> Optional[LoopState]:
        if not self._state:
            return None
        self._state.status = "paused"
        save_loop(self.session_id, self._state)
        return self._state

    def resume(self) -> Optional[LoopState]:
        if not self._state:
            return None
        self._state.status = "active"
        save_loop(self.session_id, self._state)
        return self._state

    def clear(self) -> None:
        if self._state is None:
            return
        self._state.status = "done"
        save_loop(self.session_id, self._state)
        self._state = None

    # --- core driver — called after every turn ------------------------

    def evaluate_after_turn(self, *, user_initiated: bool = True) -> Dict[str, Any]:
        """Check interval and budget.  Return a decision dict.

        ``user_initiated`` distinguishes a real user prompt (True) from a
        continuation prompt we fed ourselves (False).

        Decision keys:
          - ``status``: current loop status after update
          - ``should_continue``: bool — caller should fire another turn
          - ``continuation_prompt``: str or None
          - ``verdict``: "fired" | "paused" | "inactive" | "budget"
          - ``reason``: str
          - ``message``: user-visible one-liner
        """
        state = self._state
        if state is None or state.status != "active":
            return {
                "status": state.status if state else None,
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "inactive",
                "reason": "no active loop",
                "message": "",
            }

        # Count the turn that just finished.
        state.turns_completed += 1

        # Check interval
        now = time.time()
        elapsed = now - state.last_fired_at
        if state.last_fired_at > 0 and elapsed < state.interval_seconds:
            return {
                "status": "active",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "fired",
                "reason": f"interval not elapsed ({int(elapsed)}s / {state.interval_seconds}s)",
                "message": "",
            }

        # Budget check
        if state.turns_completed >= self.default_max_turns:
            state.status = "paused"
            save_loop(self.session_id, state)
            return {
                "status": "paused",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "budget",
                "reason": f"turn budget exhausted ({state.turns_completed}/{self.default_max_turns})",
                "message": (
                    f"⏸ Loop paused — {state.turns_completed}/{self.default_max_turns} turns used. "
                    "Use /loop resume to keep going, or /loop clear to stop."
                ),
            }

        # Fire!
        state.last_fired_at = now
        save_loop(self.session_id, state)
        return {
            "status": "active",
            "should_continue": True,
            "continuation_prompt": f"[Loop check] {state.prompt}",
            "verdict": "fired",
            "reason": f"interval elapsed ({int(elapsed)}s / {state.interval_seconds}s)",
            "message": (
                f"↻ Loop check ({state.turns_completed}/{self.default_max_turns}): {state.prompt}"
            ),
        }


__all__ = [
    "LoopState",
    "LoopManager",
    "DEFAULT_MAX_TURNS",
    "load_loop",
    "save_loop",
    "_parse_interval",
]
