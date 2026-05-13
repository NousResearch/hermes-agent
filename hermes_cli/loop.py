"""Persistent session loops — background idle‑aware scheduler.

A loop is a prompt that repeats at a user‑specified interval.  A background
daemon thread ticks every second, checks whether the agent is idle (not
mid‑turn), and injects the loop prompt into the session's pending‑input
queue.  This matches Claude Code's /loop behaviour: non‑blocking, fires
only between turns, user can type at any time.

State is persisted in SessionDB's ``state_meta`` table keyed by
``loop:<session_id>`` so ``/loop resume`` picks it up.
"""

from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Optional

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
    """Per-session loop state + background scheduler.

    The CLI and gateway each hold one ``LoopManager`` per live session.
    When a loop is set, a background daemon thread starts ticking every
    second.  It only injects the loop prompt when the agent is idle.
    """

    def __init__(self, session_id: str, *, default_max_turns: int = DEFAULT_MAX_TURNS):
        self.session_id = session_id
        self.default_max_turns = int(default_max_turns or DEFAULT_MAX_TURNS)
        self._state: Optional[LoopState] = load_loop(session_id)
        self._scheduler: Optional[LoopScheduler] = None

    # --- introspection ------------------------------------------------

    @property
    def state(self) -> Optional[LoopState]:
        return self._state

    def is_active(self) -> bool:
        return self._state is not None and self._state.status == "active"

    def is_running(self) -> bool:
        return self._scheduler is not None and self._scheduler.running

    def status_line(self) -> str:
        s = self._state
        if s is None or s.status == "done":
            return "No active loop. Set one with /loop <prompt>."
        running = "running" if self.is_running() else "stopped"
        turns = f"{s.turns_completed}/{self.default_max_turns} turns"
        if s.status == "active":
            return f"⊙ Loop (active, {running}, {turns}): {s.prompt}"
        if s.status == "paused":
            return f"⏸ Loop (paused, {running}, {turns}): {s.prompt}"
        return f"Loop ({s.status}, {running}, {turns}): {s.prompt}"

    # --- mutation -----------------------------------------------------

    def set(self, prompt: str, *,
            interval_seconds: Optional[int] = None,
            pending_input: Optional[queue.Queue] = None,
            is_idle: Optional[Callable[[], bool]] = None,
            on_message: Optional[Callable[[str], None]] = None) -> LoopState:
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("loop prompt is empty")
        self._stop_scheduler()
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
        if pending_input is not None and is_idle is not None:
            self._scheduler = LoopScheduler(
                self.session_id, state,
                pending_input=pending_input,
                is_idle=is_idle,
                on_message=on_message,
            )
            self._scheduler.start()
        return state

    def pause(self, reason: str = "user-paused") -> Optional[LoopState]:
        if not self._state:
            return None
        self._state.status = "paused"
        save_loop(self.session_id, self._state)
        self._stop_scheduler()
        return self._state

    def resume(self, *,
               pending_input: Optional[queue.Queue] = None,
               is_idle: Optional[Callable[[], bool]] = None,
               on_message: Optional[Callable[[str], None]] = None) -> Optional[LoopState]:
        if not self._state:
            return None
        self._state.status = "active"
        save_loop(self.session_id, self._state)
        if pending_input is not None and is_idle is not None:
            self._scheduler = LoopScheduler(
                self.session_id, self._state,
                pending_input=pending_input,
                is_idle=is_idle,
                on_message=on_message,
            )
            self._scheduler.start()
        return self._state

    def clear(self) -> None:
        if self._state is None:
            return
        self._stop_scheduler()
        self._state.status = "done"
        save_loop(self.session_id, self._state)
        self._state = None

    def shutdown(self) -> None:
        """Stop scheduler — called on CLI exit."""
        self._stop_scheduler()

    # --- internal -----------------------------------------------------

    def _stop_scheduler(self) -> None:
        if self._scheduler is not None:
            try:
                self._scheduler.stop()
            except Exception as exc:
                logger.debug("LoopManager: scheduler stop failed: %s", exc)
            self._scheduler = None


# ──────────────────────────────────────────────────────────────────────
# Background scheduler
# ──────────────────────────────────────────────────────────────────────


class LoopScheduler:
    """Background daemon thread that ticks every second.

    On each tick:
    1. Checks ``is_idle()`` — if agent is busy, skip
    2. Computes elapsed since last fire
    3. If interval elapsed, injects loop prompt into ``pending_input``
    4. Updates state and persists

    This is the Claude Code ``createCronScheduler`` equivalent:
    non‑blocking, idle‑gated, cooperative.
    """

    TICK_INTERVAL = 1.0  # seconds — Claude Code uses 1s

    def __init__(
        self,
        session_id: str,
        state: LoopState,
        *,
        pending_input: queue.Queue,
        is_idle: Callable[[], bool],
        on_message: Optional[Callable[[str], None]] = None,
    ):
        self._session_id = session_id
        self._state = state
        self._pending_input = pending_input
        self._is_idle = is_idle
        self._on_message = on_message
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(
                target=self._tick_loop,
                name=f"loop-scheduler-{self._session_id[:8]}",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._running = False

    def _tick_loop(self) -> None:
        """Main scheduler loop — runs in daemon thread."""
        while self._running:
            time.sleep(self.TICK_INTERVAL)
            try:
                self._tick()
            except Exception as exc:
                logger.debug("LoopScheduler tick error: %s", exc)

    def _tick(self) -> None:
        """One tick of the scheduler.

        Claude Code equivalent: the ``v()`` function in ``createCronScheduler``.
        """
        # 1. Idle gate — skip if agent is busy (equivalent: isLoading())
        if not self._is_idle():
            return

        # 2. Reload state — might have been modified externally
        state = load_loop(self._session_id)
        if state is None:
            self._running = False
            return
        if state.status != "active":
            return
        self._state = state

        # 3. Interval check — has enough time elapsed?
        now = time.time()
        elapsed = now - state.last_fired_at
        if state.last_fired_at > 0 and elapsed < state.interval_seconds:
            return

        # 4. Fire!
        state.last_fired_at = now
        state.turns_completed += 1
        save_loop(self._session_id, state)

        prompt = f"[Loop check] {state.prompt}"
        try:
            self._pending_input.put(prompt)
        except Exception as exc:
            logger.debug("LoopScheduler: failed to enqueue prompt: %s", exc)
            return

        if self._on_message is not None:
            try:
                self._on_message(
                    f"↻ Loop check ({state.turns_completed}): {state.prompt}"
                )
            except Exception:
                pass


__all__ = [
    "LoopState",
    "LoopManager",
    "LoopScheduler",
    "DEFAULT_MAX_TURNS",
    "load_loop",
    "save_loop",
    "_parse_interval",
]
