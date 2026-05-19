"""Persistent session loops — background idle‑aware scheduler.

A loop is a prompt that repeats at a user‑specified interval.  A background
daemon thread ticks every second, checks whether the agent is idle (not
mid‑turn), and injects the loop prompt into the session.

Multiple loops coexist in one session, each with an auto‑generated UID.
No names — every ``/loop <interval> <prompt>`` creates a new loop.

State is persisted in SessionDB's ``state_meta`` table under
``loop:<session_id>:<uid>`` keys.  A registry at ``loop:<session_id>:__ids__``
tracks all active UIDs.
"""

from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Constants & defaults
# ──────────────────────────────────────────────────────────────────────

DEFAULT_MAX_TURNS = 20
MIN_INTERVAL_SECONDS = 60       # 1 minute minimum
DEFAULT_INTERVAL_SECONDS = 300   # 5 minutes


# ──────────────────────────────────────────────────────────────────────
# Dataclass
# ──────────────────────────────────────────────────────────────────────


@dataclass
class LoopState:
    """Serializable loop state stored per session."""

    prompt: str
    id: Optional[str] = None        # auto-generated short UID, e.g. "a3f1c2"; None = fork compat
    interval_seconds: int = 300
    status: str = "active"           # active | paused | done
    last_fired_at: float = 0.0
    created_at: float = 0.0
    turns_completed: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "LoopState":
        data = json.loads(raw)
        return cls(
            prompt=data.get("prompt", ""),
            id=data.get("id") or None,
            interval_seconds=int(data.get("interval_seconds", 300) or 300),
            status=data.get("status", "active"),
            last_fired_at=float(data.get("last_fired_at", 0.0) or 0.0),
            created_at=float(data.get("created_at", 0.0) or 0.0),
            turns_completed=int(data.get("turns_completed", 0) or 0),
        )


# ──────────────────────────────────────────────────────────────────────
# UID helpers
# ──────────────────────────────────────────────────────────────────────


def _gen_uid() -> str:
    """Generate a short loop UID — 6 hex chars from a UUID4."""
    return uuid.uuid4().hex[:6]


# ──────────────────────────────────────────────────────────────────────
# Persistence (SessionDB state_meta)
# ──────────────────────────────────────────────────────────────────────


def _loop_key(session_id: str, uid: str) -> str:
    return f"loop:{session_id}:{uid}"


def _ids_registry_key(session_id: str) -> str:
    return f"loop:{session_id}:__ids__"


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


def _load_ids(session_id: str) -> List[str]:
    """Load the list of loop UIDs from the registry."""
    if not session_id:
        return []
    db = _get_session_db()
    if db is None:
        return []
    try:
        raw = db.get_meta(_ids_registry_key(session_id))
    except Exception:
        return []
    if not raw:
        return []
    try:
        ids = json.loads(raw)
        if isinstance(ids, list):
            return [str(i) for i in ids]
    except Exception:
        pass
    return []


def _save_ids(session_id: str, ids: List[str]) -> None:
    if not session_id:
        return
    db = _get_session_db()
    if db is None:
        return
    try:
        if ids:
            db.set_meta(_ids_registry_key(session_id), json.dumps(ids))
        else:
            db.set_meta(_ids_registry_key(session_id), "")
    except Exception as exc:
        logger.debug("LoopManager: save_ids failed: %s", exc)


def _add_id_to_registry(session_id: str, uid: str) -> None:
    ids = _load_ids(session_id)
    if uid not in ids:
        ids.append(uid)
        _save_ids(session_id, ids)


def _remove_id_from_registry(session_id: str, uid: str) -> None:
    ids = _load_ids(session_id)
    if uid in ids:
        ids.remove(uid)
        _save_ids(session_id, ids)


def _load_loop(session_id: str, uid: str) -> Optional[LoopState]:
    """Load a single loop by UID from SessionDB."""
    if not session_id or not uid:
        return None
    db = _get_session_db()
    if db is None:
        return None
    try:
        raw = db.get_meta(_loop_key(session_id, uid))
    except Exception as exc:
        logger.debug("LoopManager: get_meta failed: %s", exc)
        return None
    if not raw:
        return None
    try:
        state = LoopState.from_json(raw)
    except Exception as exc:
        logger.warning("LoopManager: could not parse loop %s/%s: %s",
                       session_id, uid, exc)
        return None
    # Fix up id for old serialized loops (pre-fork JSON without "id" key).
    # The uid from the registry is authoritative — assign it if missing.
    if state.id is None:
        state.id = uid
    return state


def load_all_loops(session_id: str) -> Dict[str, LoopState]:
    """Load all loops for a session. Returns dict keyed by UID."""
    states: Dict[str, LoopState] = {}
    if not session_id:
        return states
    for uid in _load_ids(session_id):
        st = _load_loop(session_id, uid)
        if st is not None and st.status != "done":
            states[uid] = st
    return states


def _save_loop(session_id: str, state: LoopState) -> None:
    """Persist a loop to SessionDB. Auto-generates id if None."""
    if not session_id:
        return
    if state.id is None:
        state.id = _gen_uid()
    if not state.id:
        return
    db = _get_session_db()
    if db is None:
        return
    try:
        db.set_meta(_loop_key(session_id, state.id), state.to_json())
        if state.status != "done":
            _add_id_to_registry(session_id, state.id)
    except Exception as exc:
        logger.debug("LoopManager: set_meta failed: %s", exc)


def _del_loop_meta(session_id: str, uid: str) -> None:
    """Remove a loop's metadata from SessionDB and registry."""
    if not session_id or not uid:
        return
    db = _get_session_db()
    if db is None:
        return
    try:
        db.set_meta(_loop_key(session_id, uid), "")
    except Exception as exc:
        logger.debug("LoopManager: delete_meta failed: %s", exc)
    _remove_id_from_registry(session_id, uid)


def _del_all_loop_meta(session_id: str) -> None:
    """Remove ALL loop metadata for a session."""
    if not session_id:
        return
    db = _get_session_db()
    if db is None:
        return
    try:
        for uid in _load_ids(session_id):
            db.set_meta(_loop_key(session_id, uid), "")
        db.set_meta(_ids_registry_key(session_id), "")
    except Exception as exc:
        logger.debug("LoopManager: delete_all_meta failed: %s", exc)


# ──────────────────────────────────────────────────────────────────────
# Interval parsing
# ──────────────────────────────────────────────────────────────────────


def _parse_interval(token: str) -> Optional[int]:
    """Parse a duration token into seconds.

    Examples:
        "5m" → 300, "30m" → 1800, "2h" → 7200, "1d" → 86400
        "60" → 60 (bare number = seconds)
    """
    if not token:
        return None
    s = token.strip().lower()
    match = re.match(
        r"^(\d+)\s*(s|sec|secs|second|seconds|m|min|mins|minute|minutes|"
        r"h|hr|hrs|hour|hours|d|day|days)$",
        s,
    )
    if not match:
        bare = re.match(r"^(\d+)$", s)
        if bare:
            return int(bare.group(1))
        return None
    value = int(match.group(1))
    unit = match.group(2)[0]  # s, m, h, or d
    multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    return value * multipliers[unit]


# ──────────────────────────────────────────────────────────────────────
# Command parser
# ──────────────────────────────────────────────────────────────────────


def _parse_loop_command(arg: str) -> dict:
    """Parse a /loop command into an action dict.

    Every ``/loop [interval] <prompt>`` creates a **new** loop with
    an auto‑generated UID.  No names, no overwriting.

    Returns a dict with at least an ``action`` key:

    ================ ===================================================
    action           meaning
    ================ ===================================================
    ``"status"``     list all loops (``/loop list`` or ``/loop`` alone)
    ``"pause_all"``  pause every loop
    ``"pause"``      pause one loop (``uid`` key present, e.g. ``#a3f1``)
    ``"resume_all"`` resume every loop
    ``"resume"``     resume one loop (``uid`` key present)
    ``"clear_all"``  clear every loop
    ``"clear"``      clear one loop (``uid`` key present)
    ``"set"``        create a new loop (``interval``, ``prompt`` keys)
    ================ ===================================================
    """
    text = (arg or "").strip()

    # Strip leading "every " prefix
    if text.lower().startswith("every "):
        text = text[6:].strip()

    if not text:
        return {"action": "status"}

    tokens = text.split()
    first = tokens[0].lower()

    # --- status / list ---
    if first in ("status", "list"):
        return {"action": "status"}

    # --- pause ---
    if first == "pause":
        if len(tokens) == 1:
            return {"action": "pause_all"}
        target = tokens[1].lstrip("#")
        return {"action": "pause", "uid": target}

    # --- resume ---
    if first == "resume":
        if len(tokens) == 1:
            return {"action": "resume_all"}
        target = tokens[1].lstrip("#")
        return {"action": "resume", "uid": target}

    # --- clear / stop / done ---
    if first in ("clear", "stop", "done"):
        if len(tokens) == 1:
            return {"action": "clear_all"}
        target = tokens[1].lstrip("#")
        return {"action": "clear", "uid": target}

    # --- /loop <interval> <prompt> → new loop ---
    parsed = _parse_interval(tokens[0])
    if parsed is not None and len(tokens) > 1:
        interval = max(parsed, MIN_INTERVAL_SECONDS)
        prompt = " ".join(tokens[1:])
        return {"action": "set", "interval": interval, "prompt": prompt}

    # --- /loop <prompt> → new loop, default interval ---
    return {"action": "set",
            "interval": DEFAULT_INTERVAL_SECONDS,
            "prompt": text}


# ──────────────────────────────────────────────────────────────────────
# LoopManager
# ──────────────────────────────────────────────────────────────────────


class LoopManager:
    """Per-session loop state + background scheduler.

    The CLI and gateway each hold one ``LoopManager`` per live session.
    A background daemon thread ticks every second, checking all loops.

    If *dispatch* is provided, the scheduler is auto‑managed.  If
    *dispatch* is ``None`` (slash_worker mode), only persistence happens.
    """

    def __init__(self, session_id: str, *,
                 default_max_turns: int = DEFAULT_MAX_TURNS,
                 dispatch: Optional[Callable[[str], bool]] = None):
        self.session_id = session_id
        self.default_max_turns = int(default_max_turns or DEFAULT_MAX_TURNS)
        self._dispatch = dispatch
        self._states: Dict[str, LoopState] = load_all_loops(session_id)
        self._scheduler: Optional[LoopScheduler] = None

        # Fork-pattern scheduler state (pending_input / is_idle mode)
        self._pending_input: Optional[queue.Queue] = None
        self._is_idle_cb: Optional[Callable[[], bool]] = None
        self._on_message: Optional[Callable[[str], None]] = None

        # Always start scheduler when dispatch is available — it polls
        # SessionDB every tick and picks up loops persisted by the
        # slash_worker (which runs with dispatch=None).
        if dispatch is not None:
            self._start_scheduler()

    # --- introspection ------------------------------------------------

    @property
    def state(self) -> Optional[LoopState]:
        """Return the first non-done loop (active preferred, then paused)."""
        for s in self._states.values():
            if s.status == "active":
                return s
        for s in self._states.values():
            if s.status == "paused":
                return s
        return None

    @property
    def all_states(self) -> Dict[str, LoopState]:
        return self._states

    def is_active(self) -> bool:
        return any(s.status == "active" for s in self._states.values())

    def is_running(self) -> bool:
        return self._scheduler is not None and self._scheduler.running

    def status_line(self) -> str:
        """Multi-line status listing all loops with UIDs and countdown."""
        # Refresh from SessionDB so countdown reflects latest last_fired_at
        fresh = load_all_loops(self.session_id)
        if fresh:
            self._states = fresh
        if not self._states:
            return "No active loops. Set one with /loop [interval] <prompt>."
        lines = []
        now = time.time()
        running = "running" if self.is_running() else "stopped"
        for uid, s in sorted(self._states.items()):
            turns = f"{s.turns_completed}/{self.default_max_turns} turns"
            if s.status == "active":
                # Compute time remaining until next tick
                if s.last_fired_at <= 0:
                    next_str = "next: now"
                else:
                    remaining = int((s.last_fired_at + s.interval_seconds) - now)
                    if remaining <= 0:
                        next_str = "next: now"
                    elif remaining < 60:
                        next_str = f"next: {remaining}s"
                    elif remaining < 3600:
                        m, sec = divmod(remaining, 60)
                        next_str = f"next: {m}m {sec}s"
                    else:
                        h, rem = divmod(remaining, 3600)
                        m, sec = divmod(rem, 60)
                        next_str = f"next: {h}h {m}m"
                lines.append(f"⊙ Loop #{uid} (active, {running}, "
                             f"{s.interval_seconds}s, {next_str}, {turns}): {s.prompt}")
            elif s.status == "paused":
                lines.append(f"⏸ Loop #{uid} (paused, "
                             f"{s.interval_seconds}s, {turns}): {s.prompt}")
        return "\n".join(lines)

    # --- mutation -----------------------------------------------------

    def add(self, prompt: str, *,
            interval_seconds: int = DEFAULT_INTERVAL_SECONDS) -> LoopState:
        """Create a new loop with an auto‑generated UID."""
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("loop prompt is empty")
        effective_interval = max(int(interval_seconds), MIN_INTERVAL_SECONDS)
        state = LoopState(
            id=_gen_uid(),
            prompt=prompt,
            interval_seconds=effective_interval,
            status="active",
            last_fired_at=0.0,
            created_at=time.time(),
            turns_completed=0,
        )
        self._states[state.id] = state
        _save_loop(self.session_id, state)
        return state

    # Legacy set() alias — always creates new loop now
    def set(self, prompt: str, *,
            interval_seconds: Optional[int] = None,
            name: Optional[str] = None,
            pending_input: Optional[queue.Queue] = None,
            is_idle: Optional[Callable[[], bool]] = None,
            on_message: Optional[Callable[[str], None]] = None) -> LoopState:
        """Create a new loop.

        *name* is ignored (kept for backward compat).
        When both *pending_input* and *is_idle* are provided, the fork-pattern
        scheduler is auto-started, injecting prompts into the queue when idle.
        """
        # Store fork-pattern params for later resume() calls
        self._pending_input = pending_input
        self._is_idle_cb = is_idle
        self._on_message = on_message

        state = self.add(prompt,
                         interval_seconds=int(interval_seconds or DEFAULT_INTERVAL_SECONDS))

        # Start fork-pattern scheduler when both callbacks provided.
        # Guard: if a dispatch-mode scheduler is already running (from
        # __init__ dispatch= kwarg), do NOT replace it — the dispatch
        # scheduler already handles all loops including this new one.
        if pending_input is not None and is_idle is not None:
            if self._dispatch is not None and self._scheduler is not None:
                logger.debug(
                    "LoopManager: fork-pattern set() called but dispatch "
                    "scheduler already running — skipping fork scheduler "
                    "(dispatch handles all loops)"
                )
            else:
                self._stop_scheduler()
                self._scheduler = LoopScheduler(
                    self.session_id, state,
                    pending_input=pending_input,
                    is_idle=is_idle,
                    on_message=on_message,
                )
                self._scheduler.start()

        return state

    def pause(self, uid: Optional[str] = None, *,
              reason: str = "user-paused") -> List[LoopState]:
        """Pause one loop by UID, or all if uid is None.

        *reason* is accepted for fork backward compat (currently unused).
        In fork-pattern (pending_input) mode, also stops the scheduler.
        """
        if uid is not None:
            s = self._states.get(uid)
            if s is None:
                s = _load_loop(self.session_id, uid)
                if s is not None and s.status != "done":
                    self._states[uid] = s
            if s is None:
                return []
            s.status = "paused"
            _save_loop(self.session_id, s)
            # Stop scheduler in fork mode
            if self._pending_input is not None:
                self._stop_scheduler()
            return [s]

        paused = []
        for s in self._states.values():
            if s.status == "active":
                s.status = "paused"
                _save_loop(self.session_id, s)
                paused.append(s)
        # Stop scheduler in fork mode
        if paused and self._pending_input is not None:
            self._stop_scheduler()
        return paused

    def resume(self, uid: Optional[str] = None, *,
               pending_input: Optional[queue.Queue] = None,
               is_idle: Optional[Callable[[], bool]] = None,
               on_message: Optional[Callable[[str], None]] = None) -> List[LoopState]:
        """Resume one loop by UID, or all if uid is None.

        In fork-pattern mode, pass *pending_input* and *is_idle* to
        restart the scheduler on resume.
        """
        # Update stored fork-pattern params if provided
        if pending_input is not None:
            self._pending_input = pending_input
        if is_idle is not None:
            self._is_idle_cb = is_idle
        if on_message is not None:
            self._on_message = on_message

        if uid is not None:
            s = self._states.get(uid)
            if s is None:
                s = _load_loop(self.session_id, uid)
                if s is not None and s.status != "done":
                    self._states[uid] = s
            if s is None:
                return []
            s.status = "active"
            s.last_fired_at = 0.0  # fire immediately on next tick
            _save_loop(self.session_id, s)
            # Restart scheduler in fork mode
            if self._pending_input is not None and self._is_idle_cb is not None:
                self._stop_scheduler()
                self._scheduler = LoopScheduler(
                    self.session_id, s,
                    pending_input=self._pending_input,
                    is_idle=self._is_idle_cb,
                    on_message=self._on_message,
                )
                self._scheduler.start()
            return [s]

        resumed = []
        for s in self._states.values():
            if s.status == "paused":
                s.status = "active"
                s.last_fired_at = 0.0
                _save_loop(self.session_id, s)
                resumed.append(s)
        # Restart scheduler in fork mode
        if resumed and self._pending_input is not None and self._is_idle_cb is not None:
            active_state = self.state  # first active loop
            if active_state is not None:
                self._stop_scheduler()
                self._scheduler = LoopScheduler(
                    self.session_id, active_state,
                    pending_input=self._pending_input,
                    is_idle=self._is_idle_cb,
                    on_message=self._on_message,
                )
                self._scheduler.start()
        return resumed

    def clear(self, uid: Optional[str] = None) -> int:
        """Clear one loop by UID, or all if uid is None. Returns count."""
        if uid is not None:
            s = self._states.pop(uid, None)
            if s is None:
                return 0
            s.status = "done"
            _save_loop(self.session_id, s)
            _del_loop_meta(self.session_id, uid)
            # Stop scheduler in fork mode if no active loops remain
            if not self.is_active() and self._pending_input is not None:
                self._stop_scheduler()
            return 1

        count = len(self._states)
        for s in list(self._states.values()):
            s.status = "done"
            _save_loop(self.session_id, s)
        self._states.clear()
        _del_all_loop_meta(self.session_id)
        # Stop scheduler when clearing all loops
        self._stop_scheduler()
        return count

    def delete(self) -> bool:
        """Remove all loops from SessionDB entirely."""
        existed = bool(self._states)
        self._stop_scheduler()
        self._states.clear()
        try:
            _del_all_loop_meta(self.session_id)
        except Exception:
            pass
        return existed

    def shutdown(self) -> None:
        self._stop_scheduler()

    # --- internal -----------------------------------------------------

    def _start_scheduler(self) -> None:
        if self._dispatch is None:
            return
        self._stop_scheduler()
        self._scheduler = LoopScheduler(
            self.session_id,
            dispatch=self._dispatch,
        )
        self._scheduler.start()

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

    Supports two modes:

    1. **dispatch mode** (upstream multi-loop): pass ``dispatch`` — a
       callable that receives each prompt and returns ``True`` on success.
       The scheduler reloads all loops from SessionDB each tick and
       dispatches any active loop whose interval has elapsed.

    2. **pending_input mode** (fork single-loop): pass ``pending_input``
       (a ``queue.Queue``) and ``is_idle`` (a ``Callable[[], bool]``).
       The scheduler checks ``is_idle()`` before each tick, and injects
       the loop prompt into ``pending_input`` when the interval elapses.
    """

    TICK_INTERVAL = 1.0

    def __init__(self, session_id: str, state: Optional[LoopState] = None, *,
                 dispatch: Optional[Callable[[str], bool]] = None,
                 pending_input: Optional[queue.Queue] = None,
                 is_idle: Optional[Callable[[], bool]] = None,
                 on_message: Optional[Callable[[str], None]] = None):
        self._session_id = session_id
        self._state = state
        self._dispatch = dispatch
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
        while self._running:
            time.sleep(self.TICK_INTERVAL)
            try:
                self._tick()
            except Exception as exc:
                logger.debug("LoopScheduler tick error: %s", exc)

    def _tick(self) -> None:
        # --- pending_input / is_idle mode (fork pattern) ---
        if self._pending_input is not None and self._is_idle is not None:
            # Idle gate — skip if agent is busy
            if not self._is_idle():
                return

            # Reload the state from DB if we have a uid
            if self._state is not None and self._state.id:
                reloaded = _load_loop(self._session_id, self._state.id)
                if reloaded is None:
                    return
                self._state = reloaded

            state = self._state
            if state is None or state.status != "active":
                return

            now = time.time()
            if state.last_fired_at > 0 and (now - state.last_fired_at) < state.interval_seconds:
                return

            # Fire — inject prompt into pending_input queue
            state.last_fired_at = now
            state.turns_completed += 1
            _save_loop(self._session_id, state)

            try:
                self._pending_input.put(state.prompt)
            except Exception as exc:
                logger.debug("LoopScheduler: failed to enqueue prompt: %s", exc)
            return

        # --- dispatch mode (upstream multi-loop pattern) ---
        states = load_all_loops(self._session_id)
        if not states:
            return

        now = time.time()
        for state in states.values():
            if state.status != "active":
                continue
            if (state.last_fired_at > 0
                    and (now - state.last_fired_at) < state.interval_seconds):
                continue
            if self._dispatch is not None and self._dispatch(state.prompt):
                state.last_fired_at = now
                state.turns_completed += 1
                _save_loop(self._session_id, state)


__all__ = [
    "LoopState",
    "LoopManager",
    "LoopScheduler",
    "DEFAULT_MAX_TURNS",
    "MIN_INTERVAL_SECONDS",
    "load_all_loops",
    "_parse_interval",
    "_parse_loop_command",
    "delete_loop",
]


# ──────────────────────────────────────────────────────────────────────
# Convenience helpers
# ──────────────────────────────────────────────────────────────────────


def delete_loop(session_id: str, uid: Optional[str] = None) -> bool:
    """Delete loop(s) from SessionDB without creating a LoopManager.

    If *uid* is None, deletes all loops.  Returns True if anything deleted.
    """
    if uid is not None:
        state = _load_loop(session_id, uid)
        if state is None:
            return False
        _del_loop_meta(session_id, uid)
        return True

    states = load_all_loops(session_id)
    if not states:
        return False
    _del_all_loop_meta(session_id)
    return True


# Public aliases for external callers (tests, gateway)
save_loop = _save_loop


def load_loop(session_id: str, uid: Optional[str] = None) -> Optional[LoopState]:
    """Load a loop from SessionDB.

    Backward-compatible: if *uid* is None, returns the first
    non-done loop for the session (fork single-loop pattern).
    """
    if uid is not None:
        return _load_loop(session_id, uid)
    # Fork compat: load all, return first non-done loop
    states = load_all_loops(session_id)
    for s in states.values():
        if s.status != "done":
            return s
    return None
