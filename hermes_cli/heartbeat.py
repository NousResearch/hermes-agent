"""Session heartbeats — recurring re-entry prompts for the current session.

A heartbeat is one user-owned recurring instruction bound to a session
(`/heartbeat every 10m Check the deployment and report meaningful changes`).
When due AND the session is idle, the prompt is injected as a normal user
turn — same mechanism as a /goal continuation, so message-role alternation
and prompt caching are untouched. If the agent is busy at the due moment,
the tick coalesces: it fires once when the session next goes idle, never
stacking a backlog.

This is deliberately session-scoped and in-process (CLI process or gateway
process must be running) — the durable cross-process scheduling surface
remains ``hermes cron`` / the ``cronjob`` tool, which runs in isolated
sessions. A heartbeat is for "keep re-entering THIS conversation", the
cron system is for "run this job on a schedule". Distinct by design.

State is persisted in SessionDB ``state_meta`` keyed by
``heartbeat:<session_id>`` so ``/resume`` picks it up.

Invariants (mirrors goals.py):
- Injection is a plain user message. No system-prompt mutation, no toolset
  swap — prompt caching stays intact.
- A real user message always wins: heartbeats only fire into an idle
  session with an empty input queue.
- Failures are contained: any DB/import error degrades to "no heartbeat",
  never to a crashed input loop.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# Floor: a heartbeat that re-enters the session more often than once a
# minute is a busy-loop, not a heartbeat. (Prime-Agent uses a similar floor.)
MIN_INTERVAL_SECONDS = 60
# How often drivers poll for due heartbeats. Not user-facing.
POLL_SECONDS = 5.0

HEARTBEAT_PROMPT_TEMPLATE = (
    "[Heartbeat — recurring instruction, fires every {interval}]\n"
    "{prompt}\n\n"
    "If there is nothing meaningful to do or report for this instruction "
    "right now, reply briefly that nothing has changed and stop — do not "
    "invent work."
)

_INTERVAL_RE = re.compile(
    r"^\s*(?:every\s+)?(\d+(?:\.\d+)?)\s*(s|sec|secs|seconds?|m|min|mins|minutes?|h|hr|hrs|hours?|d|days?)\s*$",
    re.IGNORECASE,
)

_UNIT_SECONDS = {
    "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
    "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
    "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
    "d": 86400, "day": 86400, "days": 86400,
}


def parse_interval(text: str) -> Optional[int]:
    """Parse ``10m`` / ``every 2h`` / ``every 90 minutes`` into seconds.

    Returns None when the text is not an interval. Values below
    ``MIN_INTERVAL_SECONDS`` are rejected (returns -1 so callers can
    distinguish "not an interval" from "too small").
    """
    if not text:
        return None
    m = _INTERVAL_RE.match(text)
    if not m:
        return None
    value = float(m.group(1))
    unit = m.group(2).lower()
    seconds = int(value * _UNIT_SECONDS[unit])
    if seconds < MIN_INTERVAL_SECONDS:
        return -1
    return seconds


def format_interval(seconds: int) -> str:
    """Human-readable interval (``600`` → ``10m``)."""
    seconds = int(seconds)
    if seconds % 86400 == 0:
        return f"{seconds // 86400}d"
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


@dataclass
class HeartbeatState:
    """Serializable per-session heartbeat."""

    prompt: str
    interval_seconds: int
    status: str = "active"          # active | paused | cleared
    created_at: float = 0.0
    last_fired_at: float = 0.0
    fire_count: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "HeartbeatState":
        data = json.loads(raw)
        return cls(
            prompt=str(data.get("prompt") or ""),
            interval_seconds=int(data.get("interval_seconds", 0) or 0),
            status=str(data.get("status") or "active"),
            created_at=float(data.get("created_at", 0.0) or 0.0),
            last_fired_at=float(data.get("last_fired_at", 0.0) or 0.0),
            fire_count=int(data.get("fire_count", 0) or 0),
        )

    def is_due(self, now: Optional[float] = None) -> bool:
        if self.status != "active" or not self.prompt or self.interval_seconds <= 0:
            return False
        now = now if now is not None else time.time()
        anchor = self.last_fired_at or self.created_at
        return (now - anchor) >= self.interval_seconds

    def render_prompt(self) -> str:
        return HEARTBEAT_PROMPT_TEMPLATE.format(
            interval=format_interval(self.interval_seconds),
            prompt=self.prompt,
        )


# ──────────────────────────────────────────────────────────────────────
# Persistence (SessionDB state_meta) — same pattern as goals.py
# ──────────────────────────────────────────────────────────────────────


def _meta_key(session_id: str) -> str:
    return f"heartbeat:{session_id}"


def _get_session_db() -> Optional[Any]:
    # Reuse the goals module's per-HERMES_HOME cached SessionDB so both
    # features share one connection instead of thrashing the file.
    try:
        from hermes_cli.goals import _get_session_db as _goals_db

        return _goals_db()
    except Exception as exc:  # pragma: no cover
        logger.debug("HeartbeatManager: SessionDB bootstrap failed (%s)", exc)
        return None


def load_heartbeat(session_id: str) -> Optional[HeartbeatState]:
    if not session_id:
        return None
    db = _get_session_db()
    if db is None:
        return None
    try:
        raw = db.get_meta(_meta_key(session_id))
    except Exception as exc:
        logger.debug("HeartbeatManager: get_meta failed: %s", exc)
        return None
    if not raw:
        return None
    try:
        state = HeartbeatState.from_json(raw)
    except Exception as exc:
        logger.warning("HeartbeatManager: could not parse stored heartbeat for %s: %s", session_id, exc)
        return None
    return None if state.status == "cleared" else state


def save_heartbeat(session_id: str, state: HeartbeatState) -> None:
    if not session_id:
        return
    db = _get_session_db()
    if db is None:
        logger.warning(
            "HeartbeatManager: heartbeat for %s not persisted — session "
            "DB unavailable (bootstrap window exceeded, in-memory state "
            "still active)",
            session_id,
        )
        return
    try:
        db.set_meta(_meta_key(session_id), state.to_json())
    except Exception as exc:
        logger.debug("HeartbeatManager: set_meta failed: %s", exc)


# ──────────────────────────────────────────────────────────────────────
# Manager — the surface CLI + gateway talk to
# ──────────────────────────────────────────────────────────────────────


class HeartbeatManager:
    """Per-session heartbeat state + due-tick decisions.

    Drivers (CLI thread / gateway task) call :meth:`due_prompt` on a poll
    cadence while the session is idle; a non-None return is the user-role
    message to inject. Firing is recorded immediately so a slow turn can't
    double-fire.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._state: Optional[HeartbeatState] = load_heartbeat(session_id)

    @property
    def state(self) -> Optional[HeartbeatState]:
        return self._state

    def has_heartbeat(self) -> bool:
        return self._state is not None and self._state.status in {"active", "paused"}

    def is_active(self) -> bool:
        return self._state is not None and self._state.status == "active"

    def status_line(self) -> str:
        s = self._state
        if s is None:
            return "No heartbeat. Set one with /heartbeat every <interval> <prompt>."
        every = format_interval(s.interval_seconds)
        fired = f", fired {s.fire_count}×" if s.fire_count else ""
        if s.status == "active":
            anchor = s.last_fired_at or s.created_at
            next_in = max(0, int(anchor + s.interval_seconds - time.time()))
            return f"♥ Heartbeat (every {every}, next in ~{next_in}s{fired}): {s.prompt}"
        if s.status == "paused":
            return f"⏸ Heartbeat (paused, every {every}{fired}): {s.prompt}"
        return f"Heartbeat ({s.status}, every {every}{fired}): {s.prompt}"

    # --- mutation -----------------------------------------------------

    def set(self, prompt: str, interval_seconds: int) -> HeartbeatState:
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("heartbeat prompt is empty")
        interval_seconds = int(interval_seconds)
        if interval_seconds < MIN_INTERVAL_SECONDS:
            raise ValueError(f"interval must be at least {MIN_INTERVAL_SECONDS}s")
        state = HeartbeatState(
            prompt=prompt,
            interval_seconds=interval_seconds,
            status="active",
            created_at=time.time(),
        )
        self._state = state
        save_heartbeat(self.session_id, state)
        return state

    def pause(self) -> Optional[HeartbeatState]:
        if not self._state:
            return None
        self._state.status = "paused"
        save_heartbeat(self.session_id, self._state)
        return self._state

    def resume(self) -> Optional[HeartbeatState]:
        if not self._state:
            return None
        self._state.status = "active"
        # Re-anchor so resuming doesn't instantly fire a stale tick.
        self._state.last_fired_at = time.time()
        save_heartbeat(self.session_id, self._state)
        return self._state

    def clear(self) -> bool:
        if self._state is None:
            return False
        self._state.status = "cleared"
        save_heartbeat(self.session_id, self._state)
        self._state = None
        return True

    # --- driver entry point --------------------------------------------

    def due_prompt(self, now: Optional[float] = None) -> Optional[str]:
        """Return the injection prompt if the heartbeat is due, else None.

        Records the fire immediately (before the turn runs) so overlapping
        polls or a long turn can never double-fire the same tick. Missed
        ticks coalesce into one — the anchor resets to NOW, not to the
        theoretical schedule.
        """
        s = self._state
        if s is None or not s.is_due(now):
            return None
        s.last_fired_at = now if now is not None else time.time()
        s.fire_count += 1
        save_heartbeat(self.session_id, s)
        return s.render_prompt()


def migrate_heartbeat_to_session(old_session_id: str, new_session_id: str) -> bool:
    """Carry a heartbeat across a compression session rotation.

    Same shape as ``goals.migrate_goal_to_session`` — copy to the child,
    archive the parent row, never raise.
    """
    if not old_session_id or not new_session_id or old_session_id == new_session_id:
        return False
    try:
        state = load_heartbeat(old_session_id)
        if state is None:
            return False
        if load_heartbeat(new_session_id) is not None:
            return False
        save_heartbeat(new_session_id, state)
        state.status = "cleared"
        save_heartbeat(old_session_id, state)
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("HeartbeatManager: migration failed: %s", exc)
        return False


__all__ = [
    "HeartbeatState",
    "HeartbeatManager",
    "parse_interval",
    "format_interval",
    "load_heartbeat",
    "save_heartbeat",
    "migrate_heartbeat_to_session",
    "HEARTBEAT_PROMPT_TEMPLATE",
    "MIN_INTERVAL_SECONDS",
    "POLL_SECONDS",
]
