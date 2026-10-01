"""Session heartbeats — recurring re-entry prompts for the current session.

Session-scoped and in-process (CLI or gateway must be running); durable cross-process scheduling stays
``hermes cron``. Invariants (mirrors goals.py): injection is a plain user message — no system-prompt
mutation or toolset swap, so prompt caching stays intact — and a real user message always wins:
heartbeats only fire into an idle session with an empty input queue."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

MIN_INTERVAL_SECONDS = 60  # floor: re-entering more often than once a minute is a busy-loop, not a heartbeat
POLL_SECONDS = 5.0  # how often drivers poll for due heartbeats; not user-facing

# Reserved meta-key id for the profile-wide heartbeat. Session keys are opaque ids, so a collision is
# possible in principle; the sentinel is rejected as a session id so it can never be shadowed.
PROFILE_SCOPE_KEY = "__profile__"

# Outcomes of taking the profile's shared tick. Distinct sentinels, not booleans: "someone else fired this
# tick" must silence the session, while "the profile heartbeat is gone" must let it carry on alone.
_PROFILE_TICK_CLAIMED = "claimed"
_PROFILE_TICK_LOST = "lost"
_PROFILE_TICK_DETACHED = "detached"

HEARTBEAT_PROMPT_TEMPLATE = (
    "[Heartbeat — recurring instruction, fires every {interval}]\n{prompt}\n\n"
    "If there is nothing meaningful to do or report for this instruction "
    "right now, reply briefly that nothing has changed and stop — do not invent work."
)

_INTERVAL_RE = re.compile(
    r"^\s*(?:every\s+)?(\d+(?:\.\d+)?)\s*(s|sec|secs|seconds?|m|min|mins|minutes?|h|hr|hrs|hours?|d|days?)\s*$", re.IGNORECASE)

_UNIT_SECONDS = {
    **dict.fromkeys(("s", "sec", "secs", "second", "seconds"), 1),
    **dict.fromkeys(("m", "min", "mins", "minute", "minutes"), 60),
    **dict.fromkeys(("h", "hr", "hrs", "hour", "hours"), 3600),
    **dict.fromkeys(("d", "day", "days"), 86400),
}

# field -> (coercer, default used when the stored value is missing/falsy)
_STATE_FIELDS = {
    "prompt": (str, ""), "interval_seconds": (int, 0), "status": (str, "active"),
    "created_at": (float, 0.0), "last_fired_at": (float, 0.0), "fire_count": (int, 0),
    # Provenance: this row was adopted from the profile heartbeat, so it still fires on the profile's
    # cadence. Persisted because drivers rebuild the manager on every poll and cannot rely on instance
    # state. Cleared the moment the user sets, pauses, resumes or clears the session's own heartbeat.
    "from_profile": (bool, False),
}


def parse_interval(text: str) -> Optional[int]:
    """Parse ``10m`` / ``every 2h`` / ``every 90 minutes`` into seconds.

    None when not an interval; below ``MIN_INTERVAL_SECONDS`` returns -1 so callers can tell "too small" apart.
    """
    m = _INTERVAL_RE.match(text) if text else None
    if not m:
        return None
    seconds = int(float(m.group(1)) * _UNIT_SECONDS[m.group(2).lower()])
    return -1 if seconds < MIN_INTERVAL_SECONDS else seconds


def format_interval(seconds: int) -> str:
    """Human-readable interval (``600`` → ``10m``)."""
    seconds = int(seconds)
    units = ((86400, "d"), (3600, "h"), (60, "m"))
    return next((f"{seconds // unit}{suffix}" for unit, suffix in units if seconds % unit == 0), f"{seconds}s")


@dataclass
class HeartbeatState:
    """Serializable per-session heartbeat."""

    prompt: str
    interval_seconds: int
    status: str = "active"          # active | paused | cleared
    created_at: float = 0.0
    last_fired_at: float = 0.0
    fire_count: int = 0
    from_profile: bool = False      # adopted from the profile heartbeat; still on the profile's cadence

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "HeartbeatState":
        data = json.loads(raw)
        return cls(**{name: coerce(data.get(name) or default) for name, (coerce, default) in _STATE_FIELDS.items()})

    def is_due(self, now: Optional[float] = None) -> bool:
        if self.status != "active" or not self.prompt or self.interval_seconds <= 0:
            return False
        return (time.time() if now is None else now) - (self.last_fired_at or self.created_at) >= self.interval_seconds

    def render_prompt(self) -> str:
        return HEARTBEAT_PROMPT_TEMPLATE.format(interval=format_interval(self.interval_seconds), prompt=self.prompt)


def _get_session_db() -> Optional[Any]:
    """Persistence goes through the goals module's per-HERMES_HOME cached SessionDB (one shared connection)."""
    try:
        from hermes_cli.goals import _get_session_db as _goals_db
        return _goals_db()
    except Exception as exc:  # pragma: no cover
        logger.debug("HeartbeatManager: SessionDB bootstrap failed (%s)", exc)
        return None


_META_PREFIX = "heartbeat:"


def load_heartbeat(session_id: str) -> Optional[HeartbeatState]:
    db = _get_session_db() if session_id else None
    if db is None:
        return None
    try:
        raw = db.get_meta(_META_PREFIX + session_id)
    except Exception as exc:
        logger.debug("HeartbeatManager: get_meta failed: %s", exc)
        return None
    try:
        state = HeartbeatState.from_json(raw) if raw else None
    except Exception as exc:
        logger.warning("HeartbeatManager: could not parse stored heartbeat for %s: %s", session_id, exc)
        return None
    return None if state is None or state.status == "cleared" else state


def store_has_active_heartbeat(db: Any) -> bool:
    """True when *db* holds an ACTIVE ``heartbeat:*`` row — or one that cannot be parsed (unknown, so
    the caller keeps its full sweep). ``clear``/``pause`` keep their rows (status ``cleared``/``paused``),
    so key existence alone is not "active". Read errors propagate: "unavailable" is the caller's call."""
    for _key, raw in db.list_meta_prefix(_META_PREFIX):
        try:
            if HeartbeatState.from_json(raw).status == "active":
                return True
        except Exception:
            return True
    return False


def _write_scope(key: str, state: HeartbeatState) -> None:
    """Persist under a raw scope key: a session id, or :data:`PROFILE_SCOPE_KEY` for the profile-wide row."""
    if not key:
        return
    db = _get_session_db()
    if db is None:
        from hermes_cli.goals import _warn_dropped_write
        _warn_dropped_write("HeartbeatManager", "heartbeat", key)
        return
    try:
        db.set_meta(_META_PREFIX + key, state.to_json())
    except Exception as exc:
        logger.debug("HeartbeatManager: set_meta failed: %s", exc)


def save_heartbeat(session_id: str, state: HeartbeatState) -> None:
    """Session-scoped write. The profile sentinel is not a session id, so it is refused here rather than
    letting a session that happens to carry that id shadow the profile-wide heartbeat."""
    if not session_id or session_id == PROFILE_SCOPE_KEY:
        return
    _write_scope(session_id, state)


def promote_session_heartbeat_to_profile(session_id: str) -> Optional[HeartbeatState]:
    """Copy *session_id*'s heartbeat to the profile scope, leaving the session's own row untouched.

    The profile row starts a fresh clock, so sessions that pick it up later wait a full interval instead of
    inheriting this session's fire history. Returns None when the session has no heartbeat to promote.
    """
    if not session_id or session_id == PROFILE_SCOPE_KEY:
        return None
    state = load_heartbeat(session_id)
    if state is None or not state.prompt:
        return None
    promoted = HeartbeatState(prompt=state.prompt, interval_seconds=state.interval_seconds,
                              status="active", created_at=time.time())
    _write_scope(PROFILE_SCOPE_KEY, promoted)
    return promoted


class HeartbeatManager:
    """Per-session heartbeat state + due-tick decisions; the surface CLI + gateway talk to.

    Drivers (CLI thread / gateway task) call :meth:`due_prompt` on a poll cadence while the session is
    idle; a non-None return is the user-role message to inject.

    ``scope="session"`` (the default) is the long-standing behaviour. ``scope="profile"`` addresses the
    profile-wide standing instruction instead of one conversation, so a new session starts with the
    behaviour already armed.

    A session manager opened in session scope *inherits* an active profile heartbeat: it reports one so
    status reads honestly, and follows the profile's own clock rather than a private one. The profile row
    stays the single cadence for the whole profile, so exactly one session fires per interval — the first
    idle one to claim it. That firing session materialises the heartbeat under its own key, so from then on
    it keeps its own history and the refund path works. Without that copy, two open sessions would share
    one ``last_fired_at`` and silently steal ticks from each other, and clearing a session heartbeat would
    disarm the profile one as collateral.
    """

    def __init__(self, session_id: str, scope: str = "session"):
        self.session_id = session_id
        self.scope = "profile" if scope == "profile" else "session"
        if self.scope == "session" and session_id == PROFILE_SCOPE_KEY:
            # A caller passing the sentinel without asking for profile scope would silently read and write
            # the shared row. Refuse it instead of aliasing the two scopes onto one key.
            raise ValueError(f"{PROFILE_SCOPE_KEY!r} is reserved for the profile heartbeat scope")
        self._state: Optional[HeartbeatState] = load_heartbeat(self._key)
        self._inherited = False
        if self._state is None and self.scope == "session":
            inherited = load_heartbeat(PROFILE_SCOPE_KEY)
            if inherited is not None and inherited.status == "active":
                # Adopt the profile's own anchor, not "now": gateway and TUI pollers build a fresh manager
                # every few seconds, so a per-instance anchor would reset forever and never come due.
                self._state = HeartbeatState(
                    prompt=inherited.prompt, interval_seconds=inherited.interval_seconds,
                    status="active", created_at=inherited.last_fired_at or inherited.created_at,
                    from_profile=True,
                )
                self._inherited = True
        self._last_claim: Optional[tuple[float, int]] = None  # (last_fired_at, fire_count) before the last due_prompt
        self._profile_claim: Optional[tuple[float, int]] = None  # same, for a tick claimed off the profile row

    @property
    def _key(self) -> str:
        """The meta-key id this manager reads and writes."""
        return PROFILE_SCOPE_KEY if self.scope == "profile" else self.session_id

    @property
    def is_inherited(self) -> bool:
        """True while the state is a profile heartbeat this session has not fired yet."""
        return self._inherited

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
            next_in = max(0, int((s.last_fired_at or s.created_at) + s.interval_seconds - time.time()))
            return f"♥ Heartbeat (every {every}, next in ~{next_in}s{fired}): {s.prompt}"
        icon = "⏸ " if s.status == "paused" else ""
        return f"{icon}Heartbeat ({s.status}, every {every}{fired}): {s.prompt}"

    def set(self, prompt: str, interval_seconds: int) -> HeartbeatState:
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("heartbeat prompt is empty")
        interval_seconds = int(interval_seconds)
        if interval_seconds < MIN_INTERVAL_SECONDS:
            raise ValueError(f"interval must be at least {MIN_INTERVAL_SECONDS}s")
        self._state = HeartbeatState(prompt=prompt, interval_seconds=interval_seconds, status="active",
                                     created_at=time.time(), from_profile=False)
        self._inherited = False  # an explicit set supersedes the inherited profile default
        _write_scope(self._key, self._state)
        return self._state

    def _set_status(self, status: str, *, reanchor: bool = False) -> Optional[HeartbeatState]:
        if not self._state:
            return None
        self._state.status = status
        if reanchor:
            self._state.last_fired_at = time.time()
        # Taking manual control of the session heartbeat detaches it from the profile's cadence.
        self._state.from_profile = False
        self._inherited = False
        _write_scope(self._key, self._state)
        return self._state

    def pause(self) -> Optional[HeartbeatState]:
        return self._set_status("paused")

    def resume(self) -> Optional[HeartbeatState]:
        # Re-anchor so resuming doesn't instantly fire a stale tick.
        return self._set_status("active", reanchor=True)

    def clear(self) -> bool:
        cleared = self._set_status("cleared") is not None
        self._state = None
        return cleared

    def due_prompt(self, now: Optional[float] = None) -> Optional[str]:
        """Return the injection prompt if the heartbeat is due, else None.

        The fire is recorded immediately (before the turn runs) so overlapping polls or a long turn can never
        double-fire the same tick. Missed ticks coalesce: the anchor resets to NOW, not the theoretical
        schedule. A profile heartbeat being inherited is written under this session here, which is the point
        where it stops being a shared default and becomes this session's own standing instruction.
        """
        s = self._state
        if s is None or not s.is_due(now):
            return None
        if s.from_profile:
            claim = self._claim_profile_tick(now)
            if claim is _PROFILE_TICK_LOST:
                return None  # another session in this profile took this tick; stay quiet until the next one
            if claim is _PROFILE_TICK_DETACHED:
                s = self._state  # the profile heartbeat is gone; this session carries on alone
        self._last_claim = (s.last_fired_at, s.fire_count)
        s.last_fired_at = now if now is not None else time.time()
        s.fire_count += 1
        self._inherited = False
        _write_scope(self._key, s)
        return s.render_prompt()

    def _claim_profile_tick(self, now: Optional[float]) -> str:
        """Take the profile's tick for this session, so exactly one session fires per interval.

        The profile row is the cadence for every session still following it, so a session that joins late
        fires on the profile's schedule instead of replaying a backlog, and two idle sessions never both
        fire the same tick. Returns :data:`_PROFILE_TICK_CLAIMED` on success, :data:`_PROFILE_TICK_LOST`
        when another session already took this tick, or :data:`_PROFILE_TICK_DETACHED` when the profile
        heartbeat is gone — the caller then fires on the session's own clock instead of going silent.
        """
        current = load_heartbeat(PROFILE_SCOPE_KEY)
        s = self._state
        if s is None:
            return _PROFILE_TICK_LOST
        if current is None or current.status != "active":
            # The profile heartbeat was cleared or paused: this session keeps the copy it already has and
            # becomes self-governing, rather than being stranded on a cadence that no longer exists.
            s.from_profile = False
            _write_scope(self._key, s)
            self._inherited = False
            return _PROFILE_TICK_DETACHED
        if (current.prompt, current.interval_seconds) != (s.prompt, s.interval_seconds):
            # The profile was re-set: follow the new instruction instead of firing stale text, and re-anchor
            # onto its clock so the change takes effect on the next tick rather than replaying this one.
            s.prompt, s.interval_seconds = current.prompt, current.interval_seconds
            s.last_fired_at = current.last_fired_at or current.created_at
            _write_scope(self._key, s)
        if not current.is_due(now):
            return _PROFILE_TICK_LOST  # another session already fired this tick
        self._profile_claim = (current.last_fired_at, current.fire_count)
        current.last_fired_at = now if now is not None else time.time()
        current.fire_count += 1
        _write_scope(PROFILE_SCOPE_KEY, current)
        return _PROFILE_TICK_CLAIMED

    def abandon_fire(self) -> bool:
        """Rewind the fire recorded by the last :meth:`due_prompt` whose turn never started, so the tick stays
        due for the next poll instead of being silently consumed. Mirrors ``LoopManager.abandon_tick``. Skipped
        (False) when the persisted state moved on — a pause/resume/clear that landed in between wins."""
        claim, s = self._last_claim, self._state
        if claim is None or s is None:
            return False
        current = load_heartbeat(self._key)
        if current is None or current.status != "active" or (current.last_fired_at, current.fire_count) != (
                s.last_fired_at, s.fire_count):
            return False
        s.last_fired_at, s.fire_count = claim
        self._last_claim = None
        _write_scope(self._key, s)
        self._refund_profile_tick()
        return True

    def _refund_profile_tick(self) -> None:
        """Hand the profile tick back too: it was one shared tick, so an unstarted turn must not consume it."""
        claim = self._profile_claim
        if claim is None:
            return
        self._profile_claim = None
        current = load_heartbeat(PROFILE_SCOPE_KEY)
        if current is None or current.status != "active":
            return
        current.last_fired_at, current.fire_count = claim
        _write_scope(PROFILE_SCOPE_KEY, current)


def migrate_heartbeat_to_session(old_session_id: str, new_session_id: str) -> bool:
    """Carry a heartbeat across a compression session rotation (copy to child, archive parent, never raise).

    Same shape as ``goals.migrate_goal_to_session``.
    """
    if not old_session_id or not new_session_id or old_session_id == new_session_id:
        return False
    if PROFILE_SCOPE_KEY in (old_session_id, new_session_id):
        return False  # the profile row is not a session; rotating a conversation never moves it
    try:
        state = load_heartbeat(old_session_id)
        if state is None or load_heartbeat(new_session_id) is not None:
            return False
        save_heartbeat(new_session_id, state)
        state.status = "cleared"
        save_heartbeat(old_session_id, state)
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("HeartbeatManager: migration failed: %s", exc)
        return False


__all__ = [
    "HeartbeatState", "HeartbeatManager", "parse_interval", "format_interval", "load_heartbeat", "save_heartbeat",
    "migrate_heartbeat_to_session", "promote_session_heartbeat_to_profile", "HEARTBEAT_PROMPT_TEMPLATE",
    "MIN_INTERVAL_SECONDS", "POLL_SECONDS", "PROFILE_SCOPE_KEY",
]
