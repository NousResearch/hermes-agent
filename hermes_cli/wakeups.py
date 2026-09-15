"""One-shot session wakeups — the agent defers its OWN continuation to a later time.

``/heartbeat`` and ``/loop`` are user-set recurring schedules; a wakeup is agent-set and fires
once: "check the deploy again in 20 minutes", "re-poll CI in an hour". The prompt the agent
chose re-enters THIS session as a plain user-role turn (same context, same prompt cache), so it
is the in-session counterpart of a one-shot cron job (which always runs in a fresh session).

State lives in SessionDB ``state_meta`` under ``wakeups:<session_id>`` (same contract as
``goals.py`` / ``loops.py`` / ``heartbeat.py``); the CLI watchdog, the TUI/Desktop notification
poller and the gateway idle watcher all drive it through :class:`WakeupManager`, so every surface
shares one fire/abandon discipline: a due wakeup is claimed (removed) before the turn is
dispatched, and a dispatch that never starts a turn re-arms it (``abandon``) instead of
silently consuming it.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Floor: anything shorter belongs in a blocking ``terminal`` wait, not a scheduler round-trip.
MIN_DELAY_SECONDS = 10
# Ceiling: a week. Longer horizons are a cron job (durable across reinstalls, deliverable anywhere).
MAX_DELAY_SECONDS = 7 * 86400
# Per-session cap so a confused model can't queue hundreds of self-pings.
MAX_PER_SESSION = 10

_META_PREFIX = "wakeups:"

WAKEUP_PROMPT_TEMPLATE = (
    "[Scheduled wakeup {wakeup_id} — you set this {ago} ago{reason}]\n{prompt}\n\n"
    "This turn was started by the wakeup you scheduled, not by the user; no one is necessarily "
    "present. Re-check the CURRENT state (don't assume earlier results still hold), do the "
    "follow-up, and report concisely. Schedule another wakeup only if there is genuinely more to "
    "wait for."
)

# ``30s`` / ``5m`` / ``2h`` / ``1d`` / ``1h30m``; a bare number is seconds.
_DELAY_TOKEN_RE = re.compile(r"^(?=\d)(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$", re.IGNORECASE)


def parse_delay(text: str) -> Optional[int]:
    """``"5m"`` → 300, ``"90"`` → 90, ``"1h30m"`` → 5400; None when unparseable."""
    raw = (text or "").strip()
    if not raw:
        return None
    if raw.isdigit():
        return int(raw)
    m = _DELAY_TOKEN_RE.match(raw)
    if not m:
        return None
    d, h, mi, s = (int(x) if x else 0 for x in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + s


def parse_when(text: str, *, now: Optional[float] = None) -> Optional[float]:
    """ISO-8601 → epoch seconds. A trailing ``Z`` or explicit offset is absolute; a naive
    timestamp is the host's local time. None when unparseable."""
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00") if raw.endswith("Z") else raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()  # naive == host local
    return parsed.timestamp()


def format_delay(seconds: float) -> str:
    """Whole-unit countdown label: ``in 5m``, ``in 2h``, ``in 1d``."""
    delta = max(0, int(round(seconds)))
    if delta < 60:
        return f"{max(1, delta)}s"
    if delta < 3600:
        return f"{round(delta / 60)}m"
    if delta < 86400:
        return f"{round(delta / 3600)}h"
    return f"{round(delta / 86400)}d"


def resolve_due_at(*, delay: Optional[str] = None, when: Optional[str] = None,
                   now: Optional[float] = None) -> tuple[Optional[float], Optional[str], str]:
    """``(due_at, error, clamp_note)`` from exactly one of ``delay`` / ``when``.

    A delay below the floor is raised to it and a horizon past the ceiling pulled back (with a
    note the tool relays to the model); an absolute time at or before now is an error, never
    silently clamped — the model asked for a specific instant that has already passed.
    """
    now = time.time() if now is None else now
    if bool(delay) == bool(when):
        return None, "give exactly one of `delay` or `when`", ""
    if delay:
        seconds = parse_delay(delay)
        if seconds is None:
            return None, f"unparseable delay {delay!r} (use 30s, 5m, 2h, 1d or a bare number of seconds)", ""
        note = ""
        if seconds < MIN_DELAY_SECONDS:
            note = f"delay raised to the {MIN_DELAY_SECONDS}s minimum"
            seconds = MIN_DELAY_SECONDS
        elif seconds > MAX_DELAY_SECONDS:
            note = "delay pulled back to the 7-day maximum — use a cron job for longer horizons"
            seconds = MAX_DELAY_SECONDS
        return now + seconds, None, note
    due = parse_when(when or "")
    if due is None:
        return None, f"unparseable ISO-8601 time {when!r} (e.g. 2026-09-13T14:30:00Z)", ""
    if due <= now:
        return None, "`when` is not in the future", ""
    if due - now > MAX_DELAY_SECONDS:
        return now + MAX_DELAY_SECONDS, None, "time pulled back to the 7-day maximum — use a cron job for longer horizons"
    return due, None, ""


@dataclass
class Wakeup:
    """One pending self-wakeup."""

    id: str
    prompt: str
    due_at: float
    created_at: float
    reason: str = ""

    def is_due(self, now: Optional[float] = None) -> bool:
        return (time.time() if now is None else now) >= self.due_at

    def render_prompt(self, now: Optional[float] = None) -> str:
        now = time.time() if now is None else now
        reason = f" — {self.reason}" if self.reason else ""
        return WAKEUP_PROMPT_TEMPLATE.format(
            wakeup_id=self.id, ago=format_delay(now - self.created_at), reason=reason, prompt=self.prompt)

    def summary(self, now: Optional[float] = None) -> Dict[str, Any]:
        now = time.time() if now is None else now
        due_iso = datetime.fromtimestamp(self.due_at, tz=timezone.utc).isoformat(timespec="seconds")
        return {
            "id": self.id, "due_at": due_iso, "due_in": format_delay(self.due_at - now),
            "reason": self.reason or None, "prompt": self.prompt,
        }


@dataclass
class WakeupList:
    """Serializable per-session wakeup set."""

    pending: List[Wakeup] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps({"pending": [asdict(w) for w in self.pending]}, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "WakeupList":
        data = json.loads(raw) or {}
        names = {f.name for f in fields(Wakeup)}
        out = []
        for item in data.get("pending") or []:
            if isinstance(item, dict) and item.get("id") and item.get("prompt"):
                out.append(Wakeup(**{k: v for k, v in item.items() if k in names}))
        return cls(pending=out)


def _get_session_db() -> Optional[Any]:
    """Persistence goes through the goals module's per-HERMES_HOME cached SessionDB (one shared connection)."""
    try:
        from hermes_cli.goals import _get_session_db as _goals_db
        return _goals_db()
    except Exception as exc:  # pragma: no cover
        logger.debug("WakeupManager: SessionDB bootstrap failed (%s)", exc)
        return None


def _meta_key(session_id: str) -> str:
    return f"{_META_PREFIX}{session_id}"


def load_wakeups(session_id: str) -> WakeupList:
    db = _get_session_db() if session_id else None
    if db is None:
        return WakeupList()
    try:
        raw = db.get_meta(_meta_key(session_id))
        return WakeupList.from_json(raw) if raw else WakeupList()
    except Exception as exc:
        logger.warning("WakeupManager: could not load wakeups for %s: %s", session_id, exc)
        return WakeupList()


def save_wakeups(session_id: str, wakeups: WakeupList) -> None:
    if not session_id:
        return
    db = _get_session_db()
    if db is None:
        from hermes_cli.goals import _warn_dropped_write
        _warn_dropped_write("WakeupManager", "wakeup", session_id)
        return
    try:
        db.set_meta(_meta_key(session_id), wakeups.to_json())
    except Exception as exc:
        logger.debug("WakeupManager: set_meta failed: %s", exc)


def list_sessions_with_wakeups() -> List[str]:
    """Session ids holding at least one pending wakeup (gateway idle watcher scan)."""
    db = _get_session_db()
    if db is None:
        return []
    out: List[str] = []
    try:
        rows = db.list_meta_prefix(_META_PREFIX)
    except Exception as exc:
        logger.debug("WakeupManager: list_meta_prefix failed: %s", exc)
        return []
    for key, raw in rows:
        sid = key[len(_META_PREFIX):]
        try:
            if sid and raw and WakeupList.from_json(raw).pending:
                out.append(sid)
        except Exception:
            continue
    return out


class WakeupManager:
    """Per-session pending wakeups + claim/abandon; the surface CLI, TUI and gateway talk to.

    Drivers poll :meth:`due_prompt` while the session is idle; a non-None return is the user-role
    message to inject. The claim is persisted BEFORE dispatch (a racing poller can't double-fire)
    and :meth:`abandon_fire` re-arms it when no turn started.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._last_claim: Optional[Wakeup] = None

    def load(self) -> WakeupList:
        return load_wakeups(self.session_id)

    def has_pending(self) -> bool:
        return bool(self.load().pending)

    def schedule(self, prompt: str, *, delay: Optional[str] = None, when: Optional[str] = None,
                 reason: str = "", now: Optional[float] = None) -> tuple[Optional[Wakeup], Optional[str], str]:
        """``(wakeup, error, clamp_note)``; ``error`` set on bad input or a full session."""
        prompt = (prompt or "").strip()
        if not prompt:
            return None, "prompt is empty", ""
        now = time.time() if now is None else now
        due_at, err, note = resolve_due_at(delay=delay, when=when, now=now)
        if err or due_at is None:
            return None, err or "could not resolve a due time", ""
        wakeups = self.load()
        if len(wakeups.pending) >= MAX_PER_SESSION:
            return None, (f"this session already holds the maximum of {MAX_PER_SESSION} pending wakeups; "
                          "cancel one first"), ""
        wakeup = Wakeup(id=f"wk_{uuid.uuid4().hex[:8]}", prompt=prompt, due_at=due_at, created_at=now,
                        reason=(reason or "").strip())
        wakeups.pending.append(wakeup)
        wakeups.pending.sort(key=lambda w: w.due_at)
        save_wakeups(self.session_id, wakeups)
        return wakeup, None, note

    def cancel(self, wakeup_id: str) -> bool:
        """Remove one pending wakeup; False when the id is unknown (already fired or cancelled)."""
        wakeups = self.load()
        before = len(wakeups.pending)
        wakeups.pending = [w for w in wakeups.pending if w.id != wakeup_id]
        if len(wakeups.pending) == before:
            return False
        save_wakeups(self.session_id, wakeups)
        return True

    def clear(self) -> int:
        wakeups = self.load()
        count = len(wakeups.pending)
        if count:
            save_wakeups(self.session_id, WakeupList())
        return count

    def status_line(self) -> str:
        pending = self.load().pending
        if not pending:
            return "No scheduled wakeups."
        now = time.time()
        lines = [f"⏰ {len(pending)} scheduled wakeup{'s' if len(pending) != 1 else ''}:"]
        for w in pending:
            label = w.reason or w.prompt
            lines.append(f"  {w.id} in {format_delay(w.due_at - now)} — {label}")
        return "\n".join(lines)

    def due_prompt(self, now: Optional[float] = None) -> Optional[str]:
        """Claim the earliest due wakeup and return its injection prompt, else None.

        The claim is persisted immediately (before the turn runs) so overlapping polls or a long
        turn can never double-fire the same wakeup; several overdue wakeups fire one per idle poll.
        """
        now = time.time() if now is None else now
        wakeups = self.load()
        due = next((w for w in wakeups.pending if w.is_due(now)), None)
        if due is None:
            return None
        wakeups.pending = [w for w in wakeups.pending if w.id != due.id]
        save_wakeups(self.session_id, wakeups)
        self._last_claim = due
        return due.render_prompt(now)

    def abandon_fire(self) -> bool:
        """Re-arm the wakeup claimed by the last :meth:`due_prompt` whose turn never started, so it
        fires on the next idle poll instead of vanishing. Mirrors ``HeartbeatManager.abandon_fire``."""
        claim = self._last_claim
        if claim is None:
            return False
        self._last_claim = None
        wakeups = self.load()
        if any(w.id == claim.id for w in wakeups.pending):
            return False
        wakeups.pending.append(claim)
        wakeups.pending.sort(key=lambda w: w.due_at)
        save_wakeups(self.session_id, wakeups)
        return True


def migrate_wakeups_to_session(old_session_id: str, new_session_id: str) -> bool:
    """Carry pending wakeups across a compression session rotation (copy to child, empty parent,
    never raise). Same shape as ``heartbeat.migrate_heartbeat_to_session``."""
    if not old_session_id or not new_session_id or old_session_id == new_session_id:
        return False
    try:
        old = load_wakeups(old_session_id)
        if not old.pending:
            return False
        new = load_wakeups(new_session_id)
        known = {w.id for w in new.pending}
        new.pending.extend(w for w in old.pending if w.id not in known)
        new.pending.sort(key=lambda w: w.due_at)
        save_wakeups(new_session_id, new)
        save_wakeups(old_session_id, WakeupList())
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("WakeupManager: migration failed: %s", exc)
        return False


__all__ = [
    "Wakeup", "WakeupList", "WakeupManager", "parse_delay", "parse_when", "format_delay", "resolve_due_at",
    "load_wakeups", "save_wakeups", "list_sessions_with_wakeups", "migrate_wakeups_to_session",
    "WAKEUP_PROMPT_TEMPLATE", "MIN_DELAY_SECONDS", "MAX_DELAY_SECONDS", "MAX_PER_SESSION",
]
