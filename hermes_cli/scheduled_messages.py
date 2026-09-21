"""Scheduled composer messages — one future user turn in ONE session, fired once.

Same durable base as ``/goal`` (``hermes_cli/goals.py``), ``/loop`` (``loops.py``) and
``/heartbeat`` (``heartbeat.py``): session-scoped state in SessionDB ``state_meta``, keyed by the
session key, fired by the session-owner process's notification poller
(``tui_gateway.session_notifications._notification_poller_loop``). Two deliberate differences:

- the payload is a **plain user message** — it re-enters through the ordinary turn path, so role
  alternation, prompt caching, history and the queueing rules are whatever a typed message gets;
- it is **one-shot and absolute**: a due time, not a cadence, and firing it is irreversible.

At-most-once: a fire is claimed (``pending`` -> ``fired``, persisted) BEFORE the turn is
dispatched, so a crash in the dispatch window loses the message instead of running it twice. A
dispatch that demonstrably never started a turn rewinds the claim (``abandon_fire``), mirroring
``heartbeat.abandon_fire``: the item stays due rather than being silently consumed.

Missed while nothing was running (app closed, machine asleep, backend restarted): the item stays
``pending`` past its due time and fires on the next poll after the session is live again. Never
dropped, never fired twice.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# SessionDB state_meta key. Heartbeat/loop read theirs as ``heartbeat:{session_id}``; same shape,
# one JSON list per session (a composer holds a handful of scheduled drafts at most).
META_PREFIX = "scheduled_message:"

MAX_PENDING = 20        # the cap that stops a runaway UI loop from queueing an unbounded list
MAX_TEXT_CHARS = 8000   # matches the composer's own pasted-draft ceiling
_RETAINED_DONE = 20     # fired/cancelled receipts kept (dedup by id); oldest are pruned
_PAST_GRACE = 60.0      # an item this far past due is accepted: clock skew across surfaces

Home = Union[str, Path, None]

_LOCK = threading.RLock()


def _meta_key(session_id: str) -> str:
    return f"{META_PREFIX}{session_id}"


def _get_session_db(home: Home = None) -> Optional[Any]:
    """SessionDB handle for ``home``, or the ambient HERMES_HOME when it is None.

    Both paths go through the state registry: one shared, refcounted writer per ``state.db``. A
    bare ``SessionDB()`` here would be the second-writer corruption shape ``goals._acquire_session_db``
    documents. The default profile rides the goals cache (same handle ``/goal`` and ``/heartbeat``
    write through); a session pinned to another profile home gets that home's db.
    """
    try:
        if home is None:
            from hermes_cli.goals import _get_session_db as _goals_db

            return _goals_db()
        from hermes_state_registry import acquire

        return acquire(Path(home).resolve() / "state.db")
    except Exception as exc:  # pragma: no cover
        logger.debug("ScheduledMessages: SessionDB bootstrap failed (%s)", exc)
        return None


@contextmanager
def _claim_lock(home: Home = None):
    """Serialize read-modify-write of the list, across threads AND processes.

    Two backends can share a HERMES_HOME (Desktop plus a messaging gateway), and the
    pending -> fired flip is the whole at-most-once guarantee — a lost race there runs the user's
    message twice. In-process the RLock covers it; the file lock is the same one
    ``cron.bot_chat_delivery`` uses. If the lock file can't be created we degrade to in-process
    only rather than refusing to schedule at all.
    """
    with _LOCK, ExitStack() as stack:
        try:
            if home is None:
                from hermes_constants import get_hermes_home

                base = Path(get_hermes_home())
            else:
                base = Path(home)
            from hermes_cli.active_sessions import _FileLock

            stack.enter_context(_FileLock(base.resolve() / "scheduled_messages.lock"))
        except Exception as exc:  # pragma: no cover - unwritable home
            logger.debug("ScheduledMessages: file lock unavailable (%s)", exc)
        yield


@dataclass
class ScheduledMessage:
    """A pending one-shot user message. ``due_at`` is epoch seconds — the client renders it in
    local time, so no stored timezone can go stale across a DST or machine move."""

    id: str
    text: str
    due_at: float
    created_at: float = 0.0
    status: str = "pending"          # pending | fired | cancelled
    fired_at: float = 0.0

    def public(self, *, now: float) -> Dict[str, Any]:
        """Wire shape for the client: what the UI renders, and nothing else."""
        return {
            "id": self.id,
            "text": self.text,
            "display_text": self.text if len(self.text) <= 200 else self.text[:200] + "…",
            "due_at": self.due_at,
            "created_at": self.created_at,
            "status": self.status,
            "overdue": self.status == "pending" and self.due_at <= now,
        }

    def is_due(self, now: Optional[float] = None) -> bool:
        return self.status == "pending" and self.due_at <= (time.time() if now is None else now)


def _parse_record(raw: Any) -> Optional[ScheduledMessage]:
    if not isinstance(raw, dict):
        return None
    try:
        message = ScheduledMessage(
            id=str(raw["id"]),
            text=str(raw["text"]),
            due_at=float(raw["due_at"]),
            created_at=float(raw.get("created_at") or 0.0),
            status=str(raw.get("status") or "pending"),
            fired_at=float(raw.get("fired_at") or 0.0),
        )
    except (KeyError, TypeError, ValueError):
        return None
    return message if message.id and message.text else None


def _load(session_id: str, home: Home = None) -> List[ScheduledMessage]:
    db = _get_session_db(home) if session_id else None
    if db is None:
        return []
    try:
        raw = db.get_meta(_meta_key(session_id))
    except Exception as exc:
        logger.debug("ScheduledMessages: get_meta failed: %s", exc)
        return []
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (ValueError, TypeError) as exc:
        logger.warning("ScheduledMessages: unreadable list for %s: %s", session_id, exc)
        return []
    items = [_parse_record(entry) for entry in data] if isinstance(data, list) else []
    return [item for item in items if item is not None]


def _save(session_id: str, items: List[ScheduledMessage], home: Home = None) -> bool:
    if not session_id:
        return False
    db = _get_session_db(home)
    if db is None:
        from hermes_cli.goals import _warn_dropped_write

        _warn_dropped_write("ScheduledMessages", "scheduled message", session_id)
        return False
    try:
        db.set_meta(_meta_key(session_id),
                    json.dumps([asdict(item) for item in items], ensure_ascii=False))
    except Exception as exc:
        logger.warning("ScheduledMessages: set_meta failed for %s: %s", session_id, exc)
        return False
    return True


def _prune(items: List[ScheduledMessage]) -> List[ScheduledMessage]:
    """Keep every pending item; retain the newest ``_RETAINED_DONE`` receipts behind them."""
    pending = [item for item in items if item.status == "pending"]
    done = [item for item in items if item.status != "pending"]
    done.sort(key=lambda item: item.created_at, reverse=True)
    return pending + done[:_RETAINED_DONE]


def schedule_message(
    session_id: str, text: str, due_at: float, *, message_id: Optional[str] = None,
    now: Optional[float] = None, home: Home = None,
) -> ScheduledMessage:
    """Persist a pending message. Raises ``ValueError`` on a payload the UI should have caught."""
    body = (text or "").strip()
    if not body:
        raise ValueError("text is required")
    if len(body) > MAX_TEXT_CHARS:
        raise ValueError(f"text is longer than {MAX_TEXT_CHARS} characters")
    due = float(due_at)
    stamp = time.time() if now is None else now
    if due < stamp - _PAST_GRACE:
        raise ValueError("due time is in the past")
    with _claim_lock(home):
        items = _load(session_id, home)
        if sum(1 for item in items if item.status == "pending") >= MAX_PENDING:
            raise ValueError(f"too many pending scheduled messages (max {MAX_PENDING})")
        if message_id and any(item.id == message_id for item in items):
            raise ValueError("message id already exists")
        message = ScheduledMessage(id=message_id or uuid.uuid4().hex, text=body, due_at=due,
                                   created_at=stamp)
        _save(session_id, _prune([*items, message]), home)
    return message


def list_messages(session_id: str, *, now: Optional[float] = None, home: Home = None) -> List[Dict[str, Any]]:
    """Pending items, soonest due first, as wire dicts."""
    stamp = time.time() if now is None else now
    items = sorted((item for item in _load(session_id, home) if item.status == "pending"),
                   key=lambda item: item.due_at)
    return [item.public(now=stamp) for item in items]


def cancel_message(session_id: str, message_id: str, *, home: Home = None) -> bool:
    """Mark a still-pending item cancelled. False when it already fired or never existed."""
    with _claim_lock(home):
        items = _load(session_id, home)
        target = next((item for item in items if item.id == message_id and item.status == "pending"), None)
        if target is None:
            return False
        target.status = "cancelled"
        return _save(session_id, _prune(items), home)


def due_items(session_id: str, *, now: Optional[float] = None, home: Home = None) -> List[ScheduledMessage]:
    """Pending items whose due time has passed — including ones missed while nothing was running."""
    stamp = time.time() if now is None else now
    return sorted((item for item in _load(session_id, home) if item.is_due(stamp)),
                  key=lambda item: item.due_at)


def claim_due(
    session_id: str, message_id: str, *, now: Optional[float] = None, home: Home = None,
) -> Optional[ScheduledMessage]:
    """Flip one due item to ``fired`` and persist it BEFORE dispatch. None when it is not claimable
    (already fired, cancelled, or not yet due) — that None is what makes a repeat scan harmless."""
    stamp = time.time() if now is None else now
    with _claim_lock(home):
        items = _load(session_id, home)
        target = next((item for item in items if item.id == message_id), None)
        if target is None or not target.is_due(stamp):
            return None
        target.status, target.fired_at = "fired", stamp
        if not _save(session_id, _prune(items), home):
            return None
        return target


def abandon_fire(session_id: str, message_id: str, *, home: Home = None) -> bool:
    """Rewind a claim whose turn never started. False when a peer already moved it on."""
    with _claim_lock(home):
        items = _load(session_id, home)
        target = next((item for item in items if item.id == message_id), None)
        if target is None or target.status != "fired":
            return False
        target.status, target.fired_at = "pending", 0.0
        return _save(session_id, _prune(items), home)


def clear_session(session_id: str, *, home: Home = None) -> None:
    """Drop every item for a session (teardown/tests)."""
    with _claim_lock(home):
        _save(session_id, [], home)


def format_due_display(due_at: float, *, tz: Any = None) -> str:
    """Local-time rendering for status lines (`2026-09-16 21:30`)."""
    from datetime import datetime

    return datetime.fromtimestamp(float(due_at), tz=tz).strftime("%Y-%m-%d %H:%M")
