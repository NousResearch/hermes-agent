"""Per-chat mute store for the gateway (the ``/mute`` toggle).

"Be quiet" is a *state command handled by the harness*, not conversational
input the model gets to argue with. While a chat is muted the gateway drops a
person's ordinary messages there deterministically — no agent turn, no tokens,
no reply. Slash commands always pierce the mute so a second ``/mute`` (and
everything else) keeps working.

A message is judged by when it ARRIVED (``MessageEvent._received_at``), not by
when it reaches the gate: a text batch flushed after the second ``/mute`` still
drops a message sent while muted, and one sent just before ``/mute`` still runs.
A turn the gateway already accepted (queued, or produced by a slash command) is
never re-judged. The runner owns those exemptions (``_mute_drops``).

State lives in ``{process HERMES_HOME}/.chat_mutes.json`` so a mute survives
gateway restarts. Keys are the runner's ``(bot, chat)`` key (the voice-mode key:
``[<bot profile>:]<platform>:<chat_id>``), so two bots in one chat mute
independently. Each entry is one interval, ``{"muted_at": <epoch>,
"expires_at": <epoch|null>}``: null is indefinite, and an unmute closes it at
the unmute's arrival. A closed interval is kept briefly for messages still in a
text batch, then pruned on the next write.

Reads never raise: an unreadable or corrupt store reads as "nothing muted"
(fail-open — a broken marker file must never silence every chat).
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_process_hermes_home
from utils import atomic_json_write
from utils.files import read_json_or_empty

logger = logging.getLogger(__name__)

_MUTE_STORE_FILENAME = ".chat_mutes.json"
# How long a closed interval still judges late-flushed messages (text batches wait seconds).
_CLOSED_RETENTION_S = 600.0

# /mute duration argument: bare integer = minutes; suffixed forms m/h/d.
_DURATION_RE = re.compile(r"^(\d+)\s*(m|min|mins|minutes?|h|hr|hrs|hours?|d|days?)?$", re.IGNORECASE)

_UNIT_SECONDS = {
    "m": 60,
    "h": 3600,
    "d": 86400,
}


def _store_path() -> Path:
    return get_process_hermes_home() / _MUTE_STORE_FILENAME


def _load_store() -> dict[str, dict[str, Any]]:
    return {k: v for k, v in read_json_or_empty(_store_path()).items() if isinstance(v, dict)}


def _bound(entry: dict[str, Any], name: str, default: float) -> float:
    try:
        value = entry.get(name)
        return default if value is None else float(value)
    except (TypeError, ValueError):
        # An unparsable expiry degrades to indefinite: the user asked for silence, and only a
        # second /mute (or a valid expiry) ends it.
        return default


def _muted_at(entry: Optional[dict[str, Any]], at: float) -> bool:
    if entry is None:
        return False
    return _bound(entry, "muted_at", 0.0) <= at < _bound(entry, "expires_at", float("inf"))


def is_muted(key: str, at: Optional[float] = None) -> bool:
    """True when the chat under *key* was muted at *at* (default: now)."""
    return _muted_at(_load_store().get(key), time.time() if at is None else at)


def event_muted(event: Any, key: str) -> bool:
    """True when *event* is a person's ordinary message that arrived while *key* was muted."""
    if event.internal or event._mute_exempt or event.get_command():
        return False
    if not is_muted(key, at=event._received_at):
        return False
    logger.debug("Dropping message for muted chat %s", key)
    return True


def _write(key: str, entry: Optional[dict[str, Any]], now: float) -> None:
    store = {k: v for k, v in _load_store().items()
             if _bound(v, "expires_at", float("inf")) > now - _CLOSED_RETENTION_S}
    if entry is not None:
        store[key] = entry
    atomic_json_write(_store_path(), store)


def set_chat_mute(key: str, duration_seconds: Optional[float] = None,
                  at: Optional[float] = None) -> dict[str, Any]:
    """Mute *key* from *at* (the /mute's arrival; default now), indefinitely or for
    ``duration_seconds``. Replaces any earlier interval. Returns the entry."""
    start = time.time() if at is None else at
    entry = {"muted_at": start,
             "expires_at": (start + duration_seconds) if duration_seconds is not None else None}
    _write(key, entry, start)
    return entry


def clear_chat_mute(key: str, at: Optional[float] = None) -> bool:
    """Close *key*'s live mute at *at* (default now). Returns False when it was not muted."""
    end = time.time() if at is None else at
    entry = _load_store().get(key)
    if not _muted_at(entry, end):
        return False
    _write(key, {**entry, "expires_at": end}, end)
    return True


def parse_mute_duration(arg: str) -> tuple[bool, Optional[float]]:
    """Parse a ``/mute`` duration argument.

    Returns ``(ok, seconds)``. ``seconds`` is None for an indefinite mute
    (empty arg or "on"). ``ok`` False means the argument was unrecognized.
    """
    arg = (arg or "").strip().lower()
    if arg in {"", "on"}:
        return True, None
    m = _DURATION_RE.match(arg)
    if not m:
        return False, None
    value = int(m.group(1))
    if value <= 0:
        return False, None
    unit = (m.group(2) or "m")[0].lower()
    return True, float(value * _UNIT_SECONDS[unit])


def format_remaining(entry: dict[str, Any], now: Optional[float] = None) -> str:
    """Human-readable remaining time for a mute entry ('' when indefinite)."""
    expires_at = entry.get("expires_at")
    if expires_at is None:
        return ""
    now = time.time() if now is None else now
    try:
        # Ceil so a freshly set "2h" mute reads "2h 0m", not "1h 59m".
        remaining = max(0, -int(-(float(expires_at) - now) // 1))
    except (TypeError, ValueError):
        return ""
    if remaining >= 86400:
        return f"{remaining // 86400}d {(remaining % 86400) // 3600}h"
    if remaining >= 3600:
        return f"{remaining // 3600}h {(remaining % 3600) // 60}m"
    return f"{max(1, remaining // 60)}m"
