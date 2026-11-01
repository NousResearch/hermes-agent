"""Is the agent making people wait? Busy-chat arrivals and the day's peak of chats running at once.

``hermes.gateway.busy_arrival.count``: one row per human message that arrives while its chat's turn
is still running (the adapter's busy path), recorded once its fate is known: it reached the agent
(its own turn started, or it was steered into the running one), it was folded into another busy
message, or it never ran. Commands, internal events, plain-text approval answers and plugin-consumed
input are not counted. The arrival rides the event object itself (``_busy_arrival``), so the merge,
admission and settle seams need no lookup; the durable FIFO's admission id links it to the turn that
later runs the restored copy.

``gateway_peak_busy_chats_bucket`` (on ``hermes.engagement.day.count``): the most distinct chats
(session keys) that had a turn running at the same moment during the UTC day, per owning profile.
Tracked in memory from the runner's running-turn set at every turn boundary and folded into the
profile's engagement day state, which reports it when the day closes. No chat id leaves the process.

This module is imported by the contract at import time: stdlib only at module level.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

BUSY_ARRIVAL_METRIC = "hermes.gateway.busy_arrival.count"
BUSY_ARRIVAL_DISPOSITIONS = frozenset({"queued", "steered", "interrupted", "merged", "dropped", "other"})
BUSY_WAIT_BUCKETS = frozenset({"lt_5s", "5s_to_30s", "30s_to_2m", "2m_to_10m", "gte_10m", "never"})
BUSY_PEAK_FIELD = "gateway_peak_busy_chats_bucket"
_WAIT_THRESHOLDS = ((5.0, "lt_5s"), (30.0, "5s_to_30s"), (120.0, "30s_to_2m"), (600.0, "2m_to_10m"))


def wait_bucket(seconds: float | None) -> str:
    if seconds is None:
        return "never"
    seconds = max(0.0, seconds)
    return next((label for limit, label in _WAIT_THRESHOLDS if seconds < limit), "gte_10m")


def busy_arrival_fields(*, platform: Any, disposition: str, seconds: float | None) -> dict[str, str]:
    from .shared_metrics_gateway import _platform

    return {
        "disposition": disposition if disposition in BUSY_ARRIVAL_DISPOSITIONS else "other",
        "platform": _platform(platform),
        "wait_bucket": wait_bucket(seconds),
    }


# ---- busy arrivals ----

class _Arrival:
    __slots__ = ("admission_id", "admitting", "arrived", "disposition", "done", "home", "merged", "parked", "platform")

    def __init__(self, platform: Any, home: str | None) -> None:
        self.arrived = time.monotonic()
        self.platform, self.home = platform, home
        self.disposition = "queued"  # admitted to the FIFO unless the busy resolution says otherwise
        self.admitting = False  # reached the runner's admission: from here on, never running = dropped
        self.admission_id: str | None = None
        self.merged: list[_Arrival] = []
        self.done = False
        self.parked = False  # its row stays queued behind a paused FIFO: not dropped, owed later


# FIFO admission id -> arrival whose row is still queued. Touched only on the gateway loop.
_queued: OrderedDict[str, _Arrival] = OrderedDict()
_QUEUED_MAX = 4096


def _tag(event: Any) -> _Arrival | None:
    tag = getattr(event, "_busy_arrival", None)
    return tag if isinstance(tag, _Arrival) and not tag.done else None


def stamp_busy_arrival(adapter: Any, event: Any) -> None:
    """A message reached the adapter while its chat is busy (``_handle_message_while_active`` hands
    it to the shared runtime's busy dispatch, re-entered for the same event when a burst ends)."""
    if getattr(event, "_busy_arrival", None) is not None or getattr(event, "internal", False) or event.get_command():
        return
    try:
        from .shared_metrics_gateway import _chat_home, _source_platform

        chat_id = getattr(event.source, "chat_id", None)
        event._busy_arrival = _Arrival(_source_platform(adapter, chat_id), _chat_home(adapter, chat_id))
    except Exception:
        logger.debug("Busy arrival not stamped", exc_info=True)


def note_busy_merged(head: Any, event: Any) -> None:
    """``event`` was folded into the held burst ``head``: its fate is the head's."""
    tag, head_tag = _tag(event), _tag(head)
    if tag is not None and head_tag is not None:
        tag.done = True
        head_tag.merged.append(tag)


def note_busy_outcome(event: Any, outcome: Any) -> None:
    """The busy resolution for a follow-up (``_BusySteerOutcome``): steered into the running turn
    (steer or redirect) reaches the agent now; a hard interrupt ends the turn so it runs next."""
    tag = _tag(event)
    if tag is None:
        return
    _bind_home(tag)
    if outcome.steered or outcome.redirected:
        _resolve(tag, "steered", time.monotonic())
    elif outcome.effective_mode == "interrupt":
        tag.disposition = "interrupted"


def note_busy_admitting(event: Any) -> None:
    """The runner is admitting the message to the routed profile's FIFO (in that profile's scope)."""
    tag = _tag(event)
    if tag is not None:
        tag.admitting = True
        _bind_home(tag)


def _bind_home(tag: _Arrival) -> None:
    from hermes_constants import get_hermes_home

    tag.home = str(get_hermes_home())


def note_busy_admitted(event: Any, admission_id: str, status: str) -> None:
    """The row is committed; ``status`` as read right after. A drain that already claimed it ran it."""
    tag = _tag(event)
    if tag is None:
        return
    if status == "queued":
        tag.admission_id = str(admission_id)
        _queued[tag.admission_id] = tag
        while len(_queued) > _QUEUED_MAX:
            _queued.popitem(last=False)[1].done = True
    elif status in ("started", "unknown", "terminal"):  # ``unknown``: it started, then its turn was lost
        _resolve(tag, tag.disposition, time.monotonic())


def note_busy_parked(event: Any) -> None:
    """The handler is answering with a pause notice (``runtime_draining``, ``session_busy``,
    ``unknown_execution``) while the message's row is still queued: it is saved, not dropped. It
    counts only if this process later runs that row (``busy_turn_started``); a restart owes it to
    the next process, which has no record of the arrival."""
    tag = _tag(event)
    if tag is not None:
        tag.parked = True


def busy_turn_started(admission_id: Any) -> None:
    """The FIFO claimed this row and is running it: a queued busy arrival reached the agent."""
    tag = _queued.pop(str(admission_id), None)
    if tag is not None and not tag.done:
        _resolve(tag, tag.disposition, time.monotonic())


def settle_busy_arrival(event: Any, *, discarded: bool = False) -> None:
    """The busy handler for ``event`` returned (or its held burst was dropped by /stop, /new):
    an arrival that reached admission and never ran was dropped. One stopped by a restart is not
    counted: its committed row outlives this process. A parked one stays owed (``note_busy_parked``)."""
    tag = _tag(event)
    if tag is None or (tag.parked and not discarded):
        return
    if tag.admission_id is not None:
        _queued.pop(tag.admission_id, None)
    if not discarded and _cancelling():
        tag.done = True
    elif discarded or tag.admitting:
        _resolve(tag, "dropped", None)
    else:
        tag.done = True  # answered before admission (approval reply, plugin consume, refused sender)


def _cancelling() -> bool:
    try:
        task = asyncio.current_task()
    except RuntimeError:  # no running loop
        return False
    return task is not None and task.cancelling() > 0


def _resolve(tag: _Arrival, disposition: str, reached: float | None) -> None:
    rows = [(tag, disposition)] + [(merged, "merged") for merged in tag.merged]
    for arrival, kind in rows:
        arrival.done = True
        seconds = None if reached is None else reached - arrival.arrived
        from .shared_metrics_gateway import _in_home, _submit

        _submit(_in_home, tag.home, _emit_busy_arrival, arrival.platform, kind, seconds)


def _emit_busy_arrival(platform: Any, disposition: str, seconds: float | None) -> None:
    from .shared_metrics_events import _emit

    _emit(BUSY_ARRIVAL_METRIC, busy_arrival_fields, platform=platform, disposition=disposition, seconds=seconds)


# ---- peak of chats with a running turn, per owning profile and UTC day ----

_peak_lock = threading.Lock()
_running: dict[str, str] = {}  # session key -> owning profile home (resolved)
_peaks: dict[str, tuple[str, int]] = {}  # home -> (UTC day, most chats running at once that day)
_resolved: dict[str, str] = {}
_now: Callable[[], float] = time.time  # test / live-probe seam


def _day(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=UTC).date().isoformat()


def _home_key(home: Any) -> str:
    raw = str(home)
    if raw not in _resolved:
        try:
            _resolved[raw] = str(Path(raw).resolve())
        except OSError:
            _resolved[raw] = raw
    return _resolved[raw]


def note_running_chat_owner(runner: Any, session_key: str, source: Any) -> None:
    """``session_key`` is being claimed outside its profile's scope (the startup auto-resume loop):
    count it toward the profile that owns the session, resolved the way its turn will resolve it."""
    try:
        home = _home_key(runner._resolve_profile_home_for_source(source))
        with _peak_lock:
            _running[str(session_key)] = home
    except Exception:
        logger.debug("Running-chat owner not resolved", exc_info=True)


def observe_running_chats(session_keys: Any) -> None:
    """The runner's running-turn set at a turn boundary (claim or release). A newly running chat
    belongs to the profile whose scope claimed it."""
    try:
        keys = {str(key) for key in session_keys}
        home = None
        with _peak_lock:
            for key in [k for k in _running if k not in keys]:
                del _running[key]
            for key in keys - set(_running):
                if home is None:
                    from hermes_constants import get_hermes_home

                    home = _home_key(get_hermes_home())
                _running[key] = home
            counts: dict[str, int] = {}
            for owner in _running.values():
                counts[owner] = counts.get(owner, 0) + 1
            today = _day(_now())
            for owner, count in counts.items():
                day, peak = _peaks.get(owner, (today, 0))
                _peaks[owner] = (today, count if day != today else max(peak, count))
    except Exception:
        logger.debug("Running-chat peak not updated", exc_info=True)


def day_peak(home: Any, day: str) -> int:
    """Most chats of ``home``'s profile running at once on UTC ``day``, as seen by this process.
    A later stored day means ``day`` closed with what was folded before; chats still running when
    the day turned count toward the new one."""
    owner = _home_key(home)
    with _peak_lock:
        running = sum(1 for h in _running.values() if h == owner)
        stored_day, peak = _peaks.get(owner, (day, 0))
    if stored_day == day:
        return max(peak, running)
    return running if stored_day < day else 0
