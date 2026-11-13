"""Telegram live-location refreshes: one shared live location is one conversation input.

A live location is a single message whose position Telegram keeps editing (an
``edited_message`` every few seconds to a minute, for up to the share's ``live_period``).
PTB's ``MessageHandler`` dispatches those edits like new messages, so every GPS tick used to
start a full agent turn and an agent reply. The first pin of a share is the user's input; a
refresh is input again only once the user has actually moved, or after a long quiet spell,
measured on Telegram's own ``edit_date`` (handler time drifts behind the per-chat queue).
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any

# A refresh becomes a turn again after this much movement from the last position that did...
LIVE_REFRESH_MIN_METERS = 75.0
# ...or once this long passed since that position (a parked user still gets a rare update).
LIVE_REFRESH_MIN_SECONDS = 900.0
_MAX_TRACKED_SHARES = 256  # memory bound only: one entry per live message still being edited


def _meters_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6_371_000 * math.asin(math.sqrt(min(1.0, h)))


def _stamp(value: Any) -> float | None:
    return value.timestamp() if hasattr(value, "timestamp") else None


def is_redundant_live_refresh(adapter: Any, msg: Any, location: Any, lat: float, lon: float) -> bool:
    """True for a live-location edit that should not start a turn (same place, too soon).

    Only a live location can be edited, so a new message (no ``edit_date``) always passes; a
    live one becomes its share's reference point. An edit without ``live_period`` is the share
    stopping and carries nothing new. A refresh of a share this process never saw start (it
    began before a restart, and its pin already had its turn) becomes the reference silently.
    """
    shares: OrderedDict = adapter.__dict__.setdefault("_telegram_live_location_shares", OrderedDict())
    chat_id = getattr(getattr(msg, "chat", None), "id", None) or getattr(msg, "chat_id", None)
    key = (str(chat_id), str(getattr(msg, "message_id", "")))
    edited_at = _stamp(getattr(msg, "edit_date", None))
    live = getattr(location, "live_period", None) is not None
    if edited_at is not None and not live:
        shares.pop(key, None)
        return True
    if not live:
        return False
    now = edited_at if edited_at is not None else _stamp(getattr(msg, "date", None))
    previous = shares.get(key)
    redundant = edited_at is not None and (
        previous is None
        or (now is not None and previous[0] is not None and now - previous[0] < LIVE_REFRESH_MIN_SECONDS
            and _meters_between(previous[1], previous[2], lat, lon) < LIVE_REFRESH_MIN_METERS))
    if not redundant or previous is None:
        shares[key] = (now, lat, lon)
        shares.move_to_end(key)
        while len(shares) > _MAX_TRACKED_SHARES:
            shares.popitem(last=False)
    return redundant
