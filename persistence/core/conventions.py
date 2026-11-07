"""Cross-capability conventions: timestamps and the row-visibility vocabulary.

Timestamps: the sole timestamp type at the persistence boundary is a timezone-aware UTC ``datetime``. Naive datetimes are rejected at the boundary; backends store and compare whatever physical representation they like but always accept and return aware-UTC values. Ordering promises are always expressed over (timestamp, id) with the id as the deterministic backend-neutral tiebreaker.

Stored instants: a physical store with dynamic typing can hold a timestamp cell no instant decodes from (a TEXT cell, ``inf``/``nan``, or a garbage double salvaged from a damaged page). Reads degrade such a cell instead of failing the page: :func:`stored_instant` reads it as unset, and a required instant reads as :data:`UNTRUSTED_INSTANT`, with a WARNING naming the session so the corrupt row can be found. The trust window is the one ``hermes_cli/timefmt.py`` applies. Every backend decodes its stored row instants through this rule, including one whose public verbs can only write aware instants (the in-memory reference), so a cell that arrived by any other route reads the same everywhere.

Row visibility: transcript rows have three observable states, and reads either see only ACTIVE rows or, with include_inactive semantics, all three. REWOUND and COMPACTION_ARCHIVED are both inactive but are distinct states; the distinction is CONTRACT because compaction/search visibility rules depend on it.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone, UTC
from enum import Enum
from typing import Any, Optional, SupportsIndex

__all__ = ["UNTRUSTED_INSTANT", "RowVisibility", "ensure_utc", "stored_instant"]

logger = logging.getLogger(__name__)

# Epoch-seconds window a stored instant must fall in to be trusted: 1970 .. ~2103, inside 32-bit ``time_t`` so every platform decodes it.
_STORED_EPOCH_MIN = 0.0
_STORED_EPOCH_MAX = 4_200_000_000.0

#: What a REQUIRED instant (``started_at``, a recency key) reads as when its stored cell is untrusted: the window floor, the one instant CLI renderers already treat as unset.
UNTRUSTED_INSTANT = datetime.fromtimestamp(_STORED_EPOCH_MIN, tz=UTC)


class RowVisibility(Enum):
    """Observable visibility state of a transcript row."""

    ACTIVE = "active"
    REWOUND = "rewound"
    COMPACTION_ARCHIVED = "compaction_archived"


def ensure_utc(value: datetime) -> datetime:
    """Return ``value`` normalized to UTC; reject naive datetimes."""
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError("persistence timestamps must be timezone-aware; got a naive datetime")
    return value.astimezone(UTC)


class _StoredInstant(datetime):
    """An aware instant whose epoch keeps the stored float's full precision."""

    _stored_epoch: float

    def timestamp(self) -> float:
        return getattr(self, "_stored_epoch", super().timestamp())

    def __reduce_ex__(self, protocol: SupportsIndex) -> Any:
        return (_instant_from_epoch, (self.timestamp(),))


def _instant_from_epoch(epoch: float) -> datetime:
    instant = _StoredInstant.fromtimestamp(epoch, tz=UTC)
    instant._stored_epoch = 0.0 if epoch == 0.0 else epoch
    return instant


def stored_instant(value: Any, *, session_id: Optional[str] = None, field: str = "timestamp") -> Optional[datetime]:
    """A stored timestamp cell as an aware-UTC instant, or ``None`` when unset or untrusted.

    Numbers, numeric strings and aware datetimes are accepted; ``None``/``""`` mean unset and stay silent. Anything else, non-finite values and values outside the trust window read as ``None`` after a WARNING naming the session and field (so one bad row degrades one cell instead of the page).
    """
    if value is None or value == "":
        return None
    instant: Optional[datetime] = None
    try:
        epoch = (instant := ensure_utc(value)).timestamp() if isinstance(value, datetime) else float(value)
    except (TypeError, ValueError):
        epoch = math.nan
    if not (_STORED_EPOCH_MIN <= epoch <= _STORED_EPOCH_MAX):  # also False for nan
        logger.warning("Ignoring corrupt %s %r%s", field, value, f" on session {session_id}" if session_id else "")
        return None
    return instant or _instant_from_epoch(epoch)
