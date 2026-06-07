"""Compact per-message timestamp context for agent turns.

Timestamps are injected at API-call time, not persisted into session history and
not added to the system prompt.  This preserves prompt-cache stability while
letting the model understand the real temporal spacing of replayed/history
messages and the current turn.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Optional

_TIMESTAMP_MARKER_RE = re.compile(r"\[sent: \d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:\d{2})\]\s*")
_TIMESTAMP_PREFIX_RE = re.compile(rf"^{_TIMESTAMP_MARKER_RE.pattern}")


def _coerce_datetime(value: Any) -> Optional[datetime]:
    """Convert supported timestamp metadata into a datetime, if possible."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        try:
            from hermes_time import get_timezone

            tz = get_timezone()
            if tz is not None:
                return datetime.fromtimestamp(float(value), tz)
            return datetime.fromtimestamp(float(value)).astimezone()
        except Exception:
            return None
    if isinstance(value, str) and value.strip():
        raw = value.strip()
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _format_sent_timestamp(dt: datetime) -> str:
    try:
        from hermes_time import get_timezone

        tz = get_timezone()
    except Exception:
        tz = None
    if tz is not None:
        dt = dt.astimezone(tz)
    elif dt.tzinfo is None:
        dt = dt.astimezone()
    return dt.isoformat(timespec="minutes")


def sent_timestamp_prefix(timestamp: Any = None) -> str:
    """Return the compact prefix used for LLM-visible message time context."""
    dt = _coerce_datetime(timestamp)
    if dt is None:
        return ""
    return f"[sent: {_format_sent_timestamp(dt)}]"


def add_sent_timestamp_prefix(content: Any, timestamp: Any = None) -> Any:
    """Prefix string content with its send time, without double-prefixing."""
    if not isinstance(content, str) or not content:
        return content
    if _TIMESTAMP_PREFIX_RE.match(content):
        return content
    prefix = sent_timestamp_prefix(timestamp)
    if not prefix:
        return content
    return f"{prefix}\n{content}"


def strip_sent_timestamp_prefix(content: Any) -> Any:
    """Remove an internal send-time prefix if the model echoes it visibly."""
    if not isinstance(content, str) or not content:
        return content
    return _TIMESTAMP_MARKER_RE.sub("", content)


# Backward-compatible name used by the first implementation on this branch.
def format_live_time_context(now: Optional[datetime] = None) -> str:
    return sent_timestamp_prefix(now)


__all__ = [
    "add_sent_timestamp_prefix",
    "format_live_time_context",
    "sent_timestamp_prefix",
    "strip_sent_timestamp_prefix",
]
