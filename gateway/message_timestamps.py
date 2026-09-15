"""Helpers for rendering gateway message timestamps exactly once.

Gateway messages need timestamps in the LLM context for temporal awareness, but
persisted message content should stay clean so replay does not accumulate
``[timestamp] [timestamp] ...`` prefixes across turns.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Optional, Tuple


# Leading timestamp prefix, either the current human format
# ``[Tue 2026-04-28 13:40:53 CEST]`` or the older ISO one
# ``[2026-04-13T17:02:06+0200]`` / ``[...+02:00]`` (human tried first).
_TIMESTAMP_PREFIX_RE = re.compile(
    r"^\[(?:"
    r"(?P<dow>[A-Z][a-z]{2}) "
    r"(?P<date>\d{4}-\d{2}-\d{2}) "
    r"(?P<time>\d{2}:\d{2}:\d{2})"
    r"(?: (?P<tz>[A-Za-z0-9_+\-/:]+))?"
    r"|(?P<iso>\d{4}-\d{2}-\d{2}T[^\]]+)"
    r")\]\s*"
)


def _localize(dt: datetime, tz) -> float:
    """Epoch for ``dt``; naive values take ``tz`` if given, else the local zone."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz) if tz is not None else dt.astimezone()
    return float(dt.timestamp())


def _parse_iso(text: str, tz=None) -> Optional[float]:
    """Parse an ISO-8601 string (incl. ``+0200`` offsets fromisoformat rejects)."""
    for parse in (datetime.fromisoformat, lambda t: datetime.strptime(t, "%Y-%m-%dT%H:%M:%S%z")):
        try:
            return _localize(parse(text), tz)
        except (TypeError, ValueError):
            continue
    return None


def _parse_timestamp_match(match: re.Match, tz=None) -> Optional[float]:
    if match.group("iso"):
        return _parse_iso(match.group("iso"), tz)
    try:
        dt = datetime.strptime(f"{match.group('date')} {match.group('time')}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return _localize(dt, tz)


def coerce_message_timestamp(ts_value: Any, tz=None) -> Optional[float]:
    """Epoch seconds from a number, datetime, ISO string, or the gateway's bracketed
    format; ``None`` when uninterpretable."""
    if isinstance(ts_value, (int, float)):
        return float(ts_value)
    if hasattr(ts_value, "timestamp"):
        try:
            return float(ts_value.timestamp())
        except Exception:
            return None
    text = ts_value.strip() if isinstance(ts_value, str) else ""
    if not text:
        return None
    match = _TIMESTAMP_PREFIX_RE.match(text)
    parsed = _parse_timestamp_match(match, tz=tz) if match is not None else None
    if parsed is not None:
        return parsed
    try:
        return float(text)
    except (TypeError, ValueError):
        return _parse_iso(text, tz)


def format_message_timestamp(ts_value: Any, tz=None) -> str:
    """Format a timestamp value as ``[Tue 2026-04-28 13:40:53 CEST]``."""
    epoch = coerce_message_timestamp(ts_value, tz=tz)
    if epoch is None:
        return ""
    dt = datetime.fromtimestamp(epoch, tz=tz) if tz is not None else datetime.fromtimestamp(epoch).astimezone()
    return f"[{dt.strftime('%a %Y-%m-%d %H:%M:%S %Z')}]"


def strip_leading_message_timestamps(content: str, tz=None) -> Tuple[str, Optional[float]]:
    """Strip leading gateway timestamp prefixes → ``(clean_content, embedded_epoch)``.
    With several prefixes the one closest to the text wins, preserving the platform-send
    time of legacy rows like ``[processing time] [platform time] [sender] message``."""
    if not isinstance(content, str) or not content:
        return content, None
    text, embedded_epoch = content, None
    while (match := _TIMESTAMP_PREFIX_RE.match(text)) is not None:
        parsed = _parse_timestamp_match(match, tz=tz)
        if parsed is not None:
            embedded_epoch = parsed
        text = text[match.end():]
    return text, embedded_epoch


def render_user_content_with_timestamp(content: str, ts_value: Any = None, tz=None) -> str:
    """Render a user message for LLM context with exactly one timestamp prefix; an
    existing prefix is stripped and its parsed time wins over ``ts_value``."""
    clean_content, embedded_epoch = strip_leading_message_timestamps(content, tz=tz)
    prefix = format_message_timestamp(ts_value if embedded_epoch is None else embedded_epoch, tz=tz)
    return f"{prefix} {clean_content}" if prefix and clean_content else (prefix or clean_content)
