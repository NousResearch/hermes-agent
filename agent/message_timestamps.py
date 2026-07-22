"""Render user-message timestamps in model context exactly once.

All Hermes surfaces can add temporal context while persisted message content stays
clean, so replay never accumulates ``[timestamp] [timestamp] ...`` prefixes.
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def message_timestamps_enabled(config: Optional[dict]) -> bool:
    """Resolve the canonical global opt-in with gateway compatibility."""
    if not isinstance(config, dict):
        return False

    configured = config.get("message_timestamps")
    if configured is not None:
        if isinstance(configured, dict):
            enabled = configured.get("enabled")
            if enabled is not None:
                return bool(enabled)
        else:
            return bool(configured)

    gateway = config.get("gateway")
    if not isinstance(gateway, dict):
        return False
    legacy = gateway.get("message_timestamps")
    return (
        bool(legacy.get("enabled", False))
        if isinstance(legacy, dict)
        else bool(legacy)
    )


def render_turn_with_message_timestamps(
    history: Optional[List[Dict[str, Any]]],
    user_message: Any,
    *,
    config: Optional[dict],
    current_timestamp: Any = None,
    tz=None,
) -> Tuple[List[Dict[str, Any]], Any, Optional[float]]:
    """Return model-only timestamp rendering without mutating caller data."""
    rendered_history = [dict(message) for message in (history or [])]
    if not message_timestamps_enabled(config):
        return rendered_history, user_message, None

    for message in rendered_history:
        if message.get("role") == "user":
            timestamp = message.get("timestamp")
            message["content"] = _render_user_message_content(
                message.get("content"),
                timestamp,
                tz=tz,
            )
            # ``api_content`` wins over ``content`` at the final API boundary.
            # Render the sidecar too, otherwise persisted prefetch/plugin bytes
            # silently erase the timestamp we just added to historical context.
            if isinstance(message.get("api_content"), str):
                message["api_content"] = render_user_content_with_timestamp(
                    message["api_content"], timestamp, tz=tz
                )

    effective_timestamp = coerce_message_timestamp(
        time.time() if current_timestamp is None else current_timestamp,
        tz=tz,
    )
    user_message = _render_user_message_content(
        user_message,
        effective_timestamp,
        tz=tz,
    )
    return rendered_history, user_message, effective_timestamp


def _render_user_message_content(content: Any, ts_value: Any, tz=None) -> Any:
    """Copy and timestamp plain text or OpenAI-style structured content."""
    if isinstance(content, str):
        return render_user_content_with_timestamp(content, ts_value, tz=tz)
    if not isinstance(content, list):
        return content

    parts = [dict(part) if isinstance(part, dict) else part for part in content]
    for part in parts:
        if isinstance(part, dict) and part.get("type") == "text":
            text = part.get("text")
            if isinstance(text, str):
                part["text"] = render_user_content_with_timestamp(
                    text, ts_value, tz=tz
                )
                return parts

    prefix = format_message_timestamp(ts_value, tz=tz)
    if prefix:
        parts.insert(0, {"type": "text", "text": prefix})
    return parts


# Current gateway format: [Tue 2026-04-28 13:40:53 CEST]
_HUMAN_TIMESTAMP_RE = re.compile(
    r"^\[(?P<dow>[A-Z][a-z]{2}) "
    r"(?P<date>\d{4}-\d{2}-\d{2}) "
    r"(?P<time>\d{2}:\d{2}:\d{2})"
    r"(?: (?P<tz>[A-Za-z0-9_+\-/:]+))?\]\s*"
)

# Older gateway format: [2026-04-13T17:02:06+0200] or [+02:00]
_ISO_TIMESTAMP_RE = re.compile(
    r"^\[(?P<iso>\d{4}-\d{2}-\d{2}T[^\]]+)\]\s*"
)


def coerce_message_timestamp(ts_value: Any, tz=None) -> Optional[float]:
    """Coerce a timestamp-like value to Unix epoch seconds.

    Accepts Unix epoch numbers, datetime objects, ISO strings, and the gateway's
    bracketed human-readable timestamp format. Returns ``None`` when the value
    cannot be interpreted.
    """
    if ts_value is None:
        return None

    if isinstance(ts_value, (int, float)):
        return float(ts_value)

    if hasattr(ts_value, "timestamp"):
        try:
            return float(ts_value.timestamp())
        except Exception:
            return None

    if isinstance(ts_value, str):
        text = ts_value.strip()
        if not text:
            return None
        parsed = _parse_timestamp_prefix(text, tz=tz)
        if parsed is not None:
            return parsed
        try:
            return float(text)
        except (TypeError, ValueError):
            pass
        try:
            dt = datetime.fromisoformat(text)
        except (TypeError, ValueError):
            try:
                dt = datetime.strptime(text, "%Y-%m-%dT%H:%M:%S%z")
            except (TypeError, ValueError):
                return None
        if dt.tzinfo is None:
            if tz is not None:
                dt = dt.replace(tzinfo=tz)
            else:
                dt = dt.astimezone()
        return float(dt.timestamp())

    return None


def format_message_timestamp(ts_value: Any, tz=None) -> str:
    """Format a timestamp value as ``[Tue 2026-04-28 13:40:53 CEST]``."""
    epoch = coerce_message_timestamp(ts_value, tz=tz)
    if epoch is None:
        return ""
    if tz is not None:
        dt = datetime.fromtimestamp(epoch, tz=tz)
    else:
        dt = datetime.fromtimestamp(epoch).astimezone()
    return "[" + dt.strftime("%a %Y-%m-%d %H:%M:%S %Z") + "]"


def strip_leading_message_timestamps(content: str, tz=None) -> Tuple[str, Optional[float]]:
    """Strip one or more leading gateway timestamp prefixes from ``content``.

    Returns ``(clean_content, embedded_epoch)``.  If multiple timestamp prefixes
    are present, the timestamp closest to the actual message text wins.  That
    preserves the original platform-send time for legacy contaminated rows like
    ``[processing time] [platform time] [sender] message``.
    """
    if not isinstance(content, str) or not content:
        return content, None

    text = content
    embedded_epoch: Optional[float] = None

    while True:
        match = _HUMAN_TIMESTAMP_RE.match(text) or _ISO_TIMESTAMP_RE.match(text)
        if not match:
            break
        parsed = _parse_timestamp_match(match, tz=tz)
        if parsed is not None:
            embedded_epoch = parsed
        text = text[match.end():]

    return text, embedded_epoch


def render_user_content_with_timestamp(content: str, ts_value: Any = None, tz=None) -> str:
    """Render a user message for LLM context with exactly one timestamp prefix.

    Existing leading timestamp prefixes are removed first.  If such a prefix was
    present, its parsed time wins over ``ts_value``; otherwise ``ts_value`` is
    formatted and prepended.  If no timestamp is available, the cleaned content is
    returned unchanged.
    """
    clean_content, embedded_epoch = strip_leading_message_timestamps(content, tz=tz)
    effective_ts = embedded_epoch if embedded_epoch is not None else ts_value
    prefix = format_message_timestamp(effective_ts, tz=tz)
    if not prefix:
        return clean_content
    if clean_content:
        return f"{prefix} {clean_content}"
    return prefix


def _parse_timestamp_prefix(text: str, tz=None) -> Optional[float]:
    match = _HUMAN_TIMESTAMP_RE.match(text) or _ISO_TIMESTAMP_RE.match(text)
    if not match:
        return None
    return _parse_timestamp_match(match, tz=tz)


def _parse_timestamp_match(match: re.Match, tz=None) -> Optional[float]:
    if "iso" in match.groupdict() and match.group("iso"):
        iso_text = match.group("iso")
        try:
            dt = datetime.fromisoformat(iso_text)
        except ValueError:
            try:
                dt = datetime.strptime(iso_text, "%Y-%m-%dT%H:%M:%S%z")
            except ValueError:
                return None
        if dt.tzinfo is None:
            if tz is not None:
                dt = dt.replace(tzinfo=tz)
            else:
                dt = dt.astimezone()
        return float(dt.timestamp())

    date_part = match.group("date")
    time_part = match.group("time")
    try:
        dt = datetime.strptime(f"{date_part} {time_part}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    if tz is not None:
        dt = dt.replace(tzinfo=tz)
    else:
        dt = dt.astimezone()
    return float(dt.timestamp())
