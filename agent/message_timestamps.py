"""Render user-message timestamps in model context exactly once.

All Hermes surfaces can add temporal context while persisted message content stays
clean, so replay never accumulates ``[timestamp] [timestamp] ...`` prefixes.
"""

from __future__ import annotations

import errno
import re
import sys
import time
from datetime import datetime, UTC
from typing import Any, Optional

from hermes_time import safe_strftime


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


def render_model_history_with_timestamps(
    history: Optional[list[dict[str, Any]]], *, config: Optional[dict], tz=None,
) -> list[dict[str, Any]]:
    """Render a disposable request view; durable content and sidecars stay clean."""
    rendered_history = [dict(message) for message in (history or [])]
    if not message_timestamps_enabled(config):
        return rendered_history

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

    return rendered_history


def render_turn_with_message_timestamps(
    history: Optional[list[dict[str, Any]]],
    user_message: Any,
    *,
    config: Optional[dict],
    current_timestamp: Any = None,
    tz=None,
) -> tuple[list[dict[str, Any]], Any, Optional[float]]:
    """Return model-only timestamp rendering without mutating caller data."""
    rendered_history = render_model_history_with_timestamps(history, config=config, tz=tz)
    if not message_timestamps_enabled(config):
        return rendered_history, user_message, None

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


def _localize(dt: datetime, tz) -> Optional[float]:
    """Epoch for ``dt`` in ``tz`` or the local zone; None outside platform limits."""
    naive_local = dt.tzinfo is None and tz is None
    try:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz) if tz is not None else dt.astimezone()
        return float(dt.timestamp())
    except (OverflowError, ValueError):
        return None
    except OSError as exc:
        if not (sys.platform == "win32" and naive_local and exc.errno == errno.EINVAL):
            return None
        # Windows' fold probe can reject the first Unix day. mktime can
        # resolve those local wall times without changing their meaning.
        # Limit recovery to that fold-free interval and reject normalization.
        wall_time = dt.timetuple()
        try:
            epoch = time.mktime(wall_time)
            if 0 <= epoch < 86_400 and time.localtime(epoch)[:6] == wall_time[:6]:
                return float(epoch + dt.microsecond / 1_000_000)
        except (OSError, OverflowError, ValueError):
            pass
        return None


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
    try:
        # An early positive epoch can fall in 1969 locally. Starting from aware
        # UTC avoids Windows' negative-time fold probe in naive astimezone().
        dt = datetime.fromtimestamp(epoch, tz=tz) if tz is not None else datetime.fromtimestamp(epoch, tz=UTC).astimezone()
    except (OSError, OverflowError, ValueError):
        return ""
    return f"[{safe_strftime(dt, '%a %Y-%m-%d %H:%M:%S %Z')}]"


def strip_leading_message_timestamps(content: str, tz=None) -> tuple[str, Optional[float]]:
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
