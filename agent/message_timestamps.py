"""Shared user-message timestamp handling for non-gateway agent turns.

The messaging gateway remains the behavioural oracle. This module shares its
formatting helpers and applies the same rendering/sidecar guard at the common
agent intake boundary without changing persisted user text.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz) if tz is not None else dt.astimezone()
    return float(dt.timestamp())


def _parse_iso(text: str, tz=None) -> Optional[float]:
    for parse in (datetime.fromisoformat, lambda value: datetime.strptime(value, "%Y-%m-%dT%H:%M:%S%z")):
        try:
            return _localize(parse(text), tz)
        except (TypeError, ValueError):
            continue
    return None


def _parse_timestamp_match(match: re.Match, tz=None) -> Optional[float]:
    if match.group("iso"):
        return _parse_iso(match.group("iso"), tz)
    try:
        dt = datetime.strptime(
            f"{match.group('date')} {match.group('time')}", "%Y-%m-%d %H:%M:%S"
        )
    except ValueError:
        return None
    return _localize(dt, tz)


def coerce_message_timestamp(ts_value: Any, tz=None) -> Optional[float]:
    """Return epoch seconds for gateway-supported timestamp values."""
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
    epoch = coerce_message_timestamp(ts_value, tz=tz)
    if epoch is None:
        return ""
    dt = datetime.fromtimestamp(epoch, tz=tz) if tz is not None else datetime.fromtimestamp(epoch).astimezone()
    return f"[{dt.strftime('%a %Y-%m-%d %H:%M:%S %Z')}]"


def strip_leading_message_timestamps(content: str, tz=None) -> Tuple[str, Optional[float]]:
    if not isinstance(content, str) or not content:
        return content, None
    text, embedded_epoch = content, None
    while (match := _TIMESTAMP_PREFIX_RE.match(text)) is not None:
        parsed = _parse_timestamp_match(match, tz=tz)
        if parsed is not None:
            embedded_epoch = parsed
        text = text[match.end():]
    return text, embedded_epoch


def prepare_fresh_user_message(content: str, ts_value: Any = None, tz=None) -> Tuple[str, Optional[float]]:
    """Apply gateway fresh-intake cleanup and supplied-time precedence."""
    clean_content, embedded_epoch = strip_leading_message_timestamps(content, tz=tz)
    supplied_epoch = coerce_message_timestamp(ts_value, tz=tz)
    return clean_content, supplied_epoch if supplied_epoch is not None else embedded_epoch


def render_user_content_with_timestamp(content: str, ts_value: Any = None, tz=None) -> str:
    clean_content, embedded_epoch = strip_leading_message_timestamps(content, tz=tz)
    prefix = format_message_timestamp(ts_value if embedded_epoch is None else embedded_epoch, tz=tz)
    return f"{prefix} {clean_content}" if prefix and clean_content else (prefix or clean_content)


def _legacy_gateway_message_timestamps_enabled(config: Optional[dict]) -> bool:
    if not isinstance(config, dict):
        return False
    gateway = config.get("gateway")
    if not isinstance(gateway, dict):
        return False
    configured = gateway.get("message_timestamps")
    if isinstance(configured, dict):
        return bool(configured.get("enabled", False))
    return bool(configured)


def message_timestamps_enabled(config: Optional[dict], *, messaging_gateway: bool = False) -> bool:
    """Resolve timestamps for the actual execution route.

    Messaging gateway turns retain their legacy-only gate. Other callers use
    an explicit top-level Boolean and otherwise retain the legacy fallback.
    """
    if messaging_gateway:
        return _legacy_gateway_message_timestamps_enabled(config)
    configured = config.get("message_timestamps") if isinstance(config, dict) else None
    if isinstance(configured, dict) and isinstance(configured.get("enabled"), bool):
        return configured["enabled"]
    return _legacy_gateway_message_timestamps_enabled(config)


_AUTO_CONTINUE_NOTE_PREFIXES = (
    "[System note: Your previous turn",
    "[System note: A new message",
)


def _strip_auto_continue_noise(content: str) -> str:
    """Match gateway replay cleanup while preserving the user's trailing text."""
    text = content
    while text.startswith(_AUTO_CONTINUE_NOTE_PREFIXES):
        end = text.find("]")
        if end < 0:
            return ""
        text = text[end + 1:].lstrip()
    return text


def render_message_timestamp_replay(
    messages: List[Dict[str, Any]], *, enabled: bool, current_turn_user_idx: int, tz=None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Build the gateway-compatible model replay without changing retained rows."""
    rendered_messages: List[Dict[str, Any]] = []
    rendered_current_idx = current_turn_user_idx
    for source_idx, source_message in enumerate(messages):
        message = dict(source_message)
        content = message.get("content")
        if message.get("role") == "user" and isinstance(content, str) and content:
            if not isinstance(message.get("api_content"), str) or not message.get("api_content"):
                message.pop("api_content", None)
            replay_timestamp = message.get("timestamp")
            # Gateway recovery cleanup applies to replayed history. The fresh
            # turn can legitimately contain the same text and must reach the
            # provider unchanged apart from normal timestamp rendering.
            if source_idx != current_turn_user_idx:
                body, embedded_timestamp = strip_leading_message_timestamps(content, tz=tz)
                cleaned = _strip_auto_continue_noise(body)
                if cleaned != body:
                    if not cleaned:
                        if source_idx < current_turn_user_idx:
                            rendered_current_idx -= 1
                        continue
                    content = cleaned
                    message["content"] = content
                    message.pop("api_content", None)
                    if embedded_timestamp is not None:
                        replay_timestamp = embedded_timestamp
            if enabled:
                rendered = render_user_content_with_timestamp(content, replay_timestamp, tz=tz)
                sidecar = message.get("api_content")
                if rendered != content and isinstance(sidecar, str) and sidecar and not (
                    sidecar == rendered or sidecar.startswith(rendered + "\n\n")
                ):
                    message.pop("api_content", None)
                message["content"] = rendered
        rendered_messages.append(message)
    return rendered_messages, rendered_current_idx
