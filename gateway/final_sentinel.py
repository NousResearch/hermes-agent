"""Gateway final-message sentinel helpers.

The sentinel is user-facing completion clarity for Telegram/Discord gateway
turns. It is intentionally separate from runtime telemetry/footer handling and
from model-generated response text.
"""

from __future__ import annotations

from typing import Any


FINAL_MESSAGE_SENTINEL = "COMPLETE"
SUPPORTED_FINAL_SENTINEL_PLATFORMS = {"telegram", "discord"}


def platform_name(platform: Any) -> str:
    """Normalize a Platform enum / raw platform value to lowercase text."""
    value = getattr(platform, "value", platform)
    return str(value or "").lower()


def is_final_sentinel_platform(platform: Any) -> bool:
    """Return True when this platform should receive final sentinels."""
    return platform_name(platform) in SUPPORTED_FINAL_SENTINEL_PLATFORMS


def strip_trailing_final_sentinel(text: str) -> str:
    """Remove a model-produced trailing standalone sentinel if present.

    The durable marker is sent by the gateway as a separate message. If the
    model includes a final standalone COMPLETE line, strip it from the body so
    the user does not see duplicates or a marker buried in the main response.
    """
    if not isinstance(text, str) or not text:
        return text

    lines = text.rstrip().splitlines()
    if not lines:
        return ""
    if lines[-1].strip() != FINAL_MESSAGE_SENTINEL:
        return text
    return "\n".join(lines[:-1]).rstrip()


def should_send_final_sentinel(
    *,
    platform: Any,
    message_type: Any = None,
    response_delivered: bool = False,
    is_ephemeral_response: bool = False,
    failed: bool = False,
    suppressed_or_stale: bool = False,
) -> bool:
    """Return True when a gateway turn should emit standalone COMPLETE.

    Gate out non-user-facing/system-ish paths. The caller must only pass
    response_delivered=True after the main visible response has actually been
    delivered (normal path) or was confirmed delivered by streaming.
    """
    if not is_final_sentinel_platform(platform):
        return False
    if not response_delivered:
        return False
    if failed or suppressed_or_stale or is_ephemeral_response:
        return False

    message_type_value = getattr(message_type, "value", message_type)
    if str(message_type_value or "").lower() == "command":
        return False

    return True
