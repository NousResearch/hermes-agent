"""User-facing final-message sentinel helpers for gateway delivery.

The sentinel is a display-only UX marker for normal Telegram/Discord replies.
It is intentionally separate from runtime telemetry/footer handling.
"""

from __future__ import annotations

import re
from typing import Any

FINAL_MESSAGE_SENTINEL = "COMPLETE"
SUPPORTED_FINAL_SENTINEL_PLATFORMS = frozenset({"telegram", "discord"})
_SENTINEL_RE = re.compile(r"(?:\n\s*)+COMPLETE\s*$")


def platform_name(platform: Any) -> str:
    """Normalize a Platform enum / raw string into a lowercase name."""
    value = getattr(platform, "value", platform)
    return str(value or "").lower()


def should_send_final_sentinel(
    platform: Any,
    *,
    is_internal: bool = False,
    is_command: bool = False,
) -> bool:
    """Return True when a standalone final sentinel should be sent."""
    if is_internal or is_command:
        return False
    return platform_name(platform) in SUPPORTED_FINAL_SENTINEL_PLATFORMS


def strip_trailing_final_sentinel(content: str) -> tuple[str, bool]:
    """Remove a model-produced trailing standalone COMPLETE if present.

    Non-streaming delivery can then send the canonical standalone marker once.
    Streaming delivery cannot retract already-sent text, so callers there should
    use this only for detection/duplicate suppression.
    """
    if not content:
        return content, False
    stripped = _SENTINEL_RE.sub("", content).rstrip()
    if stripped != content.rstrip():
        return stripped, True
    return content, False
