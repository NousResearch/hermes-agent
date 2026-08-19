"""Backward-compatible imports for gateway timestamp helpers.

The implementation lives at the agent layer so every Hermes surface can render
the same clean, timestamp-aware model context.
"""

from agent.message_timestamps import (  # noqa: F401
    coerce_message_timestamp,
    format_message_timestamp,
    render_user_content_with_timestamp,
    strip_leading_message_timestamps,
)

__all__ = [
    "coerce_message_timestamp",
    "format_message_timestamp",
    "render_user_content_with_timestamp",
    "strip_leading_message_timestamps",
]
