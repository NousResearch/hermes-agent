"""One-way notification for transcript rows a plugin injected.

The adviser writes a real user-role row (display_kind=adviser) so the model sees the advice and
the transcript survives resume. A live surface needs the same event at injection time, or the
card only appears after a rehydrate. The callback is installed by the surface (the gateway) and
is optional: an agent without it behaves exactly as before.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def notify_plugin_message(agent: Any, row: Any) -> None:
    """Tell the surface about one injected row. Never raises into the turn path."""
    if not isinstance(row, dict):
        return
    callback = getattr(agent, "plugin_message_callback", None)
    if not callable(callback):
        return
    try:
        callback(dict(row))
    except Exception:
        logger.debug("plugin_message_callback failed", exc_info=True)


__all__ = ["notify_plugin_message"]
