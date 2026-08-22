"""WhatsApp platform plugin.

Keep this module import-light: deferred plugin discovery imports it to register
client tools without materializing the gateway adapter.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register(ctx) -> None:
    try:
        from .tools import register_tools

        register_tools(ctx)
    except Exception:
        logger.warning("WhatsApp: failed to register group admin tool", exc_info=True)

    from .adapter import register as register_platform

    register_platform(ctx)


__all__ = ["register"]
