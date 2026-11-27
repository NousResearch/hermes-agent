"""WhatsApp platform plugin."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

__all__ = ["register"]


def register(ctx) -> None:
    """Register outbound client tools and the inbound platform adapter."""
    try:
        from .tools import register_tools

        register_tools(ctx)
    except Exception:  # noqa: BLE001 - preserve platform loading if a client tool breaks
        logger.warning("WhatsApp: failed to register client tools", exc_info=True)

    from .adapter import register as register_adapter

    register_adapter(ctx)
