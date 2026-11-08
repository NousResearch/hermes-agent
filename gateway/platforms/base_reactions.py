"""The reaction contract ``BasePlatformAdapter`` inherits: one flag plus two default coroutines."""

from typing import Any, Optional


class ReactionSupportMixin:
    """Reaction surface split out of the base facade to keep ``base.py`` from growing.

    Adapters that can react set ``SUPPORTS_REACTIONS = True`` and override these; everyone else
    inherits the stubs, which answer ``success: False`` instead of guessing a target.
    """

    # Whether this adapter can attach/retract emoji reactions. Off by default: the two coroutines
    # below always exist, so callers advertise the capability by flag, never by method probing.
    SUPPORTS_REACTIONS: bool = False

    async def add_reaction(
        self,
        chat_id: str,
        emoji: str,
        message_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Attach ``emoji`` to a message in ``chat_id``; at least ``{"success": bool}`` plus
        ``error`` on failure. One signature over platform-native shapes (Signal: author +
        timestamp; Telegram: replaces all reactions; WhatsApp: bridge, original message key).

        ``message_id=None`` means "the inbound message being responded to" — adapters that cannot
        resolve that target return ``success: False`` rather than guessing."""
        return {"success": False, "error": f"{self.name} does not support message reactions"}

    async def remove_reaction(
        self,
        chat_id: str,
        message_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Retract a reaction previously added by ``add_reaction`` (best-effort); same result shape."""
        return {"success": False, "error": f"{self.name} does not support message reactions"}
