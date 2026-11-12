"""Telegram replacement-adapter delegation: a turn that started on an adapter the reconnect watcher
later retired keeps that instance, so its sends, edits and deletes hand off to the live replacement
in the SAME profile (#94498, #98228, #143364)."""

import asyncio
import logging
from typing import Optional

from hermes_agent.gateway.platforms.base import SendResult

logger = logging.getLogger("plugins.platforms.telegram.adapter")


class TelegramReplacementMixin:
    """Replacement lookup and reconnect wait for ``TelegramAdapter``."""

    def _replacement_telegram_adapter(self) -> "Optional[TelegramReplacementMixin]":
        """Live adapter if the reconnect watcher replaced us in our own profile's adapter map
        (an in-flight turn still holds the old instance whose ``_bot`` stays None).

        The lookup stays inside this adapter's profile: ``runner.adapters`` holds only the PRIMARY
        profile's map, so resolving there hands a torn-down secondary the primary's bot and its
        in-flight progress leaves under another profile's identity (#143364).
        ``_adapters_for_profile`` already encodes the contract: a named profile with no live adapter
        gets ``{}`` (fail closed), the primary keeps its own map."""
        lookup = getattr(getattr(self, "gateway_runner", None), "_adapters_for_profile", None)
        if lookup is None:
            return None
        live = (lookup(getattr(self, "_owner_profile", None)) or {}).get(self.platform)
        if live is not None and live is not self and getattr(live, "_bot", None):
            return live
        return None

    async def _edit_via_replacement(self, chat_id: str, message_id: str, content: str, **kwargs):
        """``edit_message`` on a retired adapter: the live replacement edits, else "Not connected" (#98228)."""
        live = self._replacement_telegram_adapter()
        if live is None:
            return SendResult(success=False, error="Not connected")
        return await live.edit_message(chat_id, message_id, content, **kwargs)

    async def _wait_for_reconnection(self) -> bool:
        """Wait for ``_bot`` or a replacement adapter; False on expiry or permanent fatal."""
        if self._bot or self._replacement_telegram_adapter() is not None:
            return True
        if self._is_permanent_fatal():
            return False
        wait_s = float(getattr(self, "_RECONNECT_WAIT_SECONDS", 15.0))
        poll_s = float(getattr(self, "_RECONNECT_POLL_INTERVAL", 0.5))
        logger.info("[%s] Not connected — waiting for reconnection (up to %.0fs)", self.name, wait_s)
        waited = 0.0
        while waited < wait_s:
            await asyncio.sleep(poll_s)
            waited += poll_s
            if self._is_permanent_fatal():
                return False
            if self._bot or self._replacement_telegram_adapter() is not None:
                logger.info("[%s] Reconnected after %.1fs", self.name, waited)
                return True
        logger.warning("[%s] Still not connected after %.0fs", self.name, wait_s)
        return False
