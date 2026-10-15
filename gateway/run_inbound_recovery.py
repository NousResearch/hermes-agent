"""Idle-command pause preservation and new-turn gates for inbound messages."""
import asyncio
import logging

from agent.i18n import t

logger = logging.getLogger("gateway.run")


class GatewayInboundRecoveryMixin:
    async def _authority_or_recovery_reply(self, event, source, quick_key, recovery_key, is_internal):
        owned, reply = await self._hm_authority_admission(event, source, quick_key, is_internal)
        if owned:
            return True, reply
        # A confirmation await may have admitted recovery while authority was checked.
        busy = (self._paused_recovery_busy_reply(quick_key)
                or self._paused_recovery_busy_reply(recovery_key))
        return busy is not None, busy

    async def _dispatch_idle_with_pause_preservation(self, event, source, quick_key, recovery_key):
        # A paused command may clear the pause; keep accepted adapter work parked until it commits.
        command_entry = (await self.async_session_store.lookup_by_session_key(recovery_key)
                         if event.get_command() else None)
        paused = getattr(command_entry, "compression_paused", False) is True
        if paused:
            event._gateway_preserve_pending = True
        try:
            return await self._hm_dispatch_idle_commands(event, source, quick_key)
        finally:
            if paused:
                recovered = await self.async_session_store.lookup_by_session_key(recovery_key)
                event._gateway_preserve_pending = (recovered is None or recovered.compression_paused)

    async def _idle_turn_gate(self, source, quick_key):
        if await asyncio.to_thread(self._is_telegram_topic_root_lobby, source):
            if self._should_send_telegram_lobby_reminder(source):
                return True, self._telegram_topic_root_lobby_message()
            return True, None
        if self._external_drain_active:
            logger.info("Refusing new turn for session %s — external drain active.", quick_key)
            return True, t("gateway.busy.draining_maintenance")
        return False, None
