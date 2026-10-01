"""Discord disconnect and voice inactivity lifecycle."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from plugins.platforms.discord.adapter import DiscordAdapter

logger = logging.getLogger("plugins.platforms.discord.adapter")


class DiscordVoiceLifecycleMixin:
    async def disconnect(self: DiscordAdapter) -> None:
        """Disconnect from Discord."""
        self._disconnecting = True
        # Cancel the liveness probe first so it can't fire a spurious fatal/reconnect mid-teardown.
        await self._cancel_liveness_task()
        # The runner cancels a disconnect that outlasts its budget, and a leave can block on a dead
        # gateway WS, so report the ended calls before leaving.
        for text_ch_id in list(self._voice_text_channels.values()):
            self._notify_voice_disconnect(text_ch_id)
        # Leave voice *before* cancelling the bot task: VoiceClient.disconnect() needs the main
        # gateway WS (run by the bot task) or it blocks until the timeout.
        for guild_id in list(self._voice_clients.keys()):
            try:
                await self.leave_voice_channel(guild_id)
            except Exception as e:  # pragma: no cover - defensive logging
                logger.debug("[%s] Error leaving voice channel %s: %s", self.name, guild_id, e)
        # Cancel the bot task before closing: after a connect() timeout client.start() may still run
        # and discord.py's reconnect loop can ignore the closed flag mid-handshake.
        await self._cancel_bot_task()
        if self._client:
            try:
                await self._client.close()
            except Exception as e:  # pragma: no cover - defensive logging
                logger.warning("[%s] Error during disconnect: %s", self.name, e, exc_info=True)
        for task in (self._post_connect_task, self._missed_message_backfill_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._running = False
        self._client = None
        self._ready_event.clear()
        self._post_connect_task = None
        self._liveness_task = None
        self._missed_message_backfill_task = None
        self._release_platform_lock()
        logger.info("[%s] Disconnected", self.name)




    def _cancel_voice_timeout(self: DiscordAdapter, guild_id: int) -> None:
        task = self._voice_timeout_tasks.pop(guild_id, None)
        if task:
            task.cancel()


    def _reset_voice_timeout(self: DiscordAdapter, guild_id: int) -> None:
        """Reset the auto-disconnect inactivity timer."""
        self._cancel_voice_timeout(guild_id)
        timeout = self._voice_timeout_limit()
        if timeout <= 0:
            logger.debug("Voice inactivity timeout disabled (guild=%d)", guild_id)
            return
        self._voice_timeout_tasks[guild_id] = asyncio.ensure_future(
            self._voice_timeout_handler(guild_id, timeout)
        )


    async def _voice_timeout_handler(self: DiscordAdapter, guild_id: int, timeout: Optional[int] = None) -> None:
        """Auto-disconnect after the configured inactivity timeout."""
        from plugins.platforms.discord.adapter import t

        timeout = self._voice_timeout_limit() if timeout is None else int(timeout)
        if timeout <= 0:
            return
        try:
            await asyncio.sleep(timeout)
        except asyncio.CancelledError:
            return
        text_ch_id = self._voice_text_channels.get(guild_id)
        # ``/voice off`` keeps the bot in the channel; only the bot's own audio counts as
        # activity, so the timer would fire every VOICE_TIMEOUT and spam "Left voice channel".
        _mode_getter = getattr(self, "_voice_mode_getter", None)
        if text_ch_id is not None and _mode_getter is not None:
            try:
                if _mode_getter(str(text_ch_id)) == "off":
                    return
            except Exception:
                pass
        await self.leave_voice_channel(guild_id)
        self._notify_voice_disconnect(text_ch_id)
        if text_ch_id and self._client:
            ch = self._client.get_channel(text_ch_id)
            if ch:
                try:
                    await ch.send(t("platform.discord.voice.left_inactivity"))
                except Exception:
                    pass



    def _notify_voice_disconnect(self: DiscordAdapter, text_ch_id: Optional[int]) -> None:
        """Tell the runner that the call bound to ``text_ch_id`` ended so it resets the voice mode."""
        if not (self._on_voice_disconnect and text_ch_id):
            return
        try:
            self._on_voice_disconnect(str(text_ch_id))
        except Exception:
            pass
