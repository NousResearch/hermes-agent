"""Discord disconnect and voice inactivity lifecycle."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Optional

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

    async def join_voice_channel(self: DiscordAdapter, channel, *, text_channel_id: int | None = None, source: dict[str, Any] | None = None) -> bool:
        """Join a voice channel; returns True on success. ``text_channel_id`` stores the
        transcription-routing binding so programmatic joins work without ``/voice join``."""
        from plugins.platforms.discord.adapter import DISCORD_AVAILABLE, VoiceReceiver

        if not self._client or not DISCORD_AVAILABLE:
            return False
        guild_id = channel.guild.id
        async with self._voice_locks.setdefault(guild_id, asyncio.Lock()):
            existing = self._voice_clients.get(guild_id)
            if existing and existing.is_connected():
                if existing.channel.id != channel.id:
                    await existing.move_to(channel)
                self._reset_voice_timeout(guild_id)
                self._bind_voice_channel(guild_id, text_channel_id, source)
                return True
            vc = await channel.connect()
            self._voice_clients[guild_id] = vc
            self._reset_voice_timeout(guild_id)
            self._bind_voice_channel(guild_id, text_channel_id, source)
            try:
                receiver = VoiceReceiver(vc, allowed_user_ids=self._allowed_user_ids)
                receiver.start()
                self._voice_receivers[guild_id] = receiver
                self._voice_listen_tasks[guild_id] = asyncio.ensure_future(
                    self._voice_listen_loop(guild_id)
                )
            except Exception as e:
                logger.warning("Voice receiver failed to start: %s", e)
            # Mixer is best-effort; failure falls back to one-shot FFmpegPCMAudio playback.
            if getattr(self, "_voice_fx_cfg", {}).get("enabled"):
                try:
                    await self._install_voice_mixer(guild_id, vc)
                except Exception as e:
                    logger.warning("Voice mixer failed to start: %s", e)
            return True

    def _bind_voice_channel(
        self: DiscordAdapter, guild_id: int, text_channel_id: Optional[int],
        source: Optional[dict[str, Any]],
    ) -> None:
        previous = self._voice_text_channels.get(guild_id)
        if text_channel_id is not None and previous is not None and previous != text_channel_id:
            self._notify_voice_disconnect(previous)
        self._bind_voice_text_channel(guild_id, text_channel_id, source)

    async def leave_voice_channel(self: DiscordAdapter, guild_id: int) -> None:
        """Disconnect from the voice channel in a guild."""
        async with self._voice_locks.setdefault(guild_id, asyncio.Lock()):
            await self._leave_voice_channel_locked(guild_id)

    async def _leave_voice_channel_locked(self: DiscordAdapter, guild_id: int) -> None:
        receiver = self._voice_receivers.pop(guild_id, None)
        pending_inputs = []
        if receiver:
            pending_inputs = receiver.flush_pending()
            receiver.stop()
        listen_task = self._voice_listen_tasks.pop(guild_id, None)
        if listen_task:
            listen_task.cancel()
        guild = self._client.get_guild(guild_id) if self._client is not None else None
        captured_for = self._voice_text_channels.get(guild_id)
        for user_id, pcm_data in pending_inputs:
            if self._is_allowed_user(str(user_id), guild=guild, is_dm=False):
                await self._process_voice_input(guild_id, user_id, pcm_data, captured_for)
        # Tear down the mixer (stops the continuous outgoing stream).
        if getattr(self, "_voice_mixers", None) is not None:
            self._voice_mixers.pop(guild_id, None)
        vc = self._voice_clients.pop(guild_id, None)
        if vc and vc.is_connected():
            try:
                if vc.is_playing():
                    vc.stop()
            except Exception:
                pass
            await vc.disconnect()
        task = self._voice_timeout_tasks.pop(guild_id, None)
        if task:
            task.cancel()
        self._voice_text_channels.pop(guild_id, None)
        self._voice_sources.pop(guild_id, None)

    async def _handle_voice_state_update(self: DiscordAdapter, member, before, after) -> None:
        """Track voice channel join/leave events."""
        bot_guild_ids = set(self._voice_clients.keys())
        if not bot_guild_ids:
            return
        guild_id = member.guild.id
        if guild_id not in bot_guild_ids:
            return
        if self._client is None:
            return
        if member == self._client.user:
            if before.channel is None or after.channel is not None:
                return
            voice_client = self._voice_clients.get(guild_id)
            async with self._voice_locks.setdefault(guild_id, asyncio.Lock()):
                if self._voice_clients.get(guild_id) is not voice_client:
                    return
                if voice_client is None or voice_client.channel.id != before.channel.id:
                    return
                self._notify_voice_disconnect(self._voice_text_channels.get(guild_id))
                await self._leave_voice_channel_locked(guild_id)
            return
        joined = before.channel is None and after.channel is not None
        left = before.channel is not None and after.channel is None
        switched = (
            before.channel is not None
            and after.channel is not None
            and before.channel != after.channel
        )
        if joined or left or switched:
            logger.info(
                "Voice state: %s (%d) %s (guild %d)",
                member.display_name,
                member.id,
                "joined " + after.channel.name if joined
                else "left " + before.channel.name if left
                else f"moved {before.channel.name} -> {after.channel.name}",
                guild_id,
            )
            # Any membership change in the bot's channel bumps the
            # DAVE (E2EE) epoch — re-resolve the receiver's decryption
            # state so it never decodes against a stale session.
            vc = self._voice_clients.get(guild_id)
            receiver = self._voice_receivers.get(guild_id)
            if vc is not None and receiver is not None:
                bot_channel = getattr(vc, "channel", None)
                if bot_channel is not None and (
                    before.channel == bot_channel
                    or after.channel == bot_channel
                ):
                    receiver.refresh_credentials("membership change")
