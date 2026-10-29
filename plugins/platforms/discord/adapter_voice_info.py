"""Who is in the bot's Discord voice channel and who is speaking, for status and prompt context."""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class DiscordVoiceInfoMixin:
    _voice_clients: dict[int, Any]
    _voice_receivers: dict[int, Any]
    _voice_text_channels: dict[int, Any]
    _voice_sources: dict[int, dict[str, Any]]
    _voice_listen_tasks: dict[int, Any]
    _voice_timeout_tasks: dict[int, Any]
    _client: Any
    discard_pending_voice_input: Any
    is_in_voice_channel: Any

    def is_listening_in_voice_channel(self, guild_id: int) -> bool:
        """Connection/playback alone is not a live speech receiver and polling loop."""
        receiver = self._voice_receivers.get(guild_id)
        task = self._voice_listen_tasks.get(guild_id)
        return bool(self.is_in_voice_channel(guild_id) and receiver and receiver._running
                    and task is not None and not task.done())

    async def _abort_voice_join(self, guild_id: int, client: Any, receiver: Any) -> None:
        """Retire a failed join without transcribing or dispatching partially captured speech.

        The caller holds the guild lock, so the ordinary leave path cannot be re-entered.
        """
        if self._voice_receivers.get(guild_id) is receiver:
            self._voice_receivers.pop(guild_id, None)
        if receiver is not None:
            try:
                receiver.stop()
            except Exception as error:  # health: allow BLE001 -- Teardown must continue independently; exception details may contain private voice data.
                logger.warning("Voice receiver cleanup failed (guild=%d, error_type=%s)", guild_id, type(error).__name__)
        if self._voice_clients.get(guild_id) is not client:
            return
        for tasks in (self._voice_listen_tasks, self._voice_timeout_tasks):
            task = tasks.pop(guild_id, None)
            if task is not None:
                task.cancel()
        self._voice_text_channels.pop(guild_id, None)
        self._voice_sources.pop(guild_id, None)
        mixers = getattr(self, "_voice_mixers", None)
        if mixers is not None:
            mixers.pop(guild_id, None)
        self._voice_clients.pop(guild_id)
        try:
            if client.is_playing():
                client.stop()
        except Exception as error:  # health: allow BLE001 -- Teardown must still disconnect; exception details may contain private voice data.
            logger.warning("Voice playback cleanup failed (guild=%d, error_type=%s)", guild_id, type(error).__name__)
        try:
            await client.disconnect()
        except Exception as error:  # health: allow BLE001 -- Failed join already reports not-ready; retain type-only cleanup diagnostics.
            logger.warning("Voice disconnect after failed join failed (guild=%d, error_type=%s)",
                           guild_id, type(error).__name__)

    def _bind_voice_text_channel(self, guild_id: int, text_channel_id: Any, source: Optional[dict]) -> None:
        """A programmatic join's transcription binding, on every path: cold join, same voice channel and
        move. The already-connected paths returned before writing it, so a successful join to another
        text channel kept answering in the old one. Moving to another text channel drops speech captured
        for the old one (as ``/voice join`` does) and, unless a new source is given, the old channel's
        bound source, whose chat would otherwise keep routing turns into the old conversation."""
        if text_channel_id is not None:
            previous = self._voice_text_channels.get(guild_id)
            if previous is not None and previous != text_channel_id:
                self.discard_pending_voice_input(guild_id)
                if source is None:
                    self._voice_sources.pop(guild_id, None)
            self._voice_text_channels[guild_id] = text_channel_id
        if source is not None:
            self._voice_sources[guild_id] = source

    def get_voice_channel_info(self, guild_id: int) -> Optional[dict[str, Any]]:
        """Return voice channel info (name, members, count, speaking user IDs) or None if not connected."""
        vc = self._voice_clients.get(guild_id)
        if not vc or not vc.is_connected():
            return None
        channel = vc.channel
        if not channel:
            return None
        members_info = []
        bot_user = self._client.user if self._client else None
        for m in channel.members:
            if bot_user and m.id == bot_user.id:
                continue  # skip the bot itself
            members_info.append({"user_id": m.id, "display_name": m.display_name, "is_bot": m.bot})
        speaking_user_ids: set = set()
        receiver = self._voice_receivers.get(guild_id)
        if receiver:
            now = time.monotonic()
            with receiver._lock:
                for ssrc, last_t in receiver._last_packet_time.items():
                    if now - last_t < 2.0:
                        uid = receiver._ssrc_to_user.get(ssrc)
                        if uid:
                            speaking_user_ids.add(uid)
        for info in members_info:
            info["is_speaking"] = info["user_id"] in speaking_user_ids
        return {
            "channel_name": channel.name, "member_count": len(members_info),
            "members": members_info, "speaking_count": len(speaking_user_ids),
        }

    def get_voice_channel_context(self, guild_id: int) -> str:
        """Return a human-readable voice channel context string for prompt injection."""
        info = self.get_voice_channel_info(guild_id)
        if not info:
            return ""
        parts = [f"[Voice channel: #{info['channel_name']} — {info['member_count']} participant(s)]"]
        for m in info["members"]:
            status = " (speaking)" if m["is_speaking"] else ""
            parts.append(f"  - {m['display_name']}{status}")
        return "\n".join(parts)
