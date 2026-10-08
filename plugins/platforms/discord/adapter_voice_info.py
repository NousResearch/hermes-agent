"""Who is in the bot's Discord voice channel and who is speaking, for status and prompt context."""
from __future__ import annotations

import time
from typing import Any, Dict, Optional


class DiscordVoiceInfoMixin:
    _voice_clients: dict[int, Any]
    _voice_receivers: dict[int, Any]
    _voice_text_channels: dict[int, Any]
    _voice_sources: dict[int, Dict[str, Any]]
    _client: Any
    discard_pending_voice_input: Any

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

    def get_voice_channel_info(self, guild_id: int) -> Optional[Dict[str, Any]]:
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
