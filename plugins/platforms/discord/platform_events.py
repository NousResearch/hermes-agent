"""Discord platform notification normalization and dispatch."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from plugins.platforms.discord.adapter import DiscordAdapter

logger = logging.getLogger("plugins.platforms.discord.adapter")


class DiscordPlatformEventsMixin:
    def _thread_id_and_chat_for_channel(self: DiscordAdapter, channel) -> tuple[Optional[str], Optional[str]]:
        """Return ``(thread_id, chat_id)``; for a thread chat_id is the thread id (dispatch session key)."""
        if channel is None:
            return None, None
        chan_id = getattr(channel, "id", None)
        if chan_id is None:
            return None, None
        from plugins.platforms.discord.adapter import discord

        is_thread = isinstance(channel, getattr(discord, "Thread", ()))
        return (str(chan_id) if is_thread else None), str(chan_id)

    def _source_for_platform_event(
        self: DiscordAdapter, *, chat_id: str, user_id: Optional[str], user_name: Optional[str],
        thread_id: Optional[str], guild_id: Optional[str], message_id: Optional[str] = None,
    ):
        """Build the SessionSource the gateway authorizes against; missing identity raises (fail closed)."""
        if not user_id or not chat_id:
            raise ValueError("gateway_platform_event requires actor and chat identities")
        return self.build_source(
            chat_id=chat_id, chat_type="thread" if thread_id else "group", user_id=user_id,
            user_name=user_name, thread_id=thread_id, guild_id=guild_id, message_id=message_id,
        )

    async def _fire_platform_event(self: DiscordAdapter, event: dict[str, Any], source) -> None:
        """Forward one envelope to the gateway boundary; no callback -> fail closed, errors never escape."""
        handler = getattr(self, "_platform_event_handler", None)
        if handler is None:
            return
        try:
            await handler(event, source)
        except Exception:
            logger.debug("[%s] gateway_platform_event dispatch error", self.name, exc_info=True)

    @staticmethod
    def _platform_events_subscribed() -> bool:
        """has_hook fast-path shared by every Discord fire-site."""
        try:
            from hermes_cli.lifecycle import has_hook
            return has_hook("gateway_platform_event")
        except Exception:
            return False

    async def _emit_platform_event(self: DiscordAdapter, event_type: str, build) -> None:
        """Normalize one event via ``build()`` -> ``(payload, source_kwargs)`` (None drops) and dispatch."""
        if not self._platform_events_subscribed():
            return
        try:
            built = build()
            if built is None:
                return
            payload, source_kwargs = built
            event = {"platform": "discord", "event_type": event_type, "payload": payload}
            source = self._source_for_platform_event(**source_kwargs)
        except Exception:
            logger.debug("[%s] %s normalize error", self.name, event_type, exc_info=True)
            return
        await self._fire_platform_event(event, source)

    def _message_event_parts(self: DiscordAdapter, message, extra_payload):
        """Shared normalizer for message edit/delete: (payload, source kwargs) or None."""
        author = getattr(message, "author", None)
        if author is not None and getattr(author, "bot", False):
            return None  # bot's own progressive edits are noise, not user events
        thread_id, chat_id = self._thread_id_and_chat_for_channel(getattr(message, "channel", None))
        message_id = getattr(message, "id", None)
        if chat_id is None or message_id is None:
            return None
        guild = getattr(message, "guild", None)
        payload = {
            "chat_id": str(chat_id)[:128], "message_id": str(message_id)[:128],
            "thread_id": thread_id[:128] if thread_id else None, **extra_payload(message, author),
        }
        return payload, dict(
            chat_id=str(chat_id), user_id=str(getattr(author, "id", "") or "") or None,
            user_name=getattr(author, "display_name", None), thread_id=thread_id,
            guild_id=str(getattr(guild, "id", "")) if guild else None, message_id=str(message_id),
        )

    @staticmethod
    def _thread_event_parts(thread, extra_payload):
        """Shared normalizer for thread create/rename; the owner is the authorized actor
        because Discord's event carries none (same trade-off as ``message_deleted``)."""
        thread_id = getattr(thread, "id", None)
        owner_id = getattr(thread, "owner_id", None)
        if thread_id is None:
            return None
        parent_id = getattr(thread, "parent_id", None)
        guild = getattr(thread, "guild", None)
        payload = {
            "thread_id": str(thread_id)[:128],
            "parent_chat_id": str(parent_id)[:128] if parent_id is not None else None,
            **extra_payload(thread, owner_id),
        }
        return payload, dict(
            chat_id=str(thread_id), user_id=str(owner_id) if owner_id is not None else None,
            user_name=None, thread_id=str(thread_id),
            guild_id=str(getattr(guild, "id", "")) if guild else None,
        )

    async def _on_platform_message_edit(self: DiscordAdapter, before, after) -> None:
        """Normalize ``on_message_edit`` into event_type ``message_edited``."""
        def _extra(message, author):
            text = getattr(message, "content", None)
            edited_at = getattr(message, "edited_at", None)
            return {
                "text": text[:8192] if isinstance(text, str) else None,
                "edited_at": (
                    str(edited_at.isoformat())[:64]
                    if edited_at is not None and hasattr(edited_at, "isoformat")
                    else None
                ),
            }
        message = after if after is not None else before
        await self._emit_platform_event("message_edited", lambda: self._message_event_parts(message, _extra))

    async def _on_platform_message_delete(self: DiscordAdapter, message) -> None:
        """Normalize ``on_message_delete`` into ``message_deleted``. Discord omits the
        deleter, so the author (the only cached identity) is the source; uncached deletions never fire."""
        def _extra(message, author):
            return {"author_id": str(getattr(author, "id", "") or "")[:128] or None}
        await self._emit_platform_event("message_deleted", lambda: self._message_event_parts(message, _extra))

    async def _on_platform_thread_create(self: DiscordAdapter, thread) -> None:
        """Normalize ``on_thread_create`` into event_type ``thread_created``."""
        def _extra(thread, owner_id):
            name = getattr(thread, "name", None)
            return {
                "name": name[:256] if isinstance(name, str) else None,
                "owner_id": str(owner_id)[:128] if owner_id is not None else None,
            }
        await self._emit_platform_event("thread_created", lambda: self._thread_event_parts(thread, _extra))

    async def _on_platform_thread_update(self: DiscordAdapter, before, after) -> None:
        """Normalize ``on_thread_update`` renames into ``thread_renamed``; non-rename updates are dropped."""
        def _build():
            old_name = getattr(before, "name", None)
            new_name = getattr(after, "name", None)
            if old_name == new_name or not isinstance(new_name, str):
                return None
            return self._thread_event_parts(after, lambda _t, _o: {
                "old_name": old_name[:256] if isinstance(old_name, str) else None,
                "new_name": new_name[:256],
            })
        await self._emit_platform_event("thread_renamed", _build)

    async def _on_raw_message_delete(self: DiscordAdapter, payload) -> None:
        self._withdraw_discord_messages(payload, (str(payload.message_id),))

    async def _on_raw_bulk_message_delete(self: DiscordAdapter, payload) -> None:
        self._withdraw_discord_messages(payload, tuple(str(mid) for mid in payload.message_ids))

    def _withdraw_discord_messages(self: DiscordAdapter, payload, message_ids: tuple[str, ...]) -> None:
        from gateway.native_message_deletion import NativeMessageDeletion
        channel = self._client.get_channel(payload.channel_id) if self._client else None
        thread_id, _ = self._thread_id_and_chat_for_channel(channel)
        self.withdraw_native_messages(NativeMessageDeletion(
            platform=self.platform,
            scope_id=str(payload.guild_id) if payload.guild_id is not None else None,
            chat_id=str(payload.channel_id), message_ids=message_ids, thread_id=thread_id))
