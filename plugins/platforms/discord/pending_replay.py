"""Current Discord source authority and native identity for pending replay."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TYPE_CHECKING

from gateway.pending_native import PendingNativeInput
from gateway.session import SessionSource
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent

if TYPE_CHECKING:
    from discord.ext.commands import Bot
    from plugins.platforms.discord.adapter import DiscordMessage
    from plugins.platforms.discord.inbound_context import DiscordPreparedInput


def _native_message(message: Any, bot_id: str) -> dict[str, Any]:
    content = message.content.strip()
    for mention in (f"<@{bot_id}>", f"<@!{bot_id}>"):
        content = content.replace(mention, "").strip()
    reference = message.reference
    return {
        "content": content,
        "attachments": [
            {
                "id": str(item.id),
                "filename": item.filename,
                "size": item.size,
                "content_type": item.content_type,
            }
            for item in message.attachments
        ],
        "reference": None
        if reference is None
        else {
            "message_id": str(reference.message_id),
            "channel_id": str(reference.channel_id),
        },
    }


class DiscordPendingReplayMixin(BasePlatformAdapter):
    _client: Bot | None
    _discord_source_admission: Callable[[DiscordMessage], tuple[bool, bool]]
    _prepare_inbound_event: Callable[..., Awaitable[DiscordPreparedInput | None]]

    def pending_native_input(self, event: MessageEvent) -> PendingNativeInput | None:
        import discord

        if (
            not isinstance(event.raw_message, discord.Message)
            or self._client is None
            or self._client.user is None
        ):
            return None
        return PendingNativeInput.capture(
            event, _native_message(event.raw_message, str(self._client.user.id))
        )

    async def revalidate_pending_event(
        self,
        event: MessageEvent,
        *,
        authorize: Callable[[SessionSource], bool] | None = None,
    ) -> MessageEvent | None:
        import discord
        from hermes_constants import get_hermes_home

        native = event._pending_native_input
        if (
            self._client is None
            or self._client.user is None
            or native is None
            or event.internal
            or event._merged_parts
            or not event.message_id
            or any(event.media_text_inlined)
        ):
            return None
        origin = (
            event.source.parent_chat_id
            if event.source.auto_thread_created
            else event.source.chat_id
        )
        if not origin:
            return None
        try:
            channel = await asyncio.wait_for(
                self._client.fetch_channel(int(origin)), timeout=10
            )
            if not isinstance(
                channel, (discord.TextChannel, discord.Thread, discord.DMChannel)
            ):
                return None
            current = await asyncio.wait_for(
                channel.fetch_message(int(event.message_id)), timeout=10
            )
            guild = current.guild
            if (
                str(current.id) != event.message_id
                or str(current.author.id) != event.source.user_id
                or (str(guild.id) if guild else None) != event.source.scope_id
                or current.message_snapshots
            ):
                return None
            if guild is not None:
                current.author = await asyncio.wait_for(
                    guild.fetch_member(current.author.id), timeout=10
                )
            admitted, role_authorized = self._discord_source_admission(current)
            if not admitted:
                return None
            content = _native_message(current, str(self._client.user.id))
            if content != native.content:
                return None
            if current.reference is not None:
                reference = current.reference
                if reference.channel_id is None or reference.message_id is None:
                    return None
                target_channel = await asyncio.wait_for(
                    self._client.fetch_channel(reference.channel_id), timeout=10
                )
                if not isinstance(
                    target_channel,
                    (discord.TextChannel, discord.Thread, discord.DMChannel),
                ):
                    return None
                reference.resolved = await asyncio.wait_for(
                    target_channel.fetch_message(reference.message_id), timeout=10
                )
                if reference.resolved.attachments:
                    return None
            target = None
            if event.source.auto_thread_created:
                target = await asyncio.wait_for(
                    self._client.fetch_channel(int(event.source.chat_id)), timeout=10
                )
                if (
                    not isinstance(target, discord.Thread)
                    or str(target.parent_id) != origin
                ):
                    return None
            prepared = await self._prepare_inbound_event(
                current,
                role_authorized,
                recovered=True,
                cached=event,
                restored_channel=target,
                authorize=authorize,
            )
            if prepared is None or prepared.event.is_command():
                return None
            verified = prepared.event
            if (
                verified.source.chat_id,
                verified.source.thread_id,
                verified.source.scope_id,
            ) != (event.source.chat_id, event.source.thread_id, event.source.scope_id):
                return None
            if self._canonicalize(verified.source) is None:
                return None
            if authorize is not None and not authorize(verified.source):
                return None
            cache = get_hermes_home().resolve() / "cache"
            if event.media_urls and (
                not all(
                    Path(path).resolve().is_relative_to(cache)
                    for path in event.media_urls
                )
                or not native.attachments_available(event.media_urls)
            ):
                return None
            verified._pending_native_input = native
            verified.timestamp = event.timestamp
            return verified
        except (
            discord.HTTPException,
            discord.InvalidData,
            OSError,
            ValueError,
            asyncio.TimeoutError,
        ):
            return None
