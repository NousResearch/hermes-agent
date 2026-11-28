"""Discord processing reactions: 👀 while a message is handled, then ✅ or ❌ by its outcome."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from gateway.platforms.event import MessageEvent, ProcessingOutcome

logger = logging.getLogger(__name__)


class DiscordReactionsMixin:
    name: str
    _client: Any
    _extra_or_env_flag: Any
    _resolve_channel: Any
    _record_discord_processing_start: Any
    _record_discord_processing_complete: Any

    async def _add_reaction(self, message: Any, emoji: str) -> bool:
        """Add an emoji reaction to a Discord message."""
        if not message or not hasattr(message, "add_reaction"):
            return False
        try:
            await message.add_reaction(emoji)
            return True
        except Exception as e:
            logger.debug("[%s] add_reaction failed (%s): %s", self.name, emoji, e)
            return False

    async def _remove_reaction(self, message: Any, emoji: str) -> bool:
        """Remove the bot's own emoji reaction from a Discord message."""
        if not message or not hasattr(message, "remove_reaction") or not self._client or not self._client.user:
            return False
        try:
            await message.remove_reaction(emoji, self._client.user)
            return True
        except Exception as e:
            logger.debug("[%s] remove_reaction failed (%s): %s", self.name, emoji, e)
            return False

    def _reactions_enabled(self) -> bool:
        """Reactions enabled via ``extra.reactions`` (YAML, per profile) or ``DISCORD_REACTIONS``."""
        return self._extra_or_env_flag("reactions", "DISCORD_REACTIONS", "true", truthy=False)

    async def _reaction_target(self, event: MessageEvent) -> Any:
        """The message to react to: the live one, or a partial message built from the event's ids when
        it has none (a queued message restored from the session queue), best effort."""
        message = event.raw_message
        if message is not None or not self._client:
            return message
        source = event.source
        # An auto-thread's starter message lives in the parent channel, not in the thread.
        channel_id = source.parent_chat_id if source.auto_thread_created else source.chat_id
        try:
            channel = await self._resolve_channel(channel_id)
            return channel.get_partial_message(int(event.message_id))
        except Exception:
            logger.debug("[%s] No reaction target for message %s", self.name, event.message_id, exc_info=True)
            return None

    async def on_processing_start(self, event: MessageEvent) -> None:
        """Record whether this event received an in-progress reaction.

        Args:
            event: Incoming message being processed.

        Returns:
            None.
        """
        acked = False
        if self._reactions_enabled():
            message = await self._reaction_target(event)
            if hasattr(message, "add_reaction"):
                acked = await self._add_reaction(message, "👀")
        event._discord_eyes_sent = acked
        await asyncio.to_thread(self._record_discord_processing_start, event, emoji_ack=acked)

    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        """Clear a sent in-progress reaction and add the enabled outcome reaction.

        Args:
            event: Incoming message whose processing finished.
            outcome: Result of processing the message.

        Returns:
            None.
        """
        await asyncio.to_thread(self._record_discord_processing_complete, event, outcome)
        reactions_enabled = self._reactions_enabled()
        eyes_sent = getattr(event, "_discord_eyes_sent", False)
        if not reactions_enabled and not eyes_sent:
            return
        message = await self._reaction_target(event)
        if hasattr(message, "add_reaction"):
            if eyes_sent or reactions_enabled:
                await self._remove_reaction(message, "👀")
            if reactions_enabled and outcome == ProcessingOutcome.SUCCESS:
                await self._add_reaction(message, "✅")
            elif reactions_enabled and outcome == ProcessingOutcome.FAILURE:
                await self._add_reaction(message, "❌")
