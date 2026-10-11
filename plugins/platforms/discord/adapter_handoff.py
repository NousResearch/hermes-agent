"""Discord handoff creation and recipient membership verification."""
import logging
from typing import Optional

from agent.i18n import t

logger = logging.getLogger(__name__)


class DiscordHandoffMixin:
    async def _handoff_thread_with_recipient(self, thread, recipient_user_id: Optional[str]) -> Optional[str]:
        """Do not expose a recipient-scoped thread until Discord confirms membership."""
        from plugins.platforms.discord import adapter as discord_adapter

        if recipient_user_id is not None:
            try:
                user_id = int(recipient_user_id)
                await thread.add_user(discord_adapter._Snowflake(user_id))
                member = await thread.fetch_member(user_id)
                if member.id != user_id:
                    raise ValueError("Discord returned a different thread member")
            except Exception as exc:
                logger.warning(
                    "[%s] Handoff thread recipient membership not verified; "
                    "falling back to the configured channel: %s", self.name, exc, exc_info=True)
                return None
        return str(thread.id)

    async def create_handoff_thread(
        self, parent_chat_id: str, name: str, *, recipient_user_id: Optional[str] = None,
    ) -> Optional[str]:
        """Create a handoff under a text channel, optionally verifying the named recipient.

        Cron supplies persisted scheduling identity; legacy two-argument callers remain unchanged.
        A membership failure returns None without trying a different thread. Only creation failure
        uses the existing seed-message fallback; DMs cannot host threads.
        """
        from plugins.platforms.discord import adapter as discord_adapter

        if not self._client or not discord_adapter.DISCORD_AVAILABLE:
            return None
        try:
            parent_id = int(parent_chat_id)
        except (TypeError, ValueError):
            return None
        try:
            parent = self._client.get_channel(parent_id)
            if parent is None:
                parent = await self._client.fetch_channel(parent_id)
        except Exception as exc:
            logger.warning(
                "[%s] Handoff thread: cannot resolve parent %s: %s", self.name, parent_chat_id, exc, exc_info=True,
            )
            return None
        # DMs, voice channels, and existing threads can't host child threads.
        if isinstance(parent, getattr(discord_adapter.discord, "DMChannel", ())):
            logger.info(
                "[%s] Handoff thread: parent %s is a DM; threads not supported here",
                self.name, parent_chat_id,
            )
            return None
        thread_name = (name or "handoff").strip()[:80] or "handoff"
        reason = "Hermes session handoff"
        try:
            create = getattr(parent, "create_thread", None)
            if create is not None:
                thread = await create(name=thread_name, auto_archive_duration=1440, reason=reason)
                return await self._handoff_thread_with_recipient(thread, recipient_user_id)
        except Exception as direct_error:
            logger.debug(
                "[%s] Handoff thread: direct create failed (%s); trying seed-message fallback",
                self.name, direct_error, exc_info=True,
            )
        try:
            send = getattr(parent, "send", None)
            if send is None:
                return None
            seed_msg = await send(t("platform.discord.thread.handoff_seed", name=thread_name))
            thread = await seed_msg.create_thread(
                name=thread_name, auto_archive_duration=1440, reason=reason,
            )
            return await self._handoff_thread_with_recipient(thread, recipient_user_id)
        except Exception as fallback_error:
            logger.warning(
                "[%s] Handoff thread: both create paths failed for parent %s: %s",
                self.name, parent_chat_id, fallback_error, exc_info=True,
            )
            return None
