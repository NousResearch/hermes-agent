from __future__ import annotations

from typing import Any, Callable

try:
    import discord
except ImportError:
    discord = None

from gateway.platforms.helpers import ThreadParticipationTracker


class DiscordThreadMentionsMixin:
    """Mention-free thread ownership and shared-human transition policy."""

    config: Any
    _gate_env: Callable[..., str]
    _extra_or_env_flag: Callable[..., bool]
    _threads: ThreadParticipationTracker
    _shared_human_threads: ThreadParticipationTracker
    _mention_free_thread_owners: dict[str, ThreadParticipationTracker]

    def _init_discord_thread_mentions(self) -> None:
        # Threads the bot participated in (no @mention needed there); persisted across restarts.
        self._threads = ThreadParticipationTracker("discord")
        # A thread becomes persistently mention-gated once a human other than its owner participates.
        self._shared_human_threads = ThreadParticipationTracker(
            "discord-shared-human",
            max_tracked=10_000,
        )
        # Separate persistent trackers keep configured owners isolated from one another.
        self._mention_free_thread_owners: dict[str, ThreadParticipationTracker] = {}

    def _discord_thread_require_mention(self) -> bool:
        """Whether bot-participated threads still require an explicit mention."""
        return self._extra_or_env_flag(
            "thread_require_mention",
            "DISCORD_THREAD_REQUIRE_MENTION",
            "false",
            truthy=True,
        )

    def _discord_thread_mention_free_users(self) -> set[str]:
        """Return human user IDs allowed mention-free one-on-one threads."""
        configured = self.config.extra.get("thread_mention_free_users")
        if configured is None:
            configured = self._gate_env("DISCORD_THREAD_MENTION_FREE_USERS")
        if isinstance(configured, (list, tuple, set)):
            values = {str(value).strip() for value in configured if str(value).strip()}
        else:
            values = {
                value.strip()
                for value in str(configured or "").split(",")
                if value.strip()
            }
        return {value for value in values if value.isdigit()}

    def _discord_thread_owner_tracker(
        self,
        user_id: str,
    ) -> ThreadParticipationTracker:
        tracker = self._mention_free_thread_owners.get(user_id)
        if tracker is None:
            tracker = ThreadParticipationTracker(
                f"discord-mention-free-owner-{user_id}",
                max_tracked=10_000,
            )
            self._mention_free_thread_owners[user_id] = tracker
        return tracker

    def _discord_thread_owner(
        self,
        thread_id: str,
        mention_free_users: set[str],
    ) -> str | None:
        for user_id in mention_free_users:
            if thread_id in self._discord_thread_owner_tracker(user_id):
                return user_id
        return None

    async def _claim_discord_thread_owner(
        self, thread_id: str | None, author: Any
    ) -> None:
        """Persist the configured human who started a thread's ambient phase."""
        if not thread_id or author is None or getattr(author, "bot", False):
            return
        author_id = str(getattr(author, "id", ""))
        mention_free_users = self._discord_thread_mention_free_users()
        if (
            author_id in mention_free_users
            and self._discord_thread_owner(thread_id, mention_free_users) is None
        ):
            await self._discord_thread_owner_tracker(author_id).mark_async(thread_id)

    async def _observe_thread_human_participant(self, message: Any) -> None:
        """Persistently mark a configured one-on-one thread as shared."""
        mention_free_users = self._discord_thread_mention_free_users()
        if (
            not mention_free_users
            or discord is None
            or not isinstance(message.channel, discord.Thread)
        ):
            return
        thread_id = str(message.channel.id)
        if thread_id not in self._threads:
            return
        author = getattr(message, "author", None)
        if author is None or getattr(author, "bot", False):
            return
        author_id = str(getattr(author, "id", ""))
        owner_id = self._discord_thread_owner(thread_id, mention_free_users)
        if owner_id is None and author_id in mention_free_users:
            await self._discord_thread_owner_tracker(author_id).mark_async(thread_id)
        elif author_id != owner_id:
            await self._shared_human_threads.mark_async(thread_id)

    def _discord_bot_thread_is_mention_free(
        self,
        message: Any,
        thread_id: str | None,
    ) -> bool:
        """Return whether prior bot participation waives this message's mention."""
        if (
            not thread_id
            or thread_id not in self._threads
            or self._discord_thread_require_mention()
        ):
            return False

        mention_free_users = self._discord_thread_mention_free_users()
        if not mention_free_users:
            return True
        # Some lightweight integrations construct an adapter without running __init__.
        if not hasattr(self, "_mention_free_thread_owners"):
            return True

        author_id = str(getattr(getattr(message, "author", None), "id", ""))
        owner_id = self._discord_thread_owner(thread_id, mention_free_users)
        return author_id == owner_id and thread_id not in self._shared_human_threads

    def _in_bot_thread(self, message: Any) -> bool:
        """Whether prior bot participation waives this thread message's mention."""
        if discord is None or not isinstance(message.channel, discord.Thread):
            return False
        return self._discord_bot_thread_is_mention_free(
            message,
            str(message.channel.id),
        )
