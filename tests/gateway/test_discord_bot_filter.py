"""Tests for Discord bot message filtering (DISCORD_ALLOW_BOTS)."""

import os
import re
import unittest
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from plugins.platforms.discord.adapter import DiscordAdapter


def _make_author(*, bot: bool = False, is_self: bool = False):
    """Create a mock Discord author."""
    author = MagicMock()
    author.bot = bot
    author.id = 99999 if is_self else 12345
    author.name = "TestBot" if bot else "TestUser"
    author.display_name = author.name
    return author


def _make_message(*, author=None, content="hello", mentions=None, is_dm=False):
    """Create a mock Discord message."""
    msg = MagicMock()
    msg.author = author or _make_author()
    msg.content = content
    msg.attachments = []
    msg.mentions = mentions or []
    msg.id = 777
    msg.type = discord.MessageType.default
    if is_dm:
        msg.channel = MagicMock(spec=discord.DMChannel)
        msg.channel.id = 111
    else:
        msg.channel = MagicMock()
        msg.channel.id = 222
        msg.channel.name = "test-channel"
        msg.channel.guild = MagicMock()
        msg.channel.guild.name = "TestServer"
        # Make isinstance checks fail for DMChannel and Thread
        type(msg.channel).__name__ = "TextChannel"
    return msg


class TestDiscordBotFilter(unittest.TestCase):
    """Test the DISCORD_ALLOW_BOTS filtering logic."""

    @staticmethod
    def _self_is_explicitly_mentioned(message, client_user):
        """Mirror adapter._self_is_explicitly_mentioned: resolved or raw mention."""
        if not client_user:
            return False
        if client_user in message.mentions:
            return True
        raw_ids = {
            m.group(1)
            for m in re.finditer(r"<@!?(\d+)>", getattr(message, "content", "") or "")
        }
        return str(client_user.id) in raw_ids

    @staticmethod
    def _self_is_raw_mentioned(message, client_user):
        """Mirror adapter._self_is_raw_mentioned: raw inline token only."""
        if not client_user:
            return False
        raw_ids = {
            m.group(1)
            for m in re.finditer(r"<@!?(\d+)>", getattr(message, "content", "") or "")
        }
        return str(client_user.id) in raw_ids

    def _run_filter(
        self,
        message,
        allow_bots="none",
        client_user=None,
        bots_require_inline_mention=False,
    ):
        """Simulate the on_message filter logic and return whether message was accepted."""
        # Replicate the exact filter logic from discord.py on_message
        if message.author == client_user:
            return False  # own messages always ignored

        if getattr(message.author, "bot", False):
            allow = allow_bots.lower().strip()
            if allow == "none":
                return False
            elif allow == "mentions":
                if not self._self_is_explicitly_mentioned(message, client_user):
                    return False
            if (
                bots_require_inline_mention
                and not self._self_is_raw_mentioned(message, client_user)
            ):
                return False
            # "all" falls through
        
        return True  # message accepted

    @staticmethod
    def _run_real_admission(message, allow_bots, client_user):
        adapter = object.__new__(DiscordAdapter)
        adapter._client = MagicMock(user=client_user)
        adapter._dedup = MagicMock()
        adapter._dedup.contains.return_value = False
        adapter._get_allow_bots = lambda: allow_bots
        adapter._discord_bots_require_inline_mention = lambda: False
        return adapter._discord_message_admission(message, claim=False)[0]

    def test_own_messages_always_ignored(self):
        """Bot's own messages are always ignored regardless of allow_bots."""
        bot_user = _make_author(is_self=True)
        msg = _make_message(author=bot_user)
        self.assertFalse(self._run_filter(msg, "all", bot_user))

    def test_human_messages_always_accepted(self):
        """Human messages are always accepted regardless of allow_bots."""
        human = _make_author(bot=False)
        msg = _make_message(author=human)
        self.assertTrue(self._run_filter(msg, "none"))
        self.assertTrue(self._run_filter(msg, "mentions"))
        self.assertTrue(self._run_filter(msg, "all"))


    def test_allow_bots_mentions_rejects_without_mention(self):
        """With allow_bots=mentions, bot messages without @mention are rejected."""
        our_user = _make_author(is_self=True)
        bot = _make_author(bot=True)
        msg = _make_message(author=bot, mentions=[])
        self.assertFalse(self._run_filter(msg, "mentions", our_user))


    def test_inline_mention_requirement_accepts_body_mention(self):
        """Opt-in guard still admits intentional inline cross-bot mentions."""
        our_user = _make_author(is_self=True)
        bot = _make_author(bot=True)
        msg = _make_message(
            author=bot,
            content=f"<@{our_user.id}> intentional handoff",
            mentions=[our_user],
        )

        self.assertTrue(
            self._run_filter(
                msg,
                "all",
                our_user,
                bots_require_inline_mention=True,
            )
        )

    def test_hook_mentions_requires_literal_raw_self_mention(self):
        our_user = _make_author(is_self=True)
        bot = _make_author(bot=True)
        reply_ping_only = _make_message(author=bot, mentions=[our_user])
        literal_mention = _make_message(
            author=bot,
            content=f"<@{our_user.id}> signed request",
            mentions=[our_user],
        )

        self.assertFalse(self._run_real_admission(reply_ping_only, "hook_mentions", our_user))
        self.assertTrue(self._run_real_admission(literal_mention, "hook_mentions", our_user))


    def test_default_is_none(self):
        """Default behavior (no env var) should be 'none'."""
        default = os.getenv("DISCORD_ALLOW_BOTS", "none")
        self.assertEqual(default, "none")


@pytest.mark.asyncio
async def test_hook_mentions_refusal_precedes_discord_side_effects():
    our_user = _make_author(is_self=True)
    bot = _make_author(bot=True)
    message = _make_message(
        author=bot, content=f"<@{our_user.id}> signed request", mentions=[our_user],
    )
    original_content = message.content
    adapter = object.__new__(DiscordAdapter)
    adapter._client = MagicMock(user=our_user)
    adapter._voice_text_channels = {}
    adapter._get_parent_channel_id = MagicMock(return_value=None)
    adapter._discord_channel_keys = MagicMock(return_value={"222"})
    adapter._get_allowed_channels = MagicMock(return_value=set())
    adapter._get_ignored_channels = MagicMock(return_value=set())
    adapter._discord_free_response_channels = MagicMock(return_value=set())
    adapter._discord_require_mention = MagicMock(return_value=True)
    adapter._in_bot_thread = MagicMock(return_value=False)
    adapter._pre_admit_hook_mentions_bot = MagicMock(return_value=(True, None))
    adapter._auto_create_thread = AsyncMock()
    adapter._collect_attachment_media = AsyncMock()
    adapter._fetch_channel_context = AsyncMock()

    assert await adapter._handle_message(message) is False
    assert message.content == original_content
    adapter._auto_create_thread.assert_not_awaited()
    adapter._collect_attachment_media.assert_not_awaited()
    adapter._fetch_channel_context.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
