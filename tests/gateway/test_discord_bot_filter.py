"""Tests for Discord bot message filtering (DISCORD_ALLOW_BOTS)."""

import os
import unittest
from unittest.mock import MagicMock


def setUpModule():
    os.environ.pop("DISCORD_ALLOW_BOTS", None)
    os.environ.pop("DISCORD_BOT_MENTION_FALLBACK", None)


def _make_author(*, bot: bool = False, is_self: bool = False):
    """Create a mock Discord author."""
    author = MagicMock()
    author.bot = bot
    author.id = 99999 if is_self else 12345
    author.name = "TestBot" if bot else "TestUser"
    author.display_name = author.name
    return author


def _make_message(*, author=None, content="hello", mentions=None, raw_mentions=None, is_dm=False):
    """Create a mock Discord message."""
    msg = MagicMock()
    msg.author = author or _make_author()
    msg.content = content
    msg.attachments = []
    msg.mentions = mentions or []
    msg.raw_mentions = raw_mentions or []
    if is_dm:
        import discord
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

    def _run_filter(self, message, allow_bots="none", client_user=None, *, bot_mention_fallback=False):
        """Simulate the on_message filter logic and return whether message was accepted."""
        # Replicate the exact filter logic from discord.py on_message
        if message.author == client_user:
            return False  # own messages always ignored

        if getattr(message.author, "bot", False):
            allow = allow_bots.lower().strip()
            if allow == "none":
                return False
            elif allow == "mentions":
                from gateway.config import PlatformConfig
                from plugins.platforms.discord.adapter import DiscordAdapter

                adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
                adapter._client = MagicMock(user=client_user)
                if not adapter._message_mentions_self(message, allow_fallback=bot_mention_fallback):
                    return False
            # "all" falls through
        
        return True  # message accepted

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

    def test_allow_bots_none_rejects_bots(self):
        """With allow_bots=none, all other bot messages are rejected."""
        bot = _make_author(bot=True)
        msg = _make_message(author=bot)
        self.assertFalse(self._run_filter(msg, "none"))

    def test_allow_bots_all_accepts_bots(self):
        """With allow_bots=all, all bot messages are accepted."""
        bot = _make_author(bot=True)
        msg = _make_message(author=bot)
        self.assertTrue(self._run_filter(msg, "all"))

    def test_allow_bots_mentions_rejects_without_mention(self):
        """With allow_bots=mentions, bot messages without @mention are rejected."""
        our_user = _make_author(is_self=True)
        bot = _make_author(bot=True)
        msg = _make_message(author=bot, mentions=[])
        self.assertFalse(self._run_filter(msg, "mentions", our_user))

    def test_allow_bots_mentions_accepts_with_mention(self):
        """With allow_bots=mentions, bot messages with @mention are accepted."""
        our_user = _make_author(is_self=True)
        bot = _make_author(bot=True)
        msg = _make_message(author=bot, mentions=[our_user])
        self.assertTrue(self._run_filter(msg, "mentions", our_user))

    def test_allow_bots_mentions_accepts_raw_mentions_without_resolved_mentions(self):
        """Bot-authored messages can lack resolved mentions but still include raw IDs."""
        our_user = _make_author(is_self=True)
        bot = _make_author(bot=True)
        msg = _make_message(author=bot, mentions=[], raw_mentions=[our_user.id])
        self.assertFalse(self._run_filter(msg, "mentions", our_user))
        self.assertTrue(self._run_filter(msg, "mentions", our_user, bot_mention_fallback=True))

    def test_allow_bots_mentions_accepts_literal_mention_without_resolved_mentions(self):
        """Bot-authored messages can preserve literal <@id> text even if mentions is empty."""
        our_user = _make_author(is_self=True)
        bot = _make_author(bot=True)
        msg = _make_message(author=bot, content=f"ping <@{our_user.id}>", mentions=[], raw_mentions=[])
        self.assertFalse(self._run_filter(msg, "mentions", our_user))
        self.assertTrue(self._run_filter(msg, "mentions", our_user, bot_mention_fallback=True))

    def test_allow_bots_mentions_accepts_nickname_literal_mention(self):
        """Discord nickname mention syntax (<@!id>) should also count."""
        our_user = _make_author(is_self=True)
        bot = _make_author(bot=True)
        msg = _make_message(author=bot, content=f"ping <@!{our_user.id}>", mentions=[], raw_mentions=[])
        self.assertFalse(self._run_filter(msg, "mentions", our_user))
        self.assertTrue(self._run_filter(msg, "mentions", our_user, bot_mention_fallback=True))

    def test_allow_bots_mentions_rejects_literal_mention_for_another_bot(self):
        """Literal mention fallback must still require this bot's ID."""
        our_user = _make_author(is_self=True)
        bot = _make_author(bot=True)
        msg = _make_message(author=bot, content="ping <@111111>", mentions=[], raw_mentions=[111111])
        self.assertFalse(self._run_filter(msg, "mentions", our_user, bot_mention_fallback=True))

    def test_discord_bot_mention_fallback_defaults_false(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.discord.adapter import DiscordAdapter

        adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
        self.assertFalse(adapter._discord_bot_mention_fallback())

    def test_discord_bot_mention_fallback_reads_platform_config(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.discord.adapter import DiscordAdapter

        adapter = DiscordAdapter(
            PlatformConfig(
                enabled=True,
                token="test-token",
                extra={"bot_mention_fallback": True},
            )
        )
        self.assertTrue(adapter._discord_bot_mention_fallback())

    def test_discord_bot_mention_fallback_reads_env_when_config_unset(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.discord.adapter import DiscordAdapter

        old = os.environ.get("DISCORD_BOT_MENTION_FALLBACK")
        os.environ["DISCORD_BOT_MENTION_FALLBACK"] = "true"
        try:
            adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
            self.assertTrue(adapter._discord_bot_mention_fallback())
        finally:
            if old is None:
                os.environ.pop("DISCORD_BOT_MENTION_FALLBACK", None)
            else:
                os.environ["DISCORD_BOT_MENTION_FALLBACK"] = old

    def test_default_is_none(self):
        """Default behavior (no env var) should be 'none'."""
        default = os.getenv("DISCORD_ALLOW_BOTS", "none")
        self.assertEqual(default, "none")

    def test_case_insensitive(self):
        """Allow_bots value should be case-insensitive."""
        bot = _make_author(bot=True)
        msg = _make_message(author=bot)
        self.assertTrue(self._run_filter(msg, "ALL"))
        self.assertTrue(self._run_filter(msg, "All"))
        self.assertFalse(self._run_filter(msg, "NONE"))
        self.assertFalse(self._run_filter(msg, "None"))


if __name__ == "__main__":
    unittest.main()
