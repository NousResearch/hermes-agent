"""Tests for Discord bot message filtering (DISCORD_ALLOW_BOTS)."""

import os
import re
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from plugins.platforms.discord.adapter import DiscordAdapter, _is_exact_heartbeat_message


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
    msg.clean_content = content
    msg.attachments = []
    msg.mentions = mentions or []
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


class TestDiscordBotFilter(unittest.IsolatedAsyncioTestCase):
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

        content = getattr(message, "clean_content", message.content) or ""
        if getattr(message.author, "bot", False) and _is_exact_heartbeat_message(content):
            return False

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

    def test_allow_bots_mentions_accepts_with_mention(self):
        """With allow_bots=mentions, bot messages with @mention are accepted."""
        our_user = _make_author(is_self=True)
        bot = _make_author(bot=True)
        msg = _make_message(author=bot, mentions=[our_user])
        self.assertTrue(self._run_filter(msg, "mentions", our_user))

    def test_allow_bots_mentions_accepts_with_raw_content_mention(self):
        """Raw <@!ID> mention counts even when message.mentions is empty."""
        our_user = _make_author(is_self=True)
        bot = _make_author(bot=True)
        msg = _make_message(author=bot, content=f"<@!{our_user.id}> relay", mentions=[])
        self.assertTrue(self._run_filter(msg, "mentions", our_user))

    def test_heartbeat_ok_bot_message_is_nonconversational(self):
        """Exact bot HEARTBEAT_OK pings must not trigger another bot."""
        our_user = _make_author(is_self=True)
        bot = _make_author(bot=True)
        msg = _make_message(author=bot, content="HEARTBEAT_OK", mentions=[our_user])
        self.assertTrue(_is_exact_heartbeat_message("HEARTBEAT_OK"))
        self.assertFalse(self._run_filter(msg, "mentions", our_user))

    def test_operational_bot_handoff_still_passes_with_mention(self):
        """Non-heartbeat Molly/Petra handoffs still pass the bot mention path."""
        our_user = _make_author(is_self=True)
        bot = _make_author(bot=True)
        msg = _make_message(
            author=bot,
            content="Owner: Molly\nAsk: Petra validate project access\nEvidence: project visible",
            mentions=[our_user],
        )
        self.assertFalse(_is_exact_heartbeat_message(msg.content))
        self.assertTrue(self._run_filter(msg, "mentions", our_user))

    async def test_real_dispatch_drops_and_persists_exact_heartbeat(self):
        adapter = object.__new__(DiscordAdapter)
        adapter._ready_event = MagicMock()
        adapter._ready_event.is_set.return_value = True
        adapter._nonconversational_messages = MagicMock()
        adapter._discord_message_admission = MagicMock(return_value=(True, False))
        adapter._handle_message = AsyncMock(return_value=True)
        msg = _make_message(author=_make_author(bot=True), content="HEARTBEAT_OK")
        msg.id = 4242

        assert await adapter._dispatch_discord_message(msg) is False
        adapter._nonconversational_messages.mark_many.assert_called_once_with(["4242"])
        adapter._discord_message_admission.assert_not_called()
        adapter._handle_message.assert_not_awaited()

    async def test_real_dispatch_hands_off_nonheartbeat_bot_message(self):
        adapter = object.__new__(DiscordAdapter)
        adapter._ready_event = MagicMock()
        adapter._ready_event.is_set.return_value = True
        adapter._nonconversational_messages = MagicMock()
        adapter._discord_message_admission = MagicMock(return_value=(True, False))
        adapter._handle_message = AsyncMock(return_value=True)
        msg = _make_message(author=_make_author(bot=True), content="Petra validate access")

        assert await adapter._dispatch_discord_message(msg) is True
        adapter._discord_message_admission.assert_called_once_with(msg, claim=True)
        adapter._handle_message.assert_awaited_once_with(msg, role_authorized=False)

    def test_inline_mention_requirement_off_preserves_reply_ping_behavior(self):
        """Default behavior: resolved reply-ping mentions still admit bot messages."""
        our_user = _make_author(is_self=True)
        bot = _make_author(bot=True)
        msg = _make_message(author=bot, content="reply-ping only", mentions=[our_user])

        self.assertTrue(
            self._run_filter(
                msg,
                "all",
                our_user,
                bots_require_inline_mention=False,
            )
        )

    def test_inline_mention_requirement_rejects_reply_ping_only(self):
        """Opt-in guard rejects bot messages where only Discord's reply-ping mentions us."""
        our_user = _make_author(is_self=True)
        bot = _make_author(bot=True)
        msg = _make_message(author=bot, content="reply-ping only", mentions=[our_user])

        self.assertFalse(
            self._run_filter(
                msg,
                "all",
                our_user,
                bots_require_inline_mention=True,
            )
        )

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


    def test_default_is_none(self):
        """Default behavior (no env var) should be 'none'."""
        with patch.dict(os.environ, {}, clear=True):
            default = os.getenv("DISCORD_ALLOW_BOTS", "none")
            self.assertEqual(default, "none")


if __name__ == "__main__":
    unittest.main()
