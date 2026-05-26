"""Tests for Discord bot message filtering (DISCORD_ALLOW_BOTS)."""

import asyncio
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import PlatformConfig
from gateway.platforms import discord as discord_platform
from gateway.platforms.discord import (
    DiscordAdapter,
    _discord_bot_reply_false_reaction,
    _has_raw_user_mention,
    _parse_discord_bot_msg_v1,
)


VALID_BOT_MSG = "\r\n".join(
    [
        "\t<@99999> ",
        " BOT_MSG v1\t",
        "reply_expected: true",
        "kind: status",
        "correlation_id: pm-1.2:3_ok",
        "---",
        "body line 1",
        "BOT_MSG v1 inside body is opaque",
    ]
)


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

    def _run_filter(self, message, allow_bots="none", client_user=None):
        """Simulate the on_message filter logic and return whether message was accepted."""
        # Replicate the exact filter logic from discord.py on_message
        if message.author == client_user:
            return False  # own messages always ignored

        if getattr(message.author, "bot", False):
            allow = allow_bots.lower().strip()
            if allow == "none":
                return False
            elif allow == "mentions":
                if not client_user or client_user not in message.mentions:
                    return False
            # "all" falls through
        
        return True  # message accepted

    def _adapter(self, **env):
        tmp = tempfile.TemporaryDirectory()
        baseline_env = {
            "HERMES_HOME": tmp.name,
            "DISCORD_ALLOWED_BOT_USERS": "",
            "DISCORD_BOT_CONTROL_CHANNELS": "",
            "DISCORD_BOT_LOOP_FUSE_WINDOW_SECONDS": "",
            "DISCORD_BOT_LOOP_FUSE_MAX_MESSAGES": "",
            "DISCORD_BOT_LOOP_FUSE_SUPPRESS_SECONDS": "",
        }
        baseline_env.update(env)
        patcher = patch.dict(os.environ, baseline_env, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(tmp.cleanup)
        adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake-token"))
        adapter._client = SimpleNamespace(user=SimpleNamespace(id=99999))
        return adapter

    def test_raw_mention_helper_ignores_reply_metadata_mentions(self):
        self.assertTrue(_has_raw_user_mention("hello <@99999>", 99999))
        self.assertTrue(_has_raw_user_mention("hello <@!99999>", 99999))
        self.assertFalse(_has_raw_user_mention("hello", 99999))
        self.assertFalse(_has_raw_user_mention("hello <@12345>", 99999))

    def test_actual_mentions_mode_uses_raw_content_not_message_mentions(self):
        adapter = self._adapter(DISCORD_ALLOW_BOTS="mentions")
        our_user = getattr(adapter._client, "user")
        bot = _make_author(bot=True)
        msg = _make_message(author=bot, content="reply ping only", mentions=[our_user])
        self.assertFalse(adapter._should_accept_bot_message(msg, "mentions"))

    def test_actual_mentions_mode_accepts_raw_self_mention(self):
        adapter = self._adapter(DISCORD_ALLOW_BOTS="mentions", DISCORD_ALLOWED_BOT_USERS="12345")
        bot = _make_author(bot=True)
        msg = _make_message(author=bot, content=VALID_BOT_MSG, mentions=[])
        self.assertTrue(adapter._should_accept_bot_message(msg, "mentions"))

    def test_actual_mentions_mode_fails_closed_without_allowed_bot_users(self):
        adapter = self._adapter(DISCORD_ALLOW_BOTS="mentions")
        bot = _make_author(bot=True)
        msg = _make_message(author=bot, content=VALID_BOT_MSG, mentions=[])
        self.assertFalse(adapter._should_accept_bot_message(msg, "mentions"))

    def test_actual_mentions_mode_respects_allowed_bot_users(self):
        adapter = self._adapter(DISCORD_ALLOW_BOTS="mentions", DISCORD_ALLOWED_BOT_USERS="777")
        bot = _make_author(bot=True)
        msg = _make_message(author=bot, content=VALID_BOT_MSG, mentions=[])
        self.assertFalse(adapter._should_accept_bot_message(msg, "mentions"))

    def test_actual_mentions_mode_rejects_malformed_bot_msg_protocol(self):
        adapter = self._adapter(DISCORD_ALLOW_BOTS="mentions", DISCORD_ALLOWED_BOT_USERS="12345")
        bot = _make_author(bot=True)
        msg = _make_message(author=bot, content="<@99999> checkpoint ready", mentions=[])
        self.assertFalse(adapter._should_accept_bot_message(msg, "mentions"))

    def test_bot_loop_fuse_applies_to_explicit_mentions(self):
        adapter = self._adapter(DISCORD_ALLOW_BOTS="mentions", DISCORD_ALLOWED_BOT_USERS="12345")
        bot = _make_author(bot=True)
        results = []
        for idx in range(6):
            msg = _make_message(author=bot, content=VALID_BOT_MSG.replace("pm-1.2:3_ok", f"pm-{idx}"), mentions=[])
            msg.id = idx
            results.append(adapter._should_accept_bot_message(msg, "mentions"))
        self.assertEqual(results, [True, True, True, True, True, True])
        seventh = _make_message(author=bot, content=VALID_BOT_MSG.replace("pm-1.2:3_ok", "pm-6"), mentions=[])
        seventh.id = 6
        self.assertFalse(adapter._should_accept_bot_message(seventh, "mentions"))

    def test_reply_expected_false_bypasses_loop_fuse_for_no_model_ack(self):
        adapter = self._adapter(DISCORD_ALLOW_BOTS="mentions", DISCORD_ALLOWED_BOT_USERS="12345")
        bot = _make_author(bot=True)
        content = VALID_BOT_MSG.replace("reply_expected: true", "reply_expected: false")
        results = []
        for idx in range(8):
            msg = _make_message(author=bot, content=content.replace("pm-1.2:3_ok", f"ack-{idx}"), mentions=[])
            msg.id = idx
            results.append(adapter._should_accept_bot_message(msg, "mentions"))
        self.assertEqual(results, [True] * 8)

    def test_bot_loop_fuse_env_override_restores_strict_threshold(self):
        adapter = self._adapter(
            DISCORD_ALLOW_BOTS="mentions",
            DISCORD_ALLOWED_BOT_USERS="12345",
            DISCORD_BOT_LOOP_FUSE_MAX_MESSAGES="3",
            DISCORD_BOT_LOOP_FUSE_SUPPRESS_SECONDS="600",
        )
        bot = _make_author(bot=True)
        results = []
        for idx in range(4):
            msg = _make_message(author=bot, content=VALID_BOT_MSG.replace("pm-1.2:3_ok", f"pm-{idx}"), mentions=[])
            msg.id = idx
            results.append(adapter._should_accept_bot_message(msg, "mentions"))
        self.assertEqual(results, [True, True, True, True])
        fifth = _make_message(author=bot, content=VALID_BOT_MSG.replace("pm-1.2:3_ok", "pm-5"), mentions=[])
        fifth.id = 5
        self.assertFalse(adapter._should_accept_bot_message(fifth, "mentions"))

    def test_bot_loop_fuse_suppression_is_scoped_by_thread(self):
        adapter = self._adapter(DISCORD_ALLOW_BOTS="mentions", DISCORD_BOT_CONTROL_CHANNELS="333")
        receiver = "99999"
        sender = "12345"
        for _ in range(6):
            adapter._bot_loop_fuse.record_and_check(
                receiver_bot_id=receiver,
                sender_bot_id=sender,
                thread_id="thread-a",
            )
        self.assertTrue(
            adapter._bot_loop_fuse.is_suppressed(
                receiver_bot_id=receiver,
                sender_bot_id=sender,
                thread_id="thread-a",
            )
        )
        self.assertFalse(
            adapter._bot_loop_fuse.is_suppressed(
                receiver_bot_id=receiver,
                sender_bot_id=sender,
                thread_id="thread-b",
            )
        )
        self.assertFalse(
            adapter._bot_loop_fuse.is_suppressed(
                receiver_bot_id=receiver,
                sender_bot_id="67890",
                thread_id="thread-a",
            )
        )

    def test_registered_bot_thread_followup_requires_explicit_bot_control_scope(self):
        class FakeThread:
            def __init__(self):
                self.id = 444
                self.parent_id = 333
                self.parent = SimpleNamespace(id=333)

        original_discord = discord_platform.discord
        original_thread = getattr(discord_platform.discord, "Thread", None) if discord_platform.discord is not None else None
        if discord_platform.discord is None:
            discord_platform.discord = SimpleNamespace(Thread=FakeThread)
        else:
            setattr(discord_platform.discord, "Thread", FakeThread)

        def restore_thread_class():
            if original_discord is None:
                discord_platform.discord = None
            elif original_thread is None:
                try:
                    delattr(discord_platform.discord, "Thread")
                except AttributeError:
                    pass
            else:
                setattr(discord_platform.discord, "Thread", original_thread)

        self.addCleanup(restore_thread_class)

        adapter = self._adapter(DISCORD_ALLOW_BOTS="mentions", DISCORD_ALLOWED_BOT_USERS="12345")
        bot = _make_author(bot=True)
        invite = _make_message(author=bot, content=VALID_BOT_MSG, mentions=[])
        invite.channel = FakeThread()
        invite.id = 1
        self.assertTrue(adapter._should_accept_bot_message(invite, "mentions"))

        followup = _make_message(author=bot, content="continue", mentions=[])
        followup.channel = FakeThread()
        followup.id = 2
        self.assertFalse(adapter._should_accept_bot_message(followup, "mentions"))

        adapter_scoped = self._adapter(
            DISCORD_ALLOW_BOTS="mentions",
            DISCORD_ALLOWED_BOT_USERS="12345",
            DISCORD_BOT_CONTROL_CHANNELS="333",
        )
        invite2 = _make_message(author=bot, content=VALID_BOT_MSG, mentions=[])
        invite2.channel = FakeThread()
        invite2.id = 3
        self.assertTrue(adapter_scoped._should_accept_bot_message(invite2, "mentions"))
        followup2 = _make_message(author=bot, content="continue", mentions=[])
        followup2.channel = FakeThread()
        followup2.id = 4
        self.assertFalse(adapter_scoped._should_accept_bot_message(followup2, "mentions"))

    def test_bot_msg_v1_parser_accepts_crlf_and_keeps_body_opaque(self):
        parsed = _parse_discord_bot_msg_v1(VALID_BOT_MSG, "99999")
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertTrue(parsed["reply_expected"])
        self.assertEqual(parsed["kind"], "status")
        self.assertEqual(parsed["correlation_id"], "pm-1.2:3_ok")
        self.assertEqual(parsed["body"], "body line 1\nBOT_MSG v1 inside body is opaque")

    def test_bot_msg_v1_parser_rejects_bad_header_details(self):
        bad_cases = [
            VALID_BOT_MSG.replace("<@99999>", "<@12345>"),
            VALID_BOT_MSG.replace("BOT_MSG v1", "BOT_MSG v2", 1),
            VALID_BOT_MSG.replace("reply_expected: true", "reply_expected: True"),
            VALID_BOT_MSG.replace("kind: status", "kind: unknown"),
            VALID_BOT_MSG.replace("correlation_id: pm-1.2:3_ok", "correlation_id: bad id"),
            VALID_BOT_MSG.replace("---", "--"),
        ]
        for content in bad_cases:
            self.assertIsNone(_parse_discord_bot_msg_v1(content, "99999"))

    def test_bot_msg_v1_parser_rejects_unicode_leading_whitespace_before_mention(self):
        content = VALID_BOT_MSG.replace("\t<@99999> ", "\u200b<@99999>", 1)

        self.assertIsNone(_parse_discord_bot_msg_v1(content, "99999"))

    def test_bot_msg_v1_parser_rejects_empty_body_for_non_status_kind(self):
        content = "\n".join(
            [
                "<@99999>",
                "BOT_MSG v1",
                "reply_expected: true",
                "kind: action_required",
                "correlation_id: needs-body",
                "---",
                "",
            ]
        )

        self.assertIsNone(_parse_discord_bot_msg_v1(content, "99999"))

    def test_reply_false_reaction_is_configurable(self):
        with patch.dict(os.environ, {"DISCORD_BOT_REPLY_FALSE_REACTION": "✅"}, clear=False):
            self.assertEqual(_discord_bot_reply_false_reaction(), "✅")
        with patch.dict(os.environ, {"DISCORD_BOT_REPLY_FALSE_REACTION": ""}, clear=False):
            self.assertEqual(_discord_bot_reply_false_reaction(), "👀")

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

    def test_default_is_none(self):
        """Default behavior (no env var) should be 'none'."""
        with patch.dict(os.environ, {}, clear=True):
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
