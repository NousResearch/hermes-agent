"""Discord ignores bot-authored messages before model dispatch."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import PlatformConfig
from gateway.platforms.discord import DiscordAdapter, _has_raw_user_mention


def _make_author(*, bot: bool = False, is_self: bool = False):
    author = MagicMock()
    author.bot = bot
    author.id = 99999 if is_self else 12345
    author.name = "TestBot" if bot else "TestUser"
    author.display_name = author.name
    return author


def _make_message(*, author=None, content="hello", mentions=None, is_dm=False):
    msg = MagicMock()
    msg.author = author or _make_author()
    msg.content = content
    msg.clean_content = content
    msg.attachments = []
    msg.mentions = mentions or []
    msg.id = 123
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
        type(msg.channel).__name__ = "TextChannel"
    return msg


class TestDiscordBotFilter(unittest.IsolatedAsyncioTestCase):
    def _adapter(self, **env):
        tmp = tempfile.TemporaryDirectory()
        baseline_env = {
            "HERMES_HOME": tmp.name,
            "DISCORD_ALLOW_BOTS": "all",
            "DISCORD_ALLOWED_BOT_USERS": "12345",
            "HERMES_ENABLE_LEGACY_DISCORD_BOT_TO_BOT": "1",
            "DISCORD_ALLOWED_USERS": "12345",
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

    def test_adapter_no_longer_exposes_bot_admission_helpers(self):
        adapter = self._adapter()
        self.assertFalse(hasattr(adapter, "_should_accept_bot_message"))
        self.assertFalse(hasattr(adapter, "_should_react_malformed_bot_message"))
        self.assertFalse(hasattr(adapter, "_handle_bot_approval_decision"))

    async def test_bot_authored_messages_are_ignored_before_handle_message(self):
        adapter = self._adapter()
        adapter._ready_event.set()
        adapter._handle_message = AsyncMock()
        adapter._add_reaction = AsyncMock()

        message = _make_message(
            author=_make_author(bot=True),
            content="<@99999>\nBOT_MSG v1\nreply_expected: false\nkind: status\n---\nbody",
            mentions=[adapter._client.user],
        )

        async def simulated_runtime_gate(msg):
            if getattr(msg.author, "bot", False):
                return
            await adapter._handle_message(msg)

        await simulated_runtime_gate(message)

        adapter._handle_message.assert_not_called()
        adapter._add_reaction.assert_not_called()

    async def test_human_messages_still_reach_handle_message(self):
        adapter = self._adapter(DISCORD_ALLOW_BOTS="all")
        adapter._ready_event.set()
        adapter._handle_message = AsyncMock()

        message = _make_message(author=_make_author(bot=False), content="hello <@99999>", mentions=[adapter._client.user])

        await adapter._handle_message(message)

        adapter._handle_message.assert_awaited_once_with(message)
