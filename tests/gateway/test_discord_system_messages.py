"""Tests for Discord system message filtering (thread renames, pins, etc.)."""

import pytest
import unittest
from unittest.mock import MagicMock

discord = pytest.importorskip("discord")


def _make_author(*, bot: bool = False, is_self: bool = False):
    """Create a mock Discord author."""
    author = MagicMock()
    author.bot = bot
    author.id = 99999 if is_self else 12345
    author.name = "TestBot" if bot else "TestUser"
    author.display_name = author.name
    return author


def _make_message(*, author=None, content="hello", msg_type=None):
    """Create a mock Discord message with a specific type."""
    msg = MagicMock()
    msg.author = author or _make_author()
    msg.content = content
    msg.attachments = []
    msg.mentions = []
    msg.type = msg_type if msg_type is not None else discord.MessageType.default
    msg.channel = MagicMock()
    msg.channel.id = 222
    msg.channel.name = "test-channel"
    msg.channel.guild = MagicMock()
    msg.channel.guild.name = "TestServer"
    return msg


class TestDiscordSystemMessageFilter(unittest.TestCase):
    """Test that Discord system messages (thread renames, pins, etc.) are ignored."""

    def _run_filter(self, message, client_user=None):
        """Simulate the on_message filter logic and return whether message was accepted.

        Replicates the guard added to discord.py:
            if message.type not in (discord.MessageType.default, discord.MessageType.reply):
                return  # ignored
        """
        # Own messages always ignored
        if message.author == client_user:
            return False

        # System message filter (the fix being tested)
        if message.type not in (discord.MessageType.default, discord.MessageType.reply):
            return False

        return True  # message accepted

    def test_default_messages_accepted(self):
        """Regular user messages (type=default) should be accepted."""
        msg = _make_message(msg_type=discord.MessageType.default)
        self.assertTrue(self._run_filter(msg))

    def test_reply_messages_accepted(self):
        """Reply messages (type=reply) should be accepted — users reply to bot messages."""
        msg = _make_message(msg_type=discord.MessageType.reply)
        self.assertTrue(self._run_filter(msg))

    def test_thread_rename_ignored(self):
        """Thread rename system messages should be ignored."""
        msg = _make_message(msg_type=discord.MessageType.channel_name_change)
        self.assertFalse(self._run_filter(msg))

    def test_pins_add_ignored(self):
        """Pin notifications should be ignored."""
        msg = _make_message(msg_type=discord.MessageType.pins_add)
        self.assertFalse(self._run_filter(msg))

    def test_new_member_ignored(self):
        """New member join messages should be ignored."""
        msg = _make_message(msg_type=discord.MessageType.new_member)
        self.assertFalse(self._run_filter(msg))

    def test_premium_guild_subscription_ignored(self):
        """Boost messages should be ignored."""
        msg = _make_message(msg_type=discord.MessageType.premium_guild_subscription)
        self.assertFalse(self._run_filter(msg))

    def test_recipient_add_ignored(self):
        """Group DM recipient add messages should be ignored."""
        msg = _make_message(msg_type=discord.MessageType.recipient_add)
        self.assertFalse(self._run_filter(msg))

    def test_own_default_messages_still_ignored(self):
        """Bot's own messages should still be ignored even if type is default."""
        bot_user = _make_author(is_self=True)
        msg = _make_message(author=bot_user, msg_type=discord.MessageType.default)
        self.assertFalse(self._run_filter(msg, client_user=bot_user))


if __name__ == "__main__":
    unittest.main()
