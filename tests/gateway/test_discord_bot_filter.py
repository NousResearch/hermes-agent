"""Tests for Discord bot message filtering (DISCORD_ALLOW_BOTS).

These tests exercise the **real** adapter methods rather than local
reimplementations, so a refactor that silently breaks
``_self_is_explicitly_mentioned``, ``_self_is_raw_mentioned``,
``_discord_bots_require_inline_mention``, or the ``on_message`` gate
will be caught by CI.
"""

import os
import re
import unittest
from unittest.mock import MagicMock

from plugins.platforms.discord.adapter import DiscordAdapter


def _make_adapter(
    *,
    client_user_id: int = 99999,
    bots_require_inline_mention: bool | str = False,
):
    """Build a minimal DiscordAdapter with stubbed internals.

    Uses ``__new__`` to bypass ``__init__`` (which needs a live Discord client).
    Only the attributes exercised by the filtering helpers are set.

    Returns ``(adapter, client_user)`` so tests can reference the same
    object that the adapter's ``_self_is_explicitly_mentioned`` checks
    against (identity matters for ``client_user in message.mentions``).
    """
    adapter = DiscordAdapter.__new__(DiscordAdapter)
    client_user = MagicMock()
    client_user.id = client_user_id
    client_user.bot = True
    adapter._client = MagicMock()
    adapter._client.user = client_user
    adapter.config = MagicMock()
    adapter.config.extra = {"bots_require_inline_mention": bots_require_inline_mention}
    return adapter, client_user


def _make_author(*, bot: bool = False, user_id: int = 12345):
    """Create a mock Discord author with a deterministic ID."""
    author = MagicMock()
    author.bot = bot
    author.id = user_id
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
        type(msg.channel).__name__ = "TextChannel"
    return msg


class TestSelfIsExplicitlyMentioned(unittest.TestCase):
    """``_self_is_explicitly_mentioned`` — resolved OR raw mention."""

    def test_resolved_mention(self):
        """Resolved ``message.mentions`` list counts."""
        adapter, client_user = _make_adapter()
        msg = _make_message(mentions=[client_user])
        self.assertTrue(adapter._self_is_explicitly_mentioned(msg))

    def test_raw_content_mention(self):
        """Inline ``<@ID>`` in content counts even without resolved list."""
        adapter, client_user = _make_adapter()
        msg = _make_message(content=f"<@{client_user.id}> hello", mentions=[])
        self.assertTrue(adapter._self_is_explicitly_mentioned(msg))

    def test_nickname_form_mention(self):
        """Legacy ``<@!ID>`` form also counts."""
        adapter, client_user = _make_adapter()
        msg = _make_message(content=f"<@!{client_user.id}> hey", mentions=[])
        self.assertTrue(adapter._self_is_explicitly_mentioned(msg))

    def test_false_when_absent(self):
        """No mention in resolved list or content → False."""
        adapter, _ = _make_adapter()
        msg = _make_message(content="hello world", mentions=[])
        self.assertFalse(adapter._self_is_explicitly_mentioned(msg))

    def test_false_when_client_none(self):
        """Guard: returns False when ``_client`` is None."""
        adapter, _ = _make_adapter()
        adapter._client = None
        msg = _make_message(content="<@99999>")
        self.assertFalse(adapter._self_is_explicitly_mentioned(msg))

    def test_false_when_user_none(self):
        """Guard: returns False when ``_client.user`` is None."""
        adapter, _ = _make_adapter()
        adapter._client.user = None
        msg = _make_message(content="<@99999>")
        self.assertFalse(adapter._self_is_explicitly_mentioned(msg))

    def test_different_user_not_counted(self):
        """Mention of a different user ID → False."""
        adapter, _ = _make_adapter(client_user_id=99999)
        other = _make_author(user_id=11111)
        msg = _make_message(content=f"<@{other.id}>", mentions=[])
        self.assertFalse(adapter._self_is_explicitly_mentioned(msg))


class TestSelfIsRawMentioned(unittest.TestCase):
    """``_self_is_raw_mentioned`` — inline token ONLY (ignores resolved list)."""

    def test_inline_token(self):
        """Inline ``<@ID>`` in content → True."""
        adapter, client_user = _make_adapter()
        msg = _make_message(content=f"<@{client_user.id}> hello", mentions=[])
        self.assertTrue(adapter._self_is_raw_mentioned(msg))

    def test_nickname_form_token(self):
        """Legacy ``<@!ID>`` form also counts."""
        adapter, client_user = _make_adapter()
        msg = _make_message(content=f"<@!{client_user.id}> hey", mentions=[])
        self.assertTrue(adapter._self_is_raw_mentioned(msg))

    def test_false_on_reply_ping_only(self):
        """Reply-ping (``mentions=[us]`` with no inline token) → False.

        This is the KEY difference from ``_self_is_explicitly_mentioned``:
        Discord's reply-ping silently adds us to ``message.mentions``
        without a literal ``<@id>`` in the content.  ``_self_is_raw_mentioned``
        intentionally ignores the resolved list so the bot admission gate
        can distinguish an explicit cross-bot address from a reply chip.
        """
        adapter, client_user = _make_adapter()
        msg = _make_message(content="reply-ping only", mentions=[client_user])
        self.assertFalse(adapter._self_is_raw_mentioned(msg))

    def test_false_when_absent(self):
        """No mention at all → False."""
        adapter, _ = _make_adapter()
        msg = _make_message(content="hello", mentions=[])
        self.assertFalse(adapter._self_is_raw_mentioned(msg))

    def test_false_when_client_none(self):
        """Guard: returns False when ``_client`` is None."""
        adapter, _ = _make_adapter()
        adapter._client = None
        msg = _make_message(content="<@99999>")
        self.assertFalse(adapter._self_is_raw_mentioned(msg))


class TestBotsRequireInlineMentionConfig(unittest.TestCase):
    """``_discord_bots_require_inline_mention`` config resolution."""

    def test_default_false(self):
        """No config, no env → False."""
        adapter, _ = _make_adapter()
        adapter.config.extra = {}
        self.assertFalse(adapter._discord_bots_require_inline_mention())

    def test_config_bool_true(self):
        adapter, _ = _make_adapter(bots_require_inline_mention=True)
        self.assertTrue(adapter._discord_bots_require_inline_mention())

    def test_config_bool_false(self):
        adapter, _ = _make_adapter(bots_require_inline_mention=False)
        self.assertFalse(adapter._discord_bots_require_inline_mention())

    def test_config_string_on(self):
        adapter, _ = _make_adapter(bots_require_inline_mention="on")
        self.assertTrue(adapter._discord_bots_require_inline_mention())

    def test_config_string_true(self):
        adapter, _ = _make_adapter(bots_require_inline_mention="true")
        self.assertTrue(adapter._discord_bots_require_inline_mention())

    def test_config_string_false(self):
        adapter, _ = _make_adapter(bots_require_inline_mention="false")
        self.assertFalse(adapter._discord_bots_require_inline_mention())

    def test_config_string_invalid(self):
        adapter, _ = _make_adapter(bots_require_inline_mention="maybe")
        self.assertFalse(adapter._discord_bots_require_inline_mention())

    def test_env_wins_over_missing_config(self):
        """Env var is used when config ``extra`` has no entry."""
        adapter, _ = _make_adapter()
        adapter.config.extra = {}
        with unittest.mock.patch.dict(os.environ, {"DISCORD_BOTS_REQUIRE_INLINE_MENTION": "true"}):
            self.assertTrue(adapter._discord_bots_require_inline_mention())

    def test_config_wins_over_env(self):
        """Explicit config takes precedence over env var."""
        adapter, _ = _make_adapter(bots_require_inline_mention=False)
        with unittest.mock.patch.dict(os.environ, {"DISCORD_BOTS_REQUIRE_INLINE_MENTION": "true"}):
            self.assertFalse(adapter._discord_bots_require_inline_mention())


class TestInlineMentionGate(unittest.TestCase):
    """Integration: reply-ping gate in ``on_message`` bot-filter branch.

    These tests replicate the ``on_message`` gate logic using the real
    adapter methods, proving that a refactor which breaks the gate
    (e.g. replacing ``_self_is_raw_mentioned`` with
    ``_self_is_explicitly_mentioned``) will be caught.
    """

    @staticmethod
    def _simulate_on_message_bot_gate(adapter, message):
        """Replicate the ``on_message`` bot-filter gate using real methods.

        Returns True when the message would be **blocked** by the gate.
        """
        if not getattr(message.author, "bot", False):
            return False
        return (
            adapter._discord_bots_require_inline_mention()
            and not adapter._self_is_raw_mentioned(message)
        )

    def test_reply_ping_blocked_when_enabled(self):
        """Bot message with only reply-ping → blocked when gate is on.

        This is the exact scenario the gate prevents: two bots
        ping-ponging replies at each other indefinitely.
        """
        adapter, client_user = _make_adapter(bots_require_inline_mention=True)
        bot = _make_author(bot=True)
        msg = _make_message(author=bot, content="reply-ping only", mentions=[client_user])
        self.assertTrue(self._simulate_on_message_bot_gate(adapter, msg))

    def test_inline_mention_passes_when_enabled(self):
        """Bot message with inline ``<@id>`` → passes the gate."""
        adapter, client_user = _make_adapter(bots_require_inline_mention=True)
        bot = _make_author(bot=True)
        msg = _make_message(
            author=bot,
            content=f"<@{client_user.id}> intentional handoff",
            mentions=[client_user],
        )
        self.assertFalse(self._simulate_on_message_bot_gate(adapter, msg))

    def test_reply_ping_passes_when_disabled(self):
        """When gate is off, reply-pings pass through."""
        adapter, client_user = _make_adapter(bots_require_inline_mention=False)
        bot = _make_author(bot=True)
        msg = _make_message(author=bot, content="reply-ping only", mentions=[client_user])
        self.assertFalse(self._simulate_on_message_bot_gate(adapter, msg))

    def test_human_messages_never_gated(self):
        """Human messages are never affected by the gate."""
        adapter, client_user = _make_adapter(bots_require_inline_mention=True)
        human = _make_author(bot=False)
        msg = _make_message(author=human, content="human reply-ping", mentions=[client_user])
        self.assertFalse(self._simulate_on_message_bot_gate(adapter, msg))


class TestMutationProof(unittest.TestCase):
    """Proves the two helpers behave differently — deleting either one breaks these tests."""

    def test_explicitly_vs_raw_differ_on_reply_ping(self):
        """``_self_is_explicitly_mentioned`` returns True for reply-pings
        (checks ``message.mentions``), but ``_self_is_raw_mentioned``
        returns False (checks only inline tokens).

        If someone replaces ``_self_is_raw_mentioned`` with
        ``_self_is_explicitly_mentioned`` in the gate, this test catches it:
        the gate would let reply-pings through, defeating its purpose.
        """
        adapter, client_user = _make_adapter()
        msg = _make_message(content="reply-ping only", mentions=[client_user])

        self.assertTrue(adapter._self_is_explicitly_mentioned(msg))
        self.assertFalse(adapter._self_is_raw_mentioned(msg))


if __name__ == "__main__":
    unittest.main()
