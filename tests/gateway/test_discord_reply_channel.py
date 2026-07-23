"""Tests for the single designated reply channel (``discord.reply_channel``).

When ``discord.reply_channel`` is set, agent replies triggered from other
guild channels (and threads under them) are redirected to that one channel,
prefixed with a ``[re: #origin]`` context line. DMs are never redirected, and
messages originating in the reply channel itself (or threads under it) reply
in place. When the key is unset, behavior is unchanged.
"""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

adapter_mod = load_plugin_adapter("discord")
DiscordAdapter = adapter_mod.DiscordAdapter


def _make_adapter(reply_channel=None):
    """Build a bare adapter with only the config surface the helpers use."""
    adapter = DiscordAdapter.__new__(DiscordAdapter)
    extra = {}
    if reply_channel is not None:
        extra["reply_channel"] = reply_channel
    adapter.config = SimpleNamespace(extra=extra)
    return adapter


def _guild_channel(channel_id, name="general", parent_id=None):
    return SimpleNamespace(
        id=channel_id, name=name, parent_id=parent_id, guild=object()
    )


def _dm_channel(channel_id):
    return SimpleNamespace(id=channel_id, name=None, parent_id=None, guild=None)


class TestDiscordReplyChannelConfig(unittest.TestCase):
    def test_unset_returns_empty(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DISCORD_REPLY_CHANNEL", None)
            self.assertEqual(_make_adapter()._discord_reply_channel(), "")

    def test_config_value_wins(self):
        adapter = _make_adapter(reply_channel="123")
        self.assertEqual(adapter._discord_reply_channel(), "123")

    def test_numeric_yaml_scalar_coerced(self):
        adapter = _make_adapter(reply_channel=1491973769726791812)
        self.assertEqual(adapter._discord_reply_channel(), "1491973769726791812")

    def test_env_fallback(self):
        adapter = _make_adapter()
        with mock.patch.dict(os.environ, {"DISCORD_REPLY_CHANNEL": " 456 "}):
            self.assertEqual(adapter._discord_reply_channel(), "456")


class TestDiscordReplyRedirect(unittest.TestCase):
    def test_off_means_no_redirect(self):
        adapter = _make_adapter(reply_channel="")
        self.assertIsNone(
            adapter._reply_redirect_target(_guild_channel(999, "random"))
        )

    def test_other_guild_channel_redirects_with_origin_name(self):
        adapter = _make_adapter(reply_channel="123")
        result = adapter._reply_redirect_target(_guild_channel(999, "random"))
        self.assertEqual(result, ("123", "random"))

    def test_dm_never_redirects(self):
        adapter = _make_adapter(reply_channel="123")
        self.assertIsNone(adapter._reply_redirect_target(_dm_channel(999)))

    def test_reply_channel_itself_replies_in_place(self):
        adapter = _make_adapter(reply_channel="123")
        self.assertIsNone(
            adapter._reply_redirect_target(_guild_channel(123, "agent-replies"))
        )

    def test_thread_under_reply_channel_replies_in_place(self):
        adapter = _make_adapter(reply_channel="123")
        thread = _guild_channel(555, "some-thread", parent_id=123)
        self.assertIsNone(adapter._reply_redirect_target(thread))

    def test_thread_under_other_channel_redirects(self):
        adapter = _make_adapter(reply_channel="123")
        thread = _guild_channel(555, "bug-thread", parent_id=999)
        self.assertEqual(
            adapter._reply_redirect_target(thread), ("123", "bug-thread")
        )

    def test_origin_without_name_uses_id(self):
        adapter = _make_adapter(reply_channel="123")
        chan = SimpleNamespace(id=999, name=None, parent_id=None, guild=object())
        self.assertEqual(adapter._reply_redirect_target(chan), ("123", "999"))


if __name__ == "__main__":
    unittest.main()
