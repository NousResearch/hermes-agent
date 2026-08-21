"""Tests for Discord channel-obfuscation handling.

Discord's August 2026 privacy change ("Channel Obfuscation") dispatches
channels a bot lacks VIEW_CHANNEL on with the name "___hidden___", flag
1 << 17 (CHANNEL_OBFUSCATED) set, and sensitive fields nulled. Enumeration
sites (channel directory, missed-message backfill) must skip these
placeholders — the HTTP API omits them entirely starting Nov 16, 2026.
"""

import unittest
from types import SimpleNamespace

from gateway.platforms.helpers import (
    DISCORD_CHANNEL_OBFUSCATED_FLAG,
    DISCORD_OBFUSCATED_CHANNEL_NAME,
    is_discord_channel_obfuscated,
)


def _channel(name="general", flag_value=0, cid=123):
    return SimpleNamespace(
        id=cid,
        name=name,
        flags=SimpleNamespace(value=flag_value),
    )


class TestIsDiscordChannelObfuscated(unittest.TestCase):
    def test_normal_channel_not_obfuscated(self):
        self.assertFalse(is_discord_channel_obfuscated(_channel()))

    def test_flag_marks_obfuscated(self):
        ch = _channel(name="whatever", flag_value=DISCORD_CHANNEL_OBFUSCATED_FLAG)
        self.assertTrue(is_discord_channel_obfuscated(ch))

    def test_flag_combined_with_other_flags(self):
        ch = _channel(flag_value=DISCORD_CHANNEL_OBFUSCATED_FLAG | (1 << 4))
        self.assertTrue(is_discord_channel_obfuscated(ch))

    def test_sentinel_name_marks_obfuscated_without_flag(self):
        # Older discord.py builds may not surface the new flag bit; the
        # sentinel name is the fallback signal.
        ch = _channel(name=DISCORD_OBFUSCATED_CHANNEL_NAME, flag_value=0)
        self.assertTrue(is_discord_channel_obfuscated(ch))

    def test_missing_flags_attribute(self):
        ch = SimpleNamespace(id=1, name="ok")
        self.assertFalse(is_discord_channel_obfuscated(ch))

    def test_non_int_flag_value_falls_back_to_name(self):
        ch = SimpleNamespace(id=1, name="ok", flags=SimpleNamespace(value=None))
        self.assertFalse(is_discord_channel_obfuscated(ch))


class TestChannelDirectorySkipsObfuscated(unittest.TestCase):
    def test_build_discord_filters_hidden_channels(self):
        from gateway import channel_directory as cd

        visible = _channel(name="general", cid=1)
        hidden_flag = _channel(name="secret", flag_value=DISCORD_CHANNEL_OBFUSCATED_FLAG, cid=2)
        hidden_name = _channel(name=DISCORD_OBFUSCATED_CHANNEL_NAME, cid=3)
        visible_forum = _channel(name="forum-open", cid=4)
        hidden_forum = _channel(
            name="forum-secret", flag_value=DISCORD_CHANNEL_OBFUSCATED_FLAG, cid=5
        )

        guild = SimpleNamespace(
            name="TestGuild",
            text_channels=[visible, hidden_flag, hidden_name],
            forum_channels=[visible_forum, hidden_forum],
        )
        adapter = SimpleNamespace(_client=SimpleNamespace(guilds=[guild]))

        original = cd._build_from_sessions
        cd._build_from_sessions = lambda platform: []
        try:
            channels = cd._build_discord(adapter)
        finally:
            cd._build_from_sessions = original

        ids = {c["id"] for c in channels}
        self.assertEqual(ids, {"1", "4"})
        types = {c["id"]: c["type"] for c in channels}
        self.assertEqual(types["1"], "channel")
        self.assertEqual(types["4"], "forum")


if __name__ == "__main__":
    unittest.main()
