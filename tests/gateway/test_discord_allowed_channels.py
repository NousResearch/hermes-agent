"""Regression guard for #14920: DISCORD_ALLOWED_CHANNELS="*" should allow all channels.

Setting allowed_channels: "*" in config (or DISCORD_ALLOWED_CHANNELS="*" as env var)
must behave as a wildcard — i.e. the bot responds in every channel. Previously the
literal string "*" was placed into the set and compared against numeric channel IDs via
set-intersection, which always produced an empty set, causing every message to be
silently dropped.
"""

import unittest


def _channel_is_allowed(channel_id: str, allowed_channels_raw: str) -> bool:
    """Replicate the channel-allow-list check from discord.py on_message."""
    if not allowed_channels_raw:
        return True
    allowed_channels = {ch.strip() for ch in allowed_channels_raw.split(",") if ch.strip()}
    if "*" in allowed_channels:
        return True
    return bool({channel_id} & allowed_channels)


class TestDiscordAllowedChannelsWildcard(unittest.TestCase):
    """Wildcard and channel-list behaviour for DISCORD_ALLOWED_CHANNELS."""

    def test_wildcard_allows_any_channel(self):
        """'*' should allow messages from any channel ID."""
        self.assertTrue(_channel_is_allowed("1234567890", "*"))

    def test_wildcard_in_list_allows_any_channel(self):
        """'*' mixed with other entries still allows any channel."""
        self.assertTrue(_channel_is_allowed("9999999999", "111,*,222"))

    def test_exact_match_allowed(self):
        """Channel ID present in the explicit list is allowed."""
        self.assertTrue(_channel_is_allowed("1234567890", "1234567890,9876543210"))

    def test_non_matching_channel_blocked(self):
        """Channel ID absent from the explicit list is blocked."""
        self.assertFalse(_channel_is_allowed("5555555555", "1234567890,9876543210"))

    def test_empty_allowlist_allows_all(self):
        """Empty DISCORD_ALLOWED_CHANNELS means no restriction."""
        self.assertTrue(_channel_is_allowed("1234567890", ""))

    def test_whitespace_only_entry_ignored(self):
        """Entries that are only whitespace are stripped and ignored."""
        self.assertFalse(_channel_is_allowed("1234567890", "  ,  "))
