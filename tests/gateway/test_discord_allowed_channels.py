"""Regression guard for #14920: wildcard "*" in Discord channel config lists.

Setting ``allowed_channels: "*"``, ``free_response_channels: "*"``, or
``ignored_channels: "*"`` in config (or their ``DISCORD_*_CHANNELS`` env var
equivalents) must behave as a wildcard — i.e. the bot responds in every
channel (or is silenced in every channel, for the ignored list). Previously
the literal string "*" was placed into a set and compared against numeric
channel IDs via set-intersection, which always produced an empty set and
caused every message to be silently dropped (for ``allowed_channels``) or
every ``free_response`` / ``ignored`` check to fail open.
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


def _channel_is_ignored(channel_id: str, ignored_channels_raw: str) -> bool:
    """Replicate the ignored-channel check from discord.py on_message."""
    ignored_channels = {
        ch.strip() for ch in ignored_channels_raw.split(",") if ch.strip()
    }
    return "*" in ignored_channels or bool({channel_id} & ignored_channels)


def _channel_is_free_response(channel_id: str, free_channels_raw: str) -> bool:
    """Replicate the free-response-channel check from discord.py on_message."""
    free_channels = {
        ch.strip() for ch in free_channels_raw.split(",") if ch.strip()
    }
    return "*" in free_channels or bool({channel_id} & free_channels)


class TestDiscordAllowedChannelsWildcard(unittest.TestCase):
    """Wildcard and channel-list behaviour for DISCORD_ALLOWED_CHANNELS."""

    def test_wildcard_allows_any_channel(self):
        """'*' should allow messages from any channel ID."""
        self.assertTrue(_channel_is_allowed("1234567890", "*"))


    def test_exact_match_allowed(self):
        """Channel ID present in the explicit list is allowed."""
        self.assertTrue(_channel_is_allowed("1234567890", "1234567890,9876543210"))


class TestDiscordIgnoredChannelsWildcard(unittest.TestCase):
    """Wildcard and channel-list behaviour for DISCORD_IGNORED_CHANNELS."""

    def test_wildcard_silences_every_channel(self):
        """'*' in ignored_channels silences the bot everywhere."""
        self.assertTrue(_channel_is_ignored("1234567890", "*"))


class TestDiscordFreeResponseChannelsWildcard(unittest.TestCase):
    """Wildcard and channel-list behaviour for DISCORD_FREE_RESPONSE_CHANNELS."""

    def test_wildcard_makes_every_channel_free_response(self):
        """'*' in free_response_channels exempts every channel from mention-required."""
        self.assertTrue(_channel_is_free_response("1234567890", "*"))


    def test_exact_match_is_free_response(self):
        self.assertTrue(_channel_is_free_response("111", "111,222"))


