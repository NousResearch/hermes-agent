"""Tests for Discord tag alias correction feature."""

import pytest
from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter


class TestTagAliasCorrection:
    """Test tag alias replacement in Discord adapter."""

    def test_basic_alias_replacement(self):
        """Test basic @name to <@id> conversion."""
        config = PlatformConfig(
            enabled=True,
            extra={"tag_aliases": {"lachlan": "556627489947123749"}}
        )
        adapter = DiscordAdapter(config=config)
        result = adapter.format_message("Hi @lachlan")
        assert result == "Hi <@556627489947123749>"

    def test_case_insensitive(self):
        """Test that matching is case-insensitive."""
        config = PlatformConfig(
            enabled=True,
            extra={"tag_aliases": {"lachlan": "556627489947123749"}}
        )
        adapter = DiscordAdapter(config=config)
        result = adapter.format_message("Hi @Lachlan and @LACHLAN")
        assert result == "Hi <@556627489947123749> and <@556627489947123749>"

    def test_preserve_correct_format(self):
        """Test that <@id> format is preserved unchanged."""
        config = PlatformConfig(
            enabled=True,
            extra={"tag_aliases": {"lachlan": "556627489947123749"}}
        )
        adapter = DiscordAdapter(config=config)
        result = adapter.format_message("Hi <@556627489947123749>")
        assert result == "Hi <@556627489947123749>"

    def test_no_space_before_at_not_matched(self):
        """Test that @ without space before is NOT matched."""
        config = PlatformConfig(
            enabled=True,
            extra={"tag_aliases": {"lachlan": "556627489947123749"}}
        )
        adapter = DiscordAdapter(config=config)
        result = adapter.format_message("test@lachlan")
        assert result == "test@lachlan"

    def test_word_before_at_not_matched(self):
        """Test that word character before @ prevents match."""
        config = PlatformConfig(
            enabled=True,
            extra={"tag_aliases": {"lachlan": "556627489947123749"}}
        )
        adapter = DiscordAdapter(config=config)
        result = adapter.format_message("x@lachlan")
        assert result == "x@lachlan"

    def test_punctuation_before_at_is_matched(self):
        """Test that punctuation before @ allows match."""
        config = PlatformConfig(
            enabled=True,
            extra={"tag_aliases": {"lachlan": "556627489947123749"}}
        )
        adapter = DiscordAdapter(config=config)
        result = adapter.format_message("Hi, @lachlan!")
        assert result == "Hi, <@556627489947123749>!"

    def test_preserve_role_mentions(self):
        """Test that @everyone and @here are preserved."""
        config = PlatformConfig(
            enabled=True,
            extra={"tag_aliases": {"lachlan": "556627489947123749"}}
        )
        adapter = DiscordAdapter(config=config)
        result = adapter.format_message("Hey @everyone, @lachlan is here")
        assert result == "Hey @everyone, <@556627489947123749> is here"

    def test_multiple_aliases(self):
        """Test multiple different aliases in one message."""
        config = PlatformConfig(
            enabled=True,
            extra={
                "tag_aliases": {
                    "lachlan": "556627489947123749",
                    "marvin": "123456789012345678"
                }
            }
        )
        adapter = DiscordAdapter(config=config)
        result = adapter.format_message("@lachlan and @marvin")
        assert result == "<@556627489947123749> and <@123456789012345678>"

    def test_no_tag_aliases_config(self):
        """Test that messages pass through unchanged when no aliases configured."""
        config = PlatformConfig(enabled=True, extra={})
        adapter = DiscordAdapter(config=config)
        result = adapter.format_message("Hi @lachlan")
        assert result == "Hi @lachlan"

    def test_tag_alias_not_in_config(self):
        """Test that unknown aliases are not replaced."""
        config = PlatformConfig(
            enabled=True,
            extra={"tag_aliases": {"lachlan": "556627489947123749"}}
        )
        adapter = DiscordAdapter(config=config)
        result = adapter.format_message("Hi @unknown")
        assert result == "Hi @unknown"
