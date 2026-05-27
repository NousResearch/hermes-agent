"""Tests for Discord tag alias correction feature."""

import pytest
from plugins.platforms.discord.adapter import DiscordAdapter
from gateway.config import PlatformConfig


class TestTagAliasCorrection:
    """Test suite for Discord tag alias functionality."""

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

    def test_mixed_mentions(self):
        """Test mixing alias and correct format in same message."""
        config = PlatformConfig(
            enabled=True,
            extra={"tag_aliases": {"lachlan": "556627489947123749"}}
        )
        adapter = DiscordAdapter(config=config)
        
        result = adapter.format_message("Hi @lachlan and <@556627489947123749>")
        assert result == "Hi <@556627489947123749> and <@556627489947123749>"

    def test_no_aliases_configured(self):
        """Test that messages pass through unchanged when no aliases configured."""
        config = PlatformConfig(enabled=True, extra={})
        adapter = DiscordAdapter(config=config)
        
        result = adapter.format_message("Hi @lachlan")
        assert result == "Hi @lachlan"

    def test_unknown_alias_unchanged(self):
        """Test that unknown aliases are not modified."""
        config = PlatformConfig(
            enabled=True,
            extra={"tag_aliases": {"lachlan": "556627489947123749"}}
        )
        adapter = DiscordAdapter(config=config)
        
        result = adapter.format_message("Hi @unknown")
        assert result == "Hi @unknown"

    def test_multiple_aliases(self):
        """Test multiple different aliases in same message."""
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
        
        result = adapter.format_message("Hey @lachlan and @marvin")
        assert result == "Hey <@556627489947123749> and <@123456789012345678>"

    def test_alias_with_punctuation(self):
        """Test alias followed by punctuation."""
        config = PlatformConfig(
            enabled=True,
            extra={"tag_aliases": {"lachlan": "556627489947123749"}}
        )
        adapter = DiscordAdapter(config=config)
        
        result = adapter.format_message("Hi @lachlan, how are you?")
        assert result == "Hi <@556627489947123749>, how are you?"

    def test_alias_at_end_of_message(self):
        """Test alias at the end of a message."""
        config = PlatformConfig(
            enabled=True,
            extra={"tag_aliases": {"lachlan": "556627489947123749"}}
        )
        adapter = DiscordAdapter(config=config)
        
        result = adapter.format_message("Check this out @lachlan")
        assert result == "Check this out <@556627489947123749>"

    def test_alias_at_start_of_message(self):
        """Test alias at the start of a message."""
        config = PlatformConfig(
            enabled=True,
            extra={"tag_aliases": {"lachlan": "556627489947123749"}}
        )
        adapter = DiscordAdapter(config=config)
        
        result = adapter.format_message("@lachlan please review this")
        assert result == "<@556627489947123749> please review this"

    def test_multiple_same_alias(self):
        """Test same alias appearing multiple times."""
        config = PlatformConfig(
            enabled=True,
            extra={"tag_aliases": {"lachlan": "556627489947123749"}}
        )
        adapter = DiscordAdapter(config=config)
        
        result = adapter.format_message("@lachlan @lachlan @lachlan")
        assert result == "<@556627489947123749> <@556627489947123749> <@556627489947123749>"

    def test_preserve_role_mentions(self):
        """Test that role mentions like @everyone are preserved."""
        config = PlatformConfig(
            enabled=True,
            extra={"tag_aliases": {"lachlan": "556627489947123749"}}
        )
        adapter = DiscordAdapter(config=config)
        
        result = adapter.format_message("Hey @everyone")
        assert result == "Hey @everyone"

    def test_preserve_here_mentions(self):
        """Test that @here is preserved."""
        config = PlatformConfig(
            enabled=True,
            extra={"tag_aliases": {"lachlan": "556627489947123749"}}
        )
        adapter = DiscordAdapter(config=config)
        
        result = adapter.format_message("Hey @here")
        assert result == "Hey @here"

    def test_empty_message(self):
        """Test empty message passes through unchanged."""
        config = PlatformConfig(
            enabled=True,
            extra={"tag_aliases": {"lachlan": "556627489947123749"}}
        )
        adapter = DiscordAdapter(config=config)
        
        result = adapter.format_message("")
        assert result == ""

    def test_no_at_symbol(self):
        """Test message without @ symbol."""
        config = PlatformConfig(
            enabled=True,
            extra={"tag_aliases": {"lachlan": "556627489947123749"}}
        )
        adapter = DiscordAdapter(config=config)
        
        result = adapter.format_message("Hello world")
        assert result == "Hello world"

    def test_already_correct_format_with_space(self):
        """Test <@ 123 > format with spaces is preserved."""
        config = PlatformConfig(
            enabled=True,
            extra={"tag_aliases": {"lachlan": "556627489947123749"}}
        )
        adapter = DiscordAdapter(config=config)
        
        result = adapter.format_message("Hi <@556627489947123749>")
        assert result == "Hi <@556627489947123749>"

    def test_tag_alias_not_in_config(self):
        """Test that tag aliases config key missing doesn't break."""
        config = PlatformConfig(
            enabled=True,
            extra={"other_setting": "value"}
        )
        adapter = DiscordAdapter(config=config)
        
        result = adapter.format_message("Hi @lachlan")
        assert result == "Hi @lachlan"

    def test_tag_aliases_none(self):
        """Test that tag_aliases=None doesn't break."""
        config = PlatformConfig(
            enabled=True,
            extra={"tag_aliases": None}
        )
        adapter = DiscordAdapter(config=config)
        
        result = adapter.format_message("Hi @lachlan")
        assert result == "Hi @lachlan"

    def test_tag_aliases_empty_dict(self):
        """Test that empty tag_aliases dict doesn't break."""
        config = PlatformConfig(
            enabled=True,
            extra={"tag_aliases": {}}
        )
        adapter = DiscordAdapter(config=config)
        
        result = adapter.format_message("Hi @lachlan")
        assert result == "Hi @lachlan"

    def test_user_id_as_string(self):
        """Test that user_id stored as string works."""
        config = PlatformConfig(
            enabled=True,
            extra={"tag_aliases": {"lachlan": "556627489947123749"}}
        )
        adapter = DiscordAdapter(config=config)
        
        result = adapter.format_message("Hi @lachlan")
        assert result == "Hi <@556627489947123749>"

    def test_user_id_as_int(self):
        """Test that user_id stored as int works."""
        config = PlatformConfig(
            enabled=True,
            extra={"tag_aliases": {"lachlan": 556627489947123749}}
        )
        adapter = DiscordAdapter(config=config)
        
        result = adapter.format_message("Hi @lachlan")
        assert result == "Hi <@556627489947123749>"

    def test_complex_message_with_code_block(self):
        """Test message with code blocks and mentions."""
        config = PlatformConfig(
            enabled=True,
            extra={"tag_aliases": {"lachlan": "556627489947123749"}}
        )
        adapter = DiscordAdapter(config=config)
        
        result = adapter.format_message(
            "Hey @lachlan, check this:\n```python\nprint('hello')\n```"
        )
        assert result == (
            "Hey <@556627489947123749>, check this:\n```python\nprint('hello')\n```"
        )

    def test_underscore_in_username(self):
        """Test usernames with underscores."""
        config = PlatformConfig(
            enabled=True,
            extra={"tag_aliases": {"rumpy_pumpy": "556627489947123749"}}
        )
        adapter = DiscordAdapter(config=config)
        
        result = adapter.format_message("Hi @rumpy_pumpy")
        assert result == "Hi <@556627489947123749>"

    def test_number_in_username(self):
        """Test usernames with numbers."""
        config = PlatformConfig(
            enabled=True,
            extra={"tag_aliases": {"marvin2": "123456789012345678"}}
        )
        adapter = DiscordAdapter(config=config)
        
        result = adapter.format_message("Hi @marvin2")
        assert result == "Hi <@123456789012345678>"
