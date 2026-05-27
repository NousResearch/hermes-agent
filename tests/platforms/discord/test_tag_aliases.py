"""Tests for Discord tag alias correction feature."""

import pytest
from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter


@pytest.fixture
def adapter_with_alias():
    """Create adapter with test tag aliases."""
    config = PlatformConfig(
        enabled=True,
        extra={"tag_aliases": {"alice": "0123456789123", "bob": "9876543210987"}}
    )
    return DiscordAdapter(config=config)


@pytest.fixture
def adapter_without_alias():
    """Create adapter without tag aliases."""
    config = PlatformConfig(enabled=True, extra={})
    return DiscordAdapter(config=config)


class TestTagAliasCorrection:
    """Test tag alias replacement in Discord adapter."""

    def test_basic_replacement(self, adapter_with_alias):
        """Test basic @name to <@id> conversion."""
        result = adapter_with_alias.format_message("Hi @alice")
        assert result == "Hi <@0123456789123>"

    def test_case_insensitive(self, adapter_with_alias):
        """Test that matching is case-insensitive."""
        result = adapter_with_alias.format_message("Hi @Alice and @ALICE")
        assert result == "Hi <@0123456789123> and <@0123456789123>"

    def test_preserve_correct_format(self, adapter_with_alias):
        """Test that <@id> format is preserved unchanged."""
        result = adapter_with_alias.format_message("Hi <@0123456789123>")
        assert result == "Hi <@0123456789123>"

    @pytest.mark.parametrize("prefix", ["test@", "x@"])
    def test_no_space_before_at_not_matched(self, adapter_with_alias, prefix):
        """Test that @ without space before is NOT matched."""
        result = adapter_with_alias.format_message(f"{prefix}alice")
        assert result == f"{prefix}alice"

    def test_punctuation_before_at_is_matched(self, adapter_with_alias):
        """Test that punctuation before @ allows match."""
        result = adapter_with_alias.format_message("Hi, @alice!")
        assert result == "Hi, <@0123456789123>!"

    def test_preserve_role_mentions(self, adapter_with_alias):
        """Test that @everyone and @here are preserved."""
        result = adapter_with_alias.format_message("Hey @everyone, @alice is here")
        assert result == "Hey @everyone, <@0123456789123> is here"

    def test_multiple_aliases(self, adapter_with_alias):
        """Test multiple different aliases in one message."""
        result = adapter_with_alias.format_message("@alice and @bob")
        assert result == "<@0123456789123> and <@9876543210987>"

    def test_no_tag_aliases_config(self, adapter_without_alias):
        """Test that messages pass through unchanged when no aliases configured."""
        result = adapter_without_alias.format_message("Hi @alice")
        assert result == "Hi @alice"

    def test_tag_alias_not_in_config(self, adapter_with_alias):
        """Test that unknown aliases are not replaced."""
        result = adapter_with_alias.format_message("Hi @unknown")
        assert result == "Hi @unknown"
