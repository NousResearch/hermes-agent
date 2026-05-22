"""Tests for Discord embed serialization (issue #26733)."""

import pytest
from types import SimpleNamespace


class MockEmbed:
    """Mock Discord embed for testing."""
    def __init__(self, title=None, description=None, fields=None, footer=None, url=None):
        self.title = title
        self.description = description
        self.fields = fields or []
        self.footer = SimpleNamespace(text=footer) if footer else None
        self.url = url


class TestSerializeEmbeds:
    """Test the _serialize_embeds method."""

    def test_empty_embeds(self):
        """Test with no embeds."""
        from gateway.platforms.discord import DiscordAdapter
        adapter = DiscordAdapter.__new__(DiscordAdapter)
        result = adapter._serialize_embeds([])
        assert result == ""

    def test_embeds_with_title_only(self):
        """Test embed with title only."""
        from gateway.platforms.discord import DiscordAdapter
        adapter = DiscordAdapter.__new__(DiscordAdapter)
        embed = MockEmbed(title="Test Title")
        result = adapter._serialize_embeds([embed])
        assert result == "**Test Title**"

    def test_embeds_with_description_only(self):
        """Test embed with description only."""
        from gateway.platforms.discord import DiscordAdapter
        adapter = DiscordAdapter.__new__(DiscordAdapter)
        embed = MockEmbed(description="Test description")
        result = adapter._serialize_embeds([embed])
        assert result == "Test description"

    def test_embeds_with_title_and_description(self):
        """Test embed with both title and description."""
        from gateway.platforms.discord import DiscordAdapter
        adapter = DiscordAdapter.__new__(DiscordAdapter)
        embed = MockEmbed(title="Title", description="Description")
        result = adapter._serialize_embeds([embed])
        assert result == "**Title**\nDescription"

    def test_embeds_with_fields(self):
        """Test embed with fields."""
        from gateway.platforms.discord import DiscordAdapter
        adapter = DiscordAdapter.__new__(DiscordAdapter)
        fields = [
            SimpleNamespace(name="Field1", value="Value1", inline=False),
            SimpleNamespace(name="Field2", value="Value2", inline=True),
        ]
        embed = MockEmbed(title="Title", fields=fields)
        result = adapter._serialize_embeds([embed])
        expected = "**Title**\n**Field1**: Value1\n`**Field2**: Value2`"
        assert result == expected

    def test_embeds_with_footer(self):
        """Test embed with footer."""
        from gateway.platforms.discord import DiscordAdapter
        adapter = DiscordAdapter.__new__(DiscordAdapter)
        embed = MockEmbed(title="Title", footer="Footer text")
        result = adapter._serialize_embeds([embed])
        assert result == "**Title**\n— Footer text"

    def test_embeds_with_url(self):
        """Test embed with URL."""
        from gateway.platforms.discord import DiscordAdapter
        adapter = DiscordAdapter.__new__(DiscordAdapter)
        embed = MockEmbed(title="Title", url="https://example.com")
        result = adapter._serialize_embeds([embed])
        assert result == "**Title**\n[Link](https://example.com)"

    def test_multiple_embeds(self):
        """Test multiple embeds separated by double newlines."""
        from gateway.platforms.discord import DiscordAdapter
        adapter = DiscordAdapter.__new__(DiscordAdapter)
        embed1 = MockEmbed(title="Title 1", description="Desc 1")
        embed2 = MockEmbed(title="Title 2", description="Desc 2")
        result = adapter._serialize_embeds([embed1, embed2])
        expected = "**Title 1**\nDesc 1\n\n**Title 2**\nDesc 2"
        assert result == expected

    def test_webhook_like_embed(self):
        """Test typical webhook embed (GitHub, Railway style)."""
        from gateway.platforms.discord import DiscordAdapter
        adapter = DiscordAdapter.__new__(DiscordAdapter)
        fields = [
            SimpleNamespace(name="Status", value="✅ Passed", inline=False),
            SimpleNamespace(name="Commit", value="abc123", inline=True),
            SimpleNamespace(name="Branch", value="main", inline=True),
        ]
        embed = MockEmbed(
            title="✅ Build #1234 succeeded",
            description="The build completed successfully.",
            fields=fields,
            footer="GitHub Actions",
            url="https://github.com/owner/repo/actions/runs/1234"
        )
        result = adapter._serialize_embeds([embed])
        lines = result.split("\n")
        assert "**✅ Build #1234 succeeded**" in lines
        assert "The build completed successfully." in lines
        assert "**Status**: ✅ Passed" in lines
        assert "`**Commit**: abc123`" in lines
        assert "— GitHub Actions" in lines
        assert "[Link](https://github.com/owner/repo/actions/runs/1234)" in lines
