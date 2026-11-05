"""Tests for Discord embed serialization (_serialize_embeds).

Covers the helper that converts Discord embed objects into plain text
so webhook messages (e.g. Railway deploy notifications) are visible to the LLM.
"""

from unittest.mock import MagicMock

import pytest

from plugins.platforms.discord.adapter import _serialize_embeds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embed(title=None, description=None, fields=None):
    """Create a mock Discord embed object."""
    embed = MagicMock()
    embed.title = title
    embed.description = description
    embed.fields = fields or []
    return embed


def _make_field(name="", value=""):
    """Create a mock Discord embed field."""
    field = MagicMock()
    field.name = name
    field.value = value
    return field


# ---------------------------------------------------------------------------
# _serialize_embeds: basic cases
# ---------------------------------------------------------------------------

class TestSerializeEmbedsNone:
    """_serialize_embeds returns None when there are no embeds."""

    def test_none_input(self):
        assert _serialize_embeds(None) is None

    def test_empty_list(self):
        assert _serialize_embeds([]) is None


class TestSerializeEmbedsTitle:
    """Title-only embeds are serialized with bold markdown."""

    def test_title_only(self):
        embeds = [_make_embed(title="Deploy Status")]
        result = _serialize_embeds(embeds)
        assert result == "**Deploy Status**"

    def test_title_with_empty_description(self):
        embeds = [_make_embed(title="Deploy Status", description=None)]
        result = _serialize_embeds(embeds)
        assert result == "**Deploy Status**"

    def test_title_with_description(self):
        embeds = [_make_embed(title="Deploy Status", description="Build succeeded")]
        result = _serialize_embeds(embeds)
        assert "**Deploy Status**" in result
        assert "Build succeeded" in result


class TestSerializeEmbedsDescription:
    """Description-only embeds (no title) are serialized."""

    def test_description_only(self):
        embeds = [_make_embed(description="Build #42 failed")]
        result = _serialize_embeds(embeds)
        assert result == "Build #42 failed"

    def test_empty_title_with_description(self):
        embeds = [_make_embed(title=None, description="Some message")]
        result = _serialize_embeds(embeds)
        assert result == "Some message"


class TestSerializeEmbedsFields:
    """Embed fields are serialized as **name**: value."""

    def test_single_field(self):
        embeds = [_make_embed(fields=[_make_field("Status", "success")])]
        result = _serialize_embeds(embeds)
        assert "**Status**: success" in result

    def test_multiple_fields(self):
        fields = [
            _make_field("Status", "failed"),
            _make_field("Duration", "m30s"),
            _make_field("Commit", "abc123"),
        ]
        embeds = [_make_embed(fields=fields)]
        result = _serialize_embeds(embeds)
        assert "**Status**: failed" in result
        assert "**Duration**: m30s" in result
        assert "**Commit**: abc123" in result

    def test_field_with_empty_name(self):
        fields = [_make_field(name="", value="some value")]
        embeds = [_make_embed(fields=fields)]
        result = _serialize_embeds(embeds)
        assert "**some value" not in result  # empty name still produces "** : value"
        assert "some value" in result

    def test_field_with_empty_value(self):
        fields = [_make_field(name="Label", value="")]
        embeds = [_make_embed(fields=fields)]
        result = _serialize_embeds(embeds)
        assert "**Label**:" in result

    def test_field_with_both_empty(self):
        """Fields where both name and value are empty are skipped."""
        fields = [_make_field(name="", value="")]
        embeds = [_make_embed(fields=fields)]
        result = _serialize_embeds(embeds)
        assert result is None


class TestSerializeEmbedsMultipleEmbeds:
    """Multiple embeds are joined with double newlines."""

    def test_two_embeds(self):
        embeds = [
            _make_embed(title="Deploy", description="started"),
            _make_embed(title="Build", description="completed"),
        ]
        result = _serialize_embeds(embeds)
        assert "**Deploy**" in result
        assert "started" in result
        assert "**Build**" in result
        assert "completed" in result
        assert "\n\n" in result

    def test_mixed_empty_and_nonempty(self):
        """Empty embeds (no title/description/fields) are skipped."""
        embeds = [
            _make_embed(),  # completely empty
            _make_embed(title="Status", description="ok"),
            _make_embed(),  # another empty one
        ]
        result = _serialize_embeds(embeds)
        assert result == "**Status**\nok"


class TestSerializeEmbedsAllEmpty:
    """When all embeds are empty, returns None."""

    def test_all_empty_embeds(self):
        embeds = [_make_embed(), _make_embed()]
        result = _serialize_embeds(embeds)
        assert result is None

    def test_embed_with_only_empty_fields(self):
        embeds = [_make_embed(fields=[_make_field("", "")])]
        result = _serialize_embeds(embeds)
        assert result is None


class TestSerializeEmbedsRealWorld:
    """Real-world webhook payload shapes (Railway, GitHub, etc.)."""

    def test_railway_deploy_notification(self):
        """Typical Railway deploy webhook with title + fields."""
        fields = [
            _make_field("Environment", "production"),
            _make_field("Status", "CRASHED"),
            _make_field("Service", "card-perks-web"),
            _make_field("Deploy URL", "https://railway.app/..."),
        ]
        embeds = [_make_embed(title="New Deployment", description="Deploy #42", fields=fields)]
        result = _serialize_embeds(embeds)
        assert "**New Deployment**" in result
        assert "Deploy #42" in result
        assert "**Environment**: production" in result
        assert "**Status**: CRASHED" in result
        assert "**Service**: card-perks-web" in result

    def test_github_push_notification(self):
        """Typical GitHub push webhook with description only."""
        embeds = [
            _make_embed(
                title="[card-perks] Push to main",
                description="3 commits by kwonye",
            )
        ]
        result = _serialize_embeds(embeds)
        assert "**[card-perks] Push to main**" in result
        assert "3 commits by kwonye" in result

    def test_single_embed_all_fields(self):
        """Embed with title, description, and multiple fields."""
        fields = [
            _make_field("Branch", "main"),
            _make_field("Commit", "abc1234"),
            _make_field("Author", "kwonye"),
        ]
        embeds = [
            _make_embed(
                title="Build Failed",
                description="The build failed with errors",
                fields=fields,
            )
        ]
        result = _serialize_embeds(embeds)
        assert "**Build Failed**" in result
        assert "The build failed with errors" in result
        assert "**Branch**: main" in result
        assert "**Commit**: abc1234" in result
        assert "**Author**: kwonye" in result


class TestSerializeEmbedsEdgeCases:
    """Edge cases and defensive coding."""

    def test_embed_with_no_fields_attribute(self):
        """Embed-like object without a fields attribute."""
        embed = MagicMock(spec=[])  # no attributes at all
        # getattr returns None for missing attributes, fields loop handles None
        result = _serialize_embeds([embed])
        assert result is None

    def test_embed_fields_is_none(self):
        """Embed with fields explicitly set to None."""
        embed = MagicMock()
        embed.title = None
        embed.description = None
        embed.fields = None
        result = _serialize_embeds([embed])
        assert result is None

    def test_whitespace_only_title(self):
        """Title with only whitespace is still included (truthy string)."""
        embeds = [_make_embed(title="  ", description="desc")]
        result = _serialize_embeds(embeds)
        # "  " is truthy, so it should be included
        assert "**" in result
        assert "desc" in result
