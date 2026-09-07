"""Tests for Discord channel_skill_bindings auto-skill resolution."""
from unittest.mock import MagicMock


def _make_adapter():
    """Create a minimal DiscordAdapter with mocked config."""
    from plugins.platforms.discord.adapter import DiscordAdapter
    adapter = object.__new__(DiscordAdapter)
    adapter.config = MagicMock()
    adapter.config.extra = {}
    return adapter


class TestResolveChannelSkills:


    def test_match_by_parent_id(self):
        adapter = _make_adapter()
        adapter.config.extra = {
            "channel_skill_bindings": [
                {"id": "200", "skills": ["forum-skill"]},
            ]
        }
        # channel_id doesn't match, but parent_id does (forum thread)
        assert adapter._resolve_channel_skills("999", parent_id="200") == ["forum-skill"]

    def test_no_match_returns_none(self):
        adapter = _make_adapter()
        adapter.config.extra = {
            "channel_skill_bindings": [
                {"id": "100", "skills": ["skill-a"]},
            ]
        }
        assert adapter._resolve_channel_skills("999") is None


