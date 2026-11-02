"""Tests for Discord channel_skill_bindings auto-skill resolution."""
from unittest.mock import MagicMock

import pytest


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

    @pytest.mark.parametrize("parent_first", [False, True])
    def test_thread_binding_wins_over_parent_in_any_config_order(self, parent_first):
        """A thread's own binding overrides the one it would inherit from its parent; the order
        the two entries are listed in must not decide which skill a new session loads."""
        bindings = [{"id": "800", "skill": "thread-specific"}, {"id": "700", "skill": "parent-default"}]
        adapter = _make_adapter()
        adapter.config.extra = {"channel_skill_bindings": bindings[::-1] if parent_first else bindings}
        assert adapter._resolve_channel_skills("800", parent_id="700") == ["thread-specific"]

    def test_no_match_returns_none(self):
        adapter = _make_adapter()
        adapter.config.extra = {
            "channel_skill_bindings": [
                {"id": "100", "skills": ["skill-a"]},
            ]
        }
        assert adapter._resolve_channel_skills("999") is None


