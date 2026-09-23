"""Tests for Slack channel_skill_bindings auto-skill resolution."""
from unittest.mock import MagicMock


def _make_adapter(extra=None):
    """Create a minimal SlackAdapter stub with the given ``config.extra``."""
    from plugins.platforms.slack.adapter import SlackAdapter
    adapter = object.__new__(SlackAdapter)
    adapter.config = MagicMock()
    adapter.config.extra = extra or {}
    return adapter


def _resolve(adapter, channel_id, parent_id=None):
    from gateway.platforms.base import resolve_channel_skills
    return resolve_channel_skills(adapter.config.extra, channel_id, parent_id)


class TestSlackResolveChannelSkills:

    def test_match_by_dm_channel_id(self):
        """The primary use case: binding a skill to a Slack DM channel."""
        adapter = _make_adapter({
            "channel_skill_bindings": [
                {"id": "D0ATH9TQ0G6", "skills": ["german-flashcards"]},
            ]
        })
        assert _resolve(adapter, "D0ATH9TQ0G6") == ["german-flashcards"]


    def test_no_match_returns_none(self):
        adapter = _make_adapter({
            "channel_skill_bindings": [
                {"id": "D0AAA", "skills": ["skill-a"]},
            ]
        })
        assert _resolve(adapter, "D0BBB") is None

    def test_single_skill_string(self):
        adapter = _make_adapter({
            "channel_skill_bindings": [
                {"id": "D0ATH9TQ0G6", "skill": "german-flashcards"},
            ]
        })
        assert _resolve(adapter, "D0ATH9TQ0G6") == ["german-flashcards"]


    def test_empty_skills_list_returns_none(self):
        adapter = _make_adapter({
            "channel_skill_bindings": [
                {"id": "D0ABC", "skills": []},
            ]
        })
        assert _resolve(adapter, "D0ABC") is None


