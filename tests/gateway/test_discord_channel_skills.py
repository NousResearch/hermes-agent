"""Tests for Discord channel skill/bundle auto-loading."""
from types import SimpleNamespace
from unittest.mock import MagicMock


def _make_adapter(extra=None):
    """Create a minimal DiscordAdapter with mocked config."""
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = object.__new__(DiscordAdapter)
    adapter.config = MagicMock()
    adapter.config.extra = extra or {}
    return adapter


class TestResolveChannelSkills:
    def test_no_bindings_returns_none(self):
        adapter = _make_adapter()
        assert adapter._resolve_channel_skills("123") is None

    def test_match_by_channel_id(self):
        adapter = _make_adapter(
            {
                "channel_skill_bindings": [
                    {"id": "100", "skills": ["skill-a", "skill-b"]},
                ]
            }
        )
        assert adapter._resolve_channel_skills("100") == ["skill-a", "skill-b"]

    def test_match_by_parent_id(self):
        adapter = _make_adapter(
            {
                "channel_skill_bindings": [
                    {"id": "200", "skills": ["forum-skill"]},
                ]
            }
        )
        # channel_id doesn't match, but parent_id does (forum thread)
        assert adapter._resolve_channel_skills("999", parent_id="200") == ["forum-skill"]

    def test_no_match_returns_none(self):
        adapter = _make_adapter(
            {
                "channel_skill_bindings": [
                    {"id": "100", "skills": ["skill-a"]},
                ]
            }
        )
        assert adapter._resolve_channel_skills("999") is None

    def test_single_skill_string(self):
        adapter = _make_adapter(
            {
                "channel_skill_bindings": [
                    {"id": "100", "skill": "solo-skill"},
                ]
            }
        )
        assert adapter._resolve_channel_skills("100") == ["solo-skill"]

    def test_dedup_preserves_order(self):
        adapter = _make_adapter(
            {
                "channel_skill_bindings": [
                    {"id": "100", "skills": ["a", "b", "a", "c", "b"]},
                ]
            }
        )
        assert adapter._resolve_channel_skills("100") == ["a", "b", "c"]

    def test_legacy_channel_skills_dict_match_by_channel_id(self):
        adapter = _make_adapter(
            {
                "channel_skills": {
                    "100": ["skill-a", "skill-b", "skill-a"],
                }
            }
        )
        assert adapter._resolve_channel_skills("100") == ["skill-a", "skill-b"]

    def test_legacy_channel_skills_dict_match_by_parent_id(self):
        adapter = _make_adapter(
            {
                "channel_skills": {
                    "200": "forum-skill",
                }
            }
        )
        assert adapter._resolve_channel_skills("999", parent_id="200") == ["forum-skill"]


class TestDiscordSlashEventAutoSkill:
    def test_build_slash_event_sets_auto_skill(self):
        adapter = _make_adapter(
            {
                "channel_skill_bindings": [
                    {"id": "321", "skills": ["ops-skill"]},
                ],
                "channel_prompts": {"321": "Command prompt"},
            }
        )
        adapter.build_source = MagicMock(return_value=SimpleNamespace())
        adapter._get_effective_topic = MagicMock(return_value=None)

        interaction = SimpleNamespace(
            channel_id=321,
            channel=SimpleNamespace(name="general", guild=None, parent_id=None),
            user=SimpleNamespace(id=1, display_name="Brenner"),
        )

        event = adapter._build_slash_event(interaction, "/retry")

        assert event.auto_skill == ["ops-skill"]
        assert event.channel_prompt == "Command prompt"
