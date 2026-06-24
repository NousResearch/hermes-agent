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
    def test_no_bindings_returns_none(self):
        adapter = _make_adapter()
        assert adapter._resolve_channel_skills("123") is None

    def test_match_by_channel_id(self):
        adapter = _make_adapter()
        adapter.config.extra = {
            "channel_skill_bindings": [
                {"id": "100", "skills": ["skill-a", "skill-b"]},
            ]
        }
        assert adapter._resolve_channel_skills("100") == ["skill-a", "skill-b"]

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

    def test_single_skill_string(self):
        adapter = _make_adapter()
        adapter.config.extra = {
            "channel_skill_bindings": [
                {"id": "100", "skill": "solo-skill"},
            ]
        }
        assert adapter._resolve_channel_skills("100") == ["solo-skill"]

    def test_dedup_preserves_order(self):
        adapter = _make_adapter()
        adapter.config.extra = {
            "channel_skill_bindings": [
                {"id": "100", "skills": ["a", "b", "a", "c", "b"]},
            ]
        }
        assert adapter._resolve_channel_skills("100") == ["a", "b", "c"]

    def test_channel_skills_alias_dict(self):
        adapter = _make_adapter()
        adapter.config.extra = {
            "channel_skills": {
                "100": ["keiya-core", "verification-before-completion"],
            }
        }
        assert adapter._resolve_channel_skills("100") == [
            "keiya-core",
            "verification-before-completion",
        ]

    def test_channel_skills_alias_parent_id(self):
        adapter = _make_adapter()
        adapter.config.extra = {
            "channel_skills": {
                "200": "galyarder-core",
            }
        }
        assert adapter._resolve_channel_skills("999", parent_id="200") == ["galyarder-core"]


def test_config_bridges_discord_channel_skills_alias(monkeypatch, tmp_path):
    from gateway.config import Platform, load_gateway_config

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "discord:\n"
        "  channel_skills:\n"
        "    '100':\n"
        "    - keiya-core\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "discord-test-token")

    config = load_gateway_config()

    discord_config = config.platforms[Platform.DISCORD]
    assert discord_config.extra.get("channel_skills") == {"100": ["keiya-core"]}
