"""Tests for hermes_cli/skills_config.py and skills_tool disabled filtering."""
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# get_disabled_skills
# ---------------------------------------------------------------------------

class TestGetDisabledSkills:
    def test_empty_config(self):
        from hermes_cli.skills_config import get_disabled_skills
        assert get_disabled_skills({}) == set()

    def test_reads_global_disabled(self):
        from hermes_cli.skills_config import get_disabled_skills
        config = {"skills": {"disabled": ["skill-a", "skill-b"]}}
        assert get_disabled_skills(config) == {"skill-a", "skill-b"}

    def test_reads_platform_disabled(self):
        from hermes_cli.skills_config import get_disabled_skills
        config = {"skills": {
            "disabled": ["skill-a"],
            "platform_disabled": {"telegram": ["skill-b"]}
        }}
        assert get_disabled_skills(config, platform="telegram") == {"skill-b"}

    def test_platform_falls_back_to_global(self):
        from hermes_cli.skills_config import get_disabled_skills
        config = {"skills": {"disabled": ["skill-a"]}}
        # no platform_disabled for cli -> falls back to global
        assert get_disabled_skills(config, platform="cli") == {"skill-a"}

    def test_missing_skills_key(self):
        from hermes_cli.skills_config import get_disabled_skills
        assert get_disabled_skills({"other": "value"}) == set()

    def test_empty_disabled_list(self):
        from hermes_cli.skills_config import get_disabled_skills
        assert get_disabled_skills({"skills": {"disabled": []}}) == set()


# ---------------------------------------------------------------------------
# save_disabled_skills
# ---------------------------------------------------------------------------

class TestSaveDisabledSkills:
    @patch("hermes_cli.skills_config.save_config")
    def test_saves_global_sorted(self, mock_save):
        from hermes_cli.skills_config import save_disabled_skills
        config = {}
        save_disabled_skills(config, {"skill-z", "skill-a"})
        assert config["skills"]["disabled"] == ["skill-a", "skill-z"]
        mock_save.assert_called_once()

    @patch("hermes_cli.skills_config.save_config")
    def test_saves_platform_disabled(self, mock_save):
        from hermes_cli.skills_config import save_disabled_skills
        config = {}
        save_disabled_skills(config, {"skill-x"}, platform="telegram")
        assert config["skills"]["platform_disabled"]["telegram"] == ["skill-x"]

    @patch("hermes_cli.skills_config.save_config")
    def test_saves_empty(self, mock_save):
        from hermes_cli.skills_config import save_disabled_skills
        config = {"skills": {"disabled": ["skill-a"]}}
        save_disabled_skills(config, set())
        assert config["skills"]["disabled"] == []

    @patch("hermes_cli.skills_config.save_config")
    def test_creates_skills_key(self, mock_save):
        from hermes_cli.skills_config import save_disabled_skills
        config = {}
        save_disabled_skills(config, {"skill-x"})
        assert "skills" in config
        assert "disabled" in config["skills"]


# ---------------------------------------------------------------------------
# _is_skill_disabled
# ---------------------------------------------------------------------------

class TestIsSkillDisabled:
    @patch("hermes_cli.config.load_config")
    def test_globally_disabled(self, mock_load):
        mock_load.return_value = {"skills": {"disabled": ["bad-skill"]}}
        from tools.skills_tool import _is_skill_disabled
        assert _is_skill_disabled("bad-skill") is True

    @patch("hermes_cli.config.load_config")
    def test_globally_enabled(self, mock_load):
        mock_load.return_value = {"skills": {"disabled": ["other"]}}
        from tools.skills_tool import _is_skill_disabled
        assert _is_skill_disabled("good-skill") is False

    @patch("hermes_cli.config.load_config")
    def test_platform_disabled(self, mock_load):
        mock_load.return_value = {"skills": {
            "disabled": [],
            "platform_disabled": {"telegram": ["tg-skill"]}
        }}
        from tools.skills_tool import _is_skill_disabled
        assert _is_skill_disabled("tg-skill", platform="telegram") is True

    @patch("hermes_cli.config.load_config")
    def test_platform_enabled_overrides_global(self, mock_load):
        mock_load.return_value = {"skills": {
            "disabled": ["skill-a"],
            "platform_disabled": {"telegram": []}
        }}
        from tools.skills_tool import _is_skill_disabled
        # telegram has explicit empty list -> skill-a is NOT disabled for telegram
        assert _is_skill_disabled("skill-a", platform="telegram") is False

    @patch("hermes_cli.config.load_config")
    def test_platform_falls_back_to_global(self, mock_load):
        mock_load.return_value = {"skills": {"disabled": ["skill-a"]}}
        from tools.skills_tool import _is_skill_disabled
        # no platform_disabled for cli -> global
        assert _is_skill_disabled("skill-a", platform="cli") is True

    @patch("hermes_cli.config.load_config")
    def test_empty_config(self, mock_load):
        mock_load.return_value = {}
        from tools.skills_tool import _is_skill_disabled
        assert _is_skill_disabled("any-skill") is False

    @patch("hermes_cli.config.load_config")
    def test_exception_returns_false(self, mock_load):
        mock_load.side_effect = Exception("config error")
        from tools.skills_tool import _is_skill_disabled
        assert _is_skill_disabled("any-skill") is False

    @patch("hermes_cli.config.load_config")
    @patch.dict("os.environ", {"HERMES_PLATFORM": "discord"})
    def test_env_var_platform(self, mock_load):
        mock_load.return_value = {"skills": {
            "platform_disabled": {"discord": ["discord-skill"]}
        }}
        from tools.skills_tool import _is_skill_disabled
        assert _is_skill_disabled("discord-skill") is True


# ---------------------------------------------------------------------------
# _find_all_skills — disabled filtering
# ---------------------------------------------------------------------------

class TestFindAllSkillsFiltering:
    @patch("tools.skills_tool._is_skill_disabled")
    @patch("tools.skills_tool.skill_matches_platform")
    @patch("tools.skills_tool.SKILLS_DIR")
    def test_disabled_skill_excluded(self, mock_dir, mock_platform, mock_disabled, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("---\nname: my-skill\ndescription: A test skill\n---\nContent")
        mock_dir.exists.return_value = True
        mock_dir.rglob.return_value = [skill_md]
        mock_platform.return_value = True
        mock_disabled.return_value = True
        from tools.skills_tool import _find_all_skills
        skills = _find_all_skills()
        assert not any(s["name"] == "my-skill" for s in skills)

    @patch("tools.skills_tool._is_skill_disabled")
    @patch("tools.skills_tool.skill_matches_platform")
    @patch("tools.skills_tool.SKILLS_DIR")
    def test_enabled_skill_included(self, mock_dir, mock_platform, mock_disabled, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("---\nname: my-skill\ndescription: A test skill\n---\nContent")
        mock_dir.exists.return_value = True
        mock_dir.rglob.return_value = [skill_md]
        mock_platform.return_value = True
        mock_disabled.return_value = False
        from tools.skills_tool import _find_all_skills
        skills = _find_all_skills()
        assert any(s["name"] == "my-skill" for s in skills)


# ---------------------------------------------------------------------------
# _get_categories
# ---------------------------------------------------------------------------

class TestGetCategories:
    def test_extracts_unique_categories(self):
        from hermes_cli.skills_config import _get_categories
        skills = [
            {"name": "a", "category": "mlops", "description": ""},
            {"name": "b", "category": "coding", "description": ""},
            {"name": "c", "category": "mlops", "description": ""},
        ]
        cats = _get_categories(skills)
        assert cats == ["coding", "mlops"]

    def test_none_becomes_uncategorized(self):
        from hermes_cli.skills_config import _get_categories
        skills = [{"name": "a", "category": None, "description": ""}]
        assert "uncategorized" in _get_categories(skills)
