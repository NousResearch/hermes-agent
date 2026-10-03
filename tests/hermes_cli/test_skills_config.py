"""Tests for hermes_cli/skills_config.py and skills_tool disabled filtering."""
from unittest.mock import patch


# ---------------------------------------------------------------------------
# get_disabled_skills
# ---------------------------------------------------------------------------

class TestGetDisabledSkills:
    def test_empty_config(self):
        from hermes_cli.skills_config import get_disabled_skills
        assert get_disabled_skills({}) == set()




    def test_null_skills_section(self):
        """``skills:`` with no value (YAML null) must not crash (#13026)."""
        from hermes_cli.skills_config import get_disabled_skills
        assert get_disabled_skills({"skills": None}) == set()
        assert get_disabled_skills({"skills": None}, platform="telegram") == set()




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


# ---------------------------------------------------------------------------
# _is_skill_disabled
# ---------------------------------------------------------------------------

class TestIsSkillDisabled:


    @patch("hermes_cli.config.load_config")
    def test_platform_disabled(self, mock_load):
        mock_load.return_value = {"skills": {
            "disabled": [],
            "platform_disabled": {"telegram": ["tg-skill"]}
        }}
        from tools.skills_tool import _is_skill_disabled
        assert _is_skill_disabled("tg-skill", platform="telegram") is True



    @patch("hermes_cli.config.load_config")
    @patch.dict("os.environ", {"HERMES_PLATFORM": "discord"})
    def test_env_var_platform(self, mock_load):
        mock_load.return_value = {"skills": {
            "platform_disabled": {"discord": ["discord-skill"]}
        }}
        from tools.skills_tool import _is_skill_disabled
        assert _is_skill_disabled("discord-skill") is True


# ---------------------------------------------------------------------------
# get_disabled_skill_names — explicit platform param & env var fallback
# ---------------------------------------------------------------------------

class TestGetDisabledSkillNames:
    """Tests for agent.skill_utils.get_disabled_skill_names."""

    def test_explicit_platform_param(self, tmp_path, monkeypatch):
        """Explicit platform= parameter should resolve per-platform list."""
        config = tmp_path / "config.yaml"
        config.write_text(
            "skills:\n"
            "  disabled:\n"
            "    - global-skill\n"
            "  platform_disabled:\n"
            "    telegram:\n"
            "      - tg-only-skill\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("HERMES_PLATFORM", raising=False)
        monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)

        from agent.skill_utils import get_disabled_skill_names
        result = get_disabled_skill_names(platform="telegram")
        assert result == {"tg-only-skill", "global-skill"}

    def test_session_platform_env_var(self, tmp_path, monkeypatch):
        """HERMES_SESSION_PLATFORM should be used when HERMES_PLATFORM is unset."""
        config = tmp_path / "config.yaml"
        config.write_text(
            "skills:\n"
            "  disabled:\n"
            "    - global-skill\n"
            "  platform_disabled:\n"
            "    discord:\n"
            "      - discord-skill\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("HERMES_PLATFORM", raising=False)
        monkeypatch.setenv("HERMES_SESSION_PLATFORM", "discord")

        from agent.skill_utils import get_disabled_skill_names
        result = get_disabled_skill_names()
        assert result == {"discord-skill", "global-skill"}

    def test_hermes_platform_takes_precedence(self, tmp_path, monkeypatch):
        """HERMES_PLATFORM should win over HERMES_SESSION_PLATFORM."""
        config = tmp_path / "config.yaml"
        config.write_text(
            "skills:\n"
            "  platform_disabled:\n"
            "    telegram:\n"
            "      - tg-skill\n"
            "    discord:\n"
            "      - discord-skill\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_PLATFORM", "telegram")
        monkeypatch.setenv("HERMES_SESSION_PLATFORM", "discord")

        from agent.skill_utils import get_disabled_skill_names
        result = get_disabled_skill_names()
        assert result == {"tg-skill"}


# ---------------------------------------------------------------------------
# _find_all_skills — disabled filtering
# ---------------------------------------------------------------------------

class TestFindAllSkillsFiltering:
    @patch("tools.skills_tool._get_disabled_skill_names", return_value={"my-skill"})
    @patch("tools.skills_tool.skill_matches_platform", return_value=True)
    def test_disabled_skill_excluded(self, mock_platform, mock_disabled, tmp_path, monkeypatch):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("---\nname: my-skill\ndescription: A test skill\n---\nContent")
        # Point SKILLS_DIR at the real tempdir so iter_skill_index_files
        # (which uses os.walk) can actually find the file.
        import tools.skills_tool as _st
        import agent.skill_utils as _su
        monkeypatch.setattr(_st, "SKILLS_DIR", tmp_path)
        monkeypatch.setattr(_su, "get_external_skills_dirs", lambda: [])
        from tools.skills_tool import _find_all_skills
        skills = _find_all_skills()
        assert not any(s["name"] == "my-skill" for s in skills)


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


# ---------------------------------------------------------------------------
# Config defaults for bundled_whitelist / bundled_enabled
# ---------------------------------------------------------------------------


class TestSkillsConfigDefaults:
    """bundled_whitelist and bundled_enabled exist in DEFAULT_CONFIG."""

    def test_bundled_whitelist_default(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG
        skills = DEFAULT_CONFIG["skills"]
        assert "bundled_whitelist" in skills
        assert skills["bundled_whitelist"] is False

    def test_bundled_enabled_default(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG
        skills = DEFAULT_CONFIG["skills"]
        assert "bundled_enabled" in skills
        assert skills["bundled_enabled"] == []
