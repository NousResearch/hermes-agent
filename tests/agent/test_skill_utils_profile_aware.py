"""Tests for profile-aware path resolution in skill_utils (#40677)."""

import pytest


class TestSkillUtilsProfileAware:
    def test_normalize_uses_skills_dir_not_module_constant(self, monkeypatch, tmp_path):
        """normalize_skill_identifier should use _skills_dir(), not SKILLS_DIR."""
        profile_b = tmp_path / "profile_b"
        (profile_b / "skills").mkdir(parents=True)

        monkeypatch.setattr(
            "tools.skills_tool.SKILLS_DIR",
            tmp_path / "stale_dir",
        )
        monkeypatch.setattr(
            "tools.skills_tool._skills_dir",
            lambda: profile_b / "skills",
        )

        from agent.skill_utils import normalize_skill_lookup_name
        result = normalize_skill_lookup_name("test-skill")
        # Should not crash with stale SKILLS_DIR; uses _skills_dir()
        assert result == "test-skill"
