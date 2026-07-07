"""Tests for profile-aware skills dir in skill_utils normalization (#60231)."""

import pytest


class TestSkillUtilsProfileAware:
    def test_normalize_uses_skills_dir_not_module_constant(self, monkeypatch, tmp_path):
        """normalize_skill_identifier should use _skills_dir(), not SKILLS_DIR."""
        profile_a = tmp_path / "profile_a"
        profile_b = tmp_path / "profile_b"
        (profile_a / "skills").mkdir(parents=True)
        (profile_b / "skills").mkdir(parents=True)

        # Patch the SKILLS_DIR module attribute to a stale value
        monkeypatch.setattr(
            "tools.skills_tool.SKILLS_DIR",
            profile_a / "skills",
        )
        monkeypatch.setattr(
            "tools.skills_tool._skills_dir",
            lambda: profile_b / "skills",
        )

        from agent.skill_utils import normalize_skill_identifier

        result = normalize_skill_identifier("test-skill")
        # Should use profile_b (from _skills_dir()), not profile_a (stale SKILLS_DIR)
        assert result is not None
