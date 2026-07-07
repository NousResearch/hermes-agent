"""Tests for profile-aware skills slash-command cache invalidation (#40677)."""

import pytest


class TestSkillCommandsCacheProfileAware:
    def test_cache_tracks_skills_dir(self, monkeypatch, tmp_path):
        """get_skill_commands should track _skill_commands_skills_dir."""
        profile_a = tmp_path / "profile_a"
        (profile_a / "skills").mkdir(parents=True)

        monkeypatch.setattr(
            "tools.skills_tool._skills_dir",
            lambda: profile_a / "skills",
        )
        monkeypatch.setattr(
            "tools.skills_tool._find_all_skills",
            lambda *a, **kw: [],
        )

        import agent.skill_commands as sc
        sc._skill_commands = {}
        sc._skill_commands_skills_dir = None
        sc.get_skill_commands()
        assert sc._skill_commands_skills_dir == profile_a / "skills"
