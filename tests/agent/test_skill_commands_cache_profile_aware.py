"""Tests for profile-aware skills slash-command cache invalidation (#60257)."""

import pytest


class TestSkillCommandsCacheProfileAware:
    def test_cache_tracks_skills_dir(self, monkeypatch, tmp_path):
        """get_skill_commands should track _skill_commands_skills_dir."""
        profile_a = tmp_path / "profile_a"
        profile_b = tmp_path / "profile_b"
        (profile_a / "skills").mkdir(parents=True)
        (profile_b / "skills").mkdir(parents=True)

        monkeypatch.setattr(
            "tools.skills_tool._skills_dir",
            lambda: profile_a / "skills",
        )
        monkeypatch.setattr(
            "tools.skills_tool._find_all_skills",
            lambda *a, **kw: [],
        )

        import agent.skill_commands as sc

        # Force a scan
        sc._skill_commands = {}
        sc._skill_commands_skills_dir = None
        sc.get_skill_commands()

        # After scan, _skill_commands_skills_dir should be set
        assert sc._skill_commands_skills_dir == profile_a / "skills"

        # Switch profile — cache should be invalidated
        monkeypatch.setattr(
            "tools.skills_tool._skills_dir",
            lambda: profile_b / "skills",
        )

        # Force re-check
        result = sc.get_skill_commands()
        assert sc._skill_commands_skills_dir == profile_b / "skills"

    def test_scan_uses_skills_dir_not_module_constant(self, monkeypatch, tmp_path):
        """scan_skill_commands should use _skills_dir(), not SKILLS_DIR."""
        profile_dir = tmp_path / "profile"
        (profile_dir / "skills").mkdir(parents=True)

        monkeypatch.setattr(
            "tools.skills_tool.SKILLS_DIR",
            tmp_path / "stale_dir",  # wrong, stale path
        )
        monkeypatch.setattr(
            "tools.skills_tool._skills_dir",
            lambda: profile_dir / "skills",
        )
        monkeypatch.setattr(
            "tools.skills_tool._find_all_skills",
            lambda *a, **kw: [],
        )

        import agent.skill_commands as sc
        sc._skill_commands = {}
        sc._skill_commands_skills_dir = None
        sc.scan_skill_commands()

        # Should use profile_dir (from _skills_dir), not stale_dir (from SKILLS_DIR)
        assert sc._skill_commands_skills_dir == profile_dir / "skills"
