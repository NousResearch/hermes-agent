"""Tests for SkillDiscovery bridge."""

import pytest
from pathlib import Path


class TestSkillDiscovery:
    def test_builtin_skills_present(self):
        from hermes_cli.code.skill_discovery import get_builtin_skills
        skills = get_builtin_skills()
        assert len(skills) >= 7
        names = {s["name"] for s in skills}
        assert "fix_build" in names
        assert "review_diff" in names
        assert "implement_feature" in names

    def test_builtin_skills_have_builtin_flag(self):
        from hermes_cli.code.skill_discovery import get_builtin_skills
        for skill in get_builtin_skills():
            assert skill["builtin"] is True
            assert skill["source"] == "builtin"

    def test_discover_skills_empty_dir(self, tmp_path):
        from hermes_cli.code.skill_discovery import discover_workspace_skills
        skills = discover_workspace_skills(tmp_path)
        assert skills == []

    def test_discover_skill_from_folder(self, tmp_path):
        from hermes_cli.code.skill_discovery import discover_workspace_skills

        skill_dir = tmp_path / ".hermes" / "skills" / "my_custom_skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "# My Custom Skill\n\n## Description\nDoes something custom.\n\n## Steps\n1. Step one\n"
        )

        skills = discover_workspace_skills(tmp_path)
        assert len(skills) == 1
        assert skills[0]["name"] == "my_custom_skill"
        assert skills[0]["title"] == "My Custom Skill"
        assert "custom" in skills[0]["description"].lower()
        assert skills[0]["builtin"] is False
        assert skills[0]["source"].startswith("workspace:")

    def test_discover_skill_without_skill_md_ignored(self, tmp_path):
        from hermes_cli.code.skill_discovery import discover_workspace_skills

        # Folder without SKILL.md
        skill_dir = tmp_path / ".hermes" / "skills" / "no_skill_md"
        skill_dir.mkdir(parents=True)
        (skill_dir / "run.sh").write_text("echo hello")

        skills = discover_workspace_skills(tmp_path)
        assert len(skills) == 0

    def test_discover_all_skills_merges_builtin_and_workspace(self, tmp_path):
        from hermes_cli.code.skill_discovery import discover_all_skills

        skill_dir = tmp_path / ".hermes" / "skills" / "custom_workflow"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# Custom Workflow\n\n## Description\nCustom.\n")

        skills = discover_all_skills(workspace_path=tmp_path)
        names = {s["name"] for s in skills}
        # Built-in skills still present
        assert "fix_build" in names
        # Custom skill also present
        assert "custom_workflow" in names

    def test_workspace_skill_overrides_builtin(self, tmp_path):
        from hermes_cli.code.skill_discovery import discover_all_skills

        # Override the fix_build built-in with a custom version
        skill_dir = tmp_path / ".hermes" / "skills" / "fix_build"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "# Fix Build (Custom)\n\n## Description\nCustom build fixer.\n"
        )

        skills = discover_all_skills(workspace_path=tmp_path)
        fix_build = next((s for s in skills if s["name"] == "fix_build"), None)
        assert fix_build is not None
        assert fix_build["builtin"] is False
        assert "Custom" in fix_build["title"]

    def test_skill_discovery_service_list(self, tmp_path):
        from hermes_cli.code.skill_discovery import SkillDiscoveryService
        svc = SkillDiscoveryService()
        skills = svc.list_skills(workspace_path=tmp_path)
        assert len(skills) >= 7  # at least builtin

    def test_skill_discovery_service_get(self):
        from hermes_cli.code.skill_discovery import SkillDiscoveryService
        svc = SkillDiscoveryService()
        skill = svc.get_skill("fix_build")
        assert skill is not None
        assert skill["name"] == "fix_build"

    def test_skill_discovery_service_get_missing(self):
        from hermes_cli.code.skill_discovery import SkillDiscoveryService
        svc = SkillDiscoveryService()
        assert svc.get_skill("nonexistent_skill_xyz") is None

    def test_skill_with_scripts_dir(self, tmp_path):
        from hermes_cli.code.skill_discovery import discover_workspace_skills

        skill_dir = tmp_path / ".hermes" / "skills" / "scripted_skill"
        (skill_dir / "scripts").mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# Scripted\n")
        (skill_dir / "scripts" / "run.sh").write_text("echo hi")

        skills = discover_workspace_skills(tmp_path)
        assert len(skills) == 1
        assert skills[0]["has_scripts"] is True
        assert skills[0]["has_resources"] is False
