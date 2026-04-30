"""Tests for Code Mode skill discovery bridge."""

import hermes_cli.code.skill_discovery as skill_discovery
from hermes_cli.code.skill_discovery import SkillDiscoveryService, discover_workspace_skills


def test_builtin_skills_are_preserved(monkeypatch, tmp_path):
    fake_dir = tmp_path / "builtin"
    fake_dir.mkdir()
    monkeypatch.setattr(
        skill_discovery,
        "scan_skill_commands",
        lambda: {
            "/fake-skill": {
                "name": "Fake Skill",
                "description": "desc",
                "skill_md_path": str(fake_dir / "SKILL.md"),
                "skill_dir": str(fake_dir),
            }
        },
    )
    skills = skill_discovery.discover_builtin_skills()
    assert isinstance(skills, list)
    assert len(skills) == 1
    assert skills[0]["name"] == "fake-skill"
    assert skills[0]["builtin"] is True


def test_workspace_skill_md_discovery(tmp_path):
    skill_dir = tmp_path / ".hermes" / "skills" / "my_skill"
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "resources").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "# My Skill\n\nCustom description.\n",
        encoding="utf-8",
    )

    skills = discover_workspace_skills(tmp_path)
    assert len(skills) == 1
    assert skills[0]["name"] == "my_skill"
    assert skills[0]["has_scripts"] is True
    assert skills[0]["has_resources"] is True


def test_skill_discovery_service_merges_sources(tmp_path):
    skill_dir = tmp_path / ".hermes" / "skills" / "workspace_skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Workspace Skill\n\nDesc\n", encoding="utf-8")

    service = SkillDiscoveryService()
    merged = service.list_skills(workspace_path=tmp_path)
    names = {item["name"] for item in merged}
    assert "workspace_skill" in names
