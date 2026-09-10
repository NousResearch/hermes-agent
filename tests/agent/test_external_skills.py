"""Tests for external skill directories (skills.external_dirs config)."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def external_skills_dir(tmp_path):
    """Create a temp dir with a sample external skill."""
    ext_dir = tmp_path / "external-skills"
    skill_dir = ext_dir / "my-external-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: my-external-skill\ndescription: A skill from an external directory\n---\n\n# My External Skill\n\nDo external things.\n"
    )
    return ext_dir


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Create a minimal HERMES_HOME with config."""
    user_home = tmp_path / "home"
    user_home.mkdir()
    monkeypatch.setenv("HOME", str(user_home))
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "skills").mkdir()
    return home


class TestGetExternalSkillsDirs:
    def test_empty_config(self, hermes_home):
        (hermes_home / "config.yaml").write_text("skills:\n  external_dirs: []\n")
        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            from agent.skill_utils import get_external_skills_dirs
            result = get_external_skills_dirs()
        assert result == []


    def test_valid_dir_returned(self, hermes_home, external_skills_dir):
        (hermes_home / "config.yaml").write_text(
            f"skills:\n  external_dirs:\n    - {external_skills_dir}\n"
        )
        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            from agent.skill_utils import get_external_skills_dirs
            result = get_external_skills_dirs()
        assert len(result) == 1
        assert result[0] == external_skills_dir.resolve()

    def test_shared_user_dir_is_not_returned_as_external(self, hermes_home):
        shared = Path.home() / ".agents" / "skills"
        shared.mkdir(parents=True)
        (hermes_home / "config.yaml").write_text(
            f"skills:\n  external_dirs:\n    - {shared}\n"
        )
        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            from agent.skill_utils import get_external_skills_dirs

            result = get_external_skills_dirs()

        assert result == []






class TestGetAllSkillsDirs:
    def test_local_always_first(self, hermes_home, external_skills_dir):
        (hermes_home / "config.yaml").write_text(
            f"skills:\n  external_dirs:\n    - {external_skills_dir}\n"
        )
        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            from agent.skill_utils import get_all_skills_dirs
            result = get_all_skills_dirs()
        assert result[0] == hermes_home / "skills"
        assert result[1] == external_skills_dir.resolve()

    def test_shared_user_dir_between_create_dir_and_external(
        self, hermes_home, external_skills_dir
    ):
        shared = Path.home() / ".agents" / "skills"
        shared.mkdir(parents=True)
        create_dir = hermes_home / "created-skills"
        create_dir.mkdir()
        (hermes_home / "config.yaml").write_text(
            "skills:\n"
            f"  create_dir: {create_dir}\n"
            "  external_dirs:\n"
            f"    - {external_skills_dir}\n"
        )
        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            from agent.skill_utils import get_all_skills_dirs

            result = get_all_skills_dirs()

        assert result == [
            hermes_home / "skills",
            create_dir.resolve(),
            shared.resolve(),
            external_skills_dir.resolve(),
        ]


class TestExternalSkillsInFindAll:
    def test_external_skills_found(self, hermes_home, external_skills_dir):
        (hermes_home / "config.yaml").write_text(
            f"skills:\n  external_dirs:\n    - {external_skills_dir}\n"
        )
        local_skills = hermes_home / "skills"
        with (
            patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}),
            patch("tools.skills_tool.SKILLS_DIR", local_skills),
        ):
            from tools.skills_tool import _find_all_skills
            skills = _find_all_skills()
        names = [s["name"] for s in skills]
        assert "my-external-skill" in names

    def test_local_takes_precedence(self, hermes_home, external_skills_dir):
        """If the same skill name exists locally and externally, local wins."""
        local_skills = hermes_home / "skills"
        local_skill = local_skills / "my-external-skill"
        local_skill.mkdir(parents=True)
        (local_skill / "SKILL.md").write_text(
            "---\nname: my-external-skill\ndescription: Local version\n---\n\nLocal.\n"
        )
        (hermes_home / "config.yaml").write_text(
            f"skills:\n  external_dirs:\n    - {external_skills_dir}\n"
        )
        with (
            patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}),
            patch("tools.skills_tool.SKILLS_DIR", local_skills),
        ):
            from tools.skills_tool import _find_all_skills
            skills = _find_all_skills()
        matching = [s for s in skills if s["name"] == "my-external-skill"]
        assert len(matching) == 1
        assert matching[0]["description"] == "Local version"

    def test_shared_user_skills_are_found(self, hermes_home):
        shared = Path.home() / ".agents" / "skills"
        shared_skill = shared / "shared-skill"
        shared_skill.mkdir(parents=True)
        (shared_skill / "SKILL.md").write_text(
            "---\nname: shared-skill\ndescription: Shared version\n---\n\nShared.\n"
        )
        (hermes_home / "config.yaml").write_text("skills:\n  external_dirs: []\n")
        local_skills = hermes_home / "skills"
        with (
            patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}),
            patch("tools.skills_tool.SKILLS_DIR", local_skills),
        ):
            from tools.skills_tool import _find_all_skills

            skills = _find_all_skills()

        assert any(s["name"] == "shared-skill" for s in skills)

    def test_local_takes_precedence_over_shared_user_skill(self, hermes_home):
        shared = Path.home() / ".agents" / "skills"
        shared_skill = shared / "shared-skill"
        shared_skill.mkdir(parents=True)
        (shared_skill / "SKILL.md").write_text(
            "---\nname: shared-skill\ndescription: Shared version\n---\n\nShared.\n"
        )
        local_skills = hermes_home / "skills"
        local_skill = local_skills / "shared-skill"
        local_skill.mkdir(parents=True)
        (local_skill / "SKILL.md").write_text(
            "---\nname: shared-skill\ndescription: Local version\n---\n\nLocal.\n"
        )
        (hermes_home / "config.yaml").write_text("skills:\n  external_dirs: []\n")
        with (
            patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}),
            patch("tools.skills_tool.SKILLS_DIR", local_skills),
        ):
            from tools.skills_tool import _find_all_skills

            skills = _find_all_skills()

        matching = [s for s in skills if s["name"] == "shared-skill"]
        assert len(matching) == 1
        assert matching[0]["description"] == "Local version"


class TestExternalSkillView:
    def test_skill_view_finds_external(self, hermes_home, external_skills_dir):
        (hermes_home / "config.yaml").write_text(
            f"skills:\n  external_dirs:\n    - {external_skills_dir}\n"
        )
        local_skills = hermes_home / "skills"
        with (
            patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}),
            patch("tools.skills_tool.SKILLS_DIR", local_skills),
        ):
            from tools.skills_tool import skill_view
            result = json.loads(skill_view("my-external-skill"))
        assert result["success"] is True
        assert "external things" in result["content"]

    def test_skill_view_finds_shared_user_skill(self, hermes_home):
        shared = Path.home() / ".agents" / "skills"
        shared_skill = shared / "shared-skill"
        shared_skill.mkdir(parents=True)
        (shared_skill / "SKILL.md").write_text(
            "---\nname: shared-skill\ndescription: Shared version\n---\n\nShared body.\n"
        )
        (hermes_home / "config.yaml").write_text("skills:\n  external_dirs: []\n")
        local_skills = hermes_home / "skills"
        with (
            patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}),
            patch("tools.skills_tool.SKILLS_DIR", local_skills),
        ):
            from tools.skills_tool import skill_view

            result = json.loads(skill_view("shared-skill"))

        assert result["success"] is True
        assert "Shared body" in result["content"]
