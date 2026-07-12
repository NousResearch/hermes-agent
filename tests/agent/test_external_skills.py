"""Tests for external skill directories (skills.external_dirs config)."""

import json
import os
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
def hermes_home(tmp_path):
    """Create a minimal HERMES_HOME with config."""
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

    def test_nonexistent_dir_skipped(self, hermes_home):
        (hermes_home / "config.yaml").write_text(
            "skills:\n  external_dirs:\n    - /nonexistent/path\n"
        )
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

    def test_duplicate_dirs_deduplicated(self, hermes_home, external_skills_dir):
        (hermes_home / "config.yaml").write_text(
            f"skills:\n  external_dirs:\n    - {external_skills_dir}\n    - {external_skills_dir}\n"
        )
        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            from agent.skill_utils import get_external_skills_dirs
            result = get_external_skills_dirs()
        assert len(result) == 1

    def test_local_skills_dir_excluded(self, hermes_home):
        local_skills = hermes_home / "skills"
        (hermes_home / "config.yaml").write_text(
            f"skills:\n  external_dirs:\n    - {local_skills}\n"
        )
        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            from agent.skill_utils import get_external_skills_dirs
            result = get_external_skills_dirs()
        assert result == []

    def test_no_config_file(self, hermes_home):
        # No config.yaml at all
        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            from agent.skill_utils import get_external_skills_dirs
            result = get_external_skills_dirs()
        assert result == []

    def test_string_value_converted_to_list(self, hermes_home, external_skills_dir):
        (hermes_home / "config.yaml").write_text(
            f"skills:\n  external_dirs: {external_skills_dir}\n"
        )
        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            from agent.skill_utils import get_external_skills_dirs
            result = get_external_skills_dirs()
        assert len(result) == 1


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

    def test_default_write_dir_is_discovered_before_external(self, hermes_home, external_skills_dir):
        write_dir = hermes_home / "shared-skills"
        write_dir.mkdir()
        (hermes_home / "config.yaml").write_text(
            "skills:\n"
            "  default_write_dir: shared-skills\n"
            "  external_dirs:\n"
            f"    - {write_dir}\n"
            f"    - {external_skills_dir}\n"
        )
        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            from agent.skill_utils import get_all_skills_dirs, is_external_skill_path
            result = get_all_skills_dirs()
            assert is_external_skill_path(write_dir / "example" / "SKILL.md") is False
        assert result == [hermes_home / "skills", write_dir.resolve(), external_skills_dir.resolve()]

    def test_nested_default_write_dir_overrides_external_parent_ownership(self, hermes_home):
        external_parent = hermes_home / "shared"
        write_dir = external_parent / "hermes-owned"
        write_dir.mkdir(parents=True)
        (hermes_home / "config.yaml").write_text(
            "skills:\n"
            f"  default_write_dir: {write_dir}\n"
            "  external_dirs:\n"
            f"    - {external_parent}\n"
        )
        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            from agent.skill_utils import is_external_skill_path
            assert is_external_skill_path(write_dir / "example" / "SKILL.md") is False
            assert is_external_skill_path(external_parent / "readonly" / "SKILL.md") is True

    def test_nested_external_dir_overrides_default_write_parent_ownership(self, hermes_home):
        write_dir = hermes_home / "shared"
        readonly_dir = write_dir / "readonly"
        readonly_dir.mkdir(parents=True)
        (hermes_home / "config.yaml").write_text(
            "skills:\n"
            f"  default_write_dir: {write_dir}\n"
            "  external_dirs:\n"
            f"    - {readonly_dir}\n"
        )
        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            from agent.skill_utils import is_external_skill_path
            assert is_external_skill_path(write_dir / "owned" / "SKILL.md") is False
            assert is_external_skill_path(readonly_dir / "example" / "SKILL.md") is True

    def test_unresolvable_default_write_dir_returns_create_error(self, hermes_home):
        loop = hermes_home / "loop"
        try:
            loop.symlink_to(loop)
        except OSError:
            pytest.skip("Symlinks not supported")
        (hermes_home / "config.yaml").write_text(
            f"skills:\n  default_write_dir: {loop}\n"
        )
        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            from agent.skill_utils import get_all_skills_dirs, get_skill_write_dir
            assert get_all_skills_dirs() == [hermes_home / "skills"]
            path, error = get_skill_write_dir()
        assert path is None
        assert error is not None
        assert "could not be resolved" in error


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

    def test_default_write_dir_skills_are_found(self, hermes_home):
        write_dir = hermes_home / "shared-skills"
        skill_dir = write_dir / "shared-created"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: shared-created\ndescription: Shared created skill.\n---\n\nBody.\n"
        )
        (hermes_home / "config.yaml").write_text(
            "skills:\n  default_write_dir: shared-skills\n"
        )
        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            from tools.skills_tool import _find_all_skills
            skills = _find_all_skills()
        assert "shared-created" in {skill["name"] for skill in skills}

    def test_skills_list_uses_default_write_dir_without_local_root(self, tmp_path):
        hermes_home = tmp_path / "hermes"
        write_dir = tmp_path / "shared-skills"
        skill_dir = write_dir / "shared-listed"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: shared-listed\ndescription: Shared listed skill.\n---\n\nBody.\n"
        )
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            f"skills:\n  default_write_dir: {write_dir}\n"
        )
        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            from tools.skills_tool import skills_list
            result = json.loads(skills_list())
        assert result["success"] is True
        assert "shared-listed" in {skill["name"] for skill in result["skills"]}


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
