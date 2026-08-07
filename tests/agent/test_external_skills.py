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


    def test_valid_dir_returned(self, hermes_home, external_skills_dir):
        (hermes_home / "config.yaml").write_text(
            f"skills:\n  external_dirs:\n    - {external_skills_dir}\n"
        )
        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            from agent.skill_utils import get_external_skills_dirs
            result = get_external_skills_dirs()
        assert len(result) == 1
        assert result[0] == external_skills_dir.resolve()

    def test_json_array_scalar_loads_multiple_dirs(self, hermes_home, tmp_path):
        first = tmp_path / "external-one"
        second = tmp_path / "external-two"
        first.mkdir()
        second.mkdir()
        encoded = json.dumps([str(first), str(second)])
        (hermes_home / "config.yaml").write_text(
            f"skills:\n  external_dirs: '{encoded}'\n"
        )

        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            from agent.skill_utils import (
                _external_dirs_cache_clear,
                get_external_skills_dirs,
            )

            _external_dirs_cache_clear()
            result = get_external_skills_dirs()

        assert result == [first.resolve(), second.resolve()]

    def test_malformed_json_scalar_reports_once(self, hermes_home, capsys):
        (hermes_home / "config.yaml").write_text(
            "skills:\n  external_dirs: '[\"/tmp/one\",]'\n"
        )

        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            from agent.skill_utils import (
                _external_dirs_cache_clear,
                get_external_skills_dirs,
            )

            _external_dirs_cache_clear()
            assert get_external_skills_dirs() == []
            assert get_external_skills_dirs() == []

        err = capsys.readouterr().err
        assert err.count("Invalid skills.external_dirs") == 1
        assert "not valid JSON" in err
        assert "hermes config set skills.external_dirs" in err

    def test_non_list_type_reports_configuration_error(self, hermes_home, capsys):
        (hermes_home / "config.yaml").write_text(
            "skills:\n  external_dirs:\n    unexpected: mapping\n"
        )

        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            from agent.skill_utils import (
                _external_dirs_cache_clear,
                get_external_skills_dirs,
            )

            _external_dirs_cache_clear()
            assert get_external_skills_dirs() == []

        err = capsys.readouterr().err
        assert "Invalid skills.external_dirs" in err
        assert "got dict" in err

    def test_expansion_relative_paths_and_duplicates_are_preserved(
        self, hermes_home, tmp_path
    ):
        relative_dir = hermes_home / "relative-skills"
        env_dir = tmp_path / "environment-skills"
        fake_home = tmp_path / "user-home"
        tilde_dir = fake_home / "shared-skills"
        for path in (relative_dir, env_dir, tilde_dir):
            path.mkdir(parents=True)

        (hermes_home / "config.yaml").write_text(
            "skills:\n"
            "  external_dirs:\n"
            "    - relative-skills\n"
            "    - ${EXTERNAL_SKILLS_TEST_DIR}\n"
            "    - ~/shared-skills\n"
            "    - relative-skills\n"
            f"    - {hermes_home / 'skills'}\n"
        )

        with patch.dict(
            os.environ,
            {
                "HERMES_HOME": str(hermes_home),
                "EXTERNAL_SKILLS_TEST_DIR": str(env_dir),
                "HOME": str(fake_home),
            },
        ):
            from agent.skill_utils import (
                _external_dirs_cache_clear,
                get_external_skills_dirs,
            )

            _external_dirs_cache_clear()
            result = get_external_skills_dirs()

        assert result == [
            relative_dir.resolve(),
            env_dir.resolve(),
            tilde_dir.resolve(),
        ]

    def test_valid_nonexistent_path_is_not_reported_as_malformed(
        self, hermes_home, capsys
    ):
        (hermes_home / "config.yaml").write_text(
            "skills:\n  external_dirs:\n    - missing-skills\n"
        )

        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            from agent.skill_utils import (
                _external_dirs_cache_clear,
                get_external_skills_dirs,
            )

            _external_dirs_cache_clear()
            assert get_external_skills_dirs() == []

        assert "Invalid skills.external_dirs" not in capsys.readouterr().err


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
