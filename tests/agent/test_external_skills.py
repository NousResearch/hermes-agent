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


class TestDuplicateSkillAcrossTiersE2E:
    """E2E for #100715: a builtin skill byte-identical in BOTH the active
    profile's tree and a registered external_dir must stay loadable.

    This exercises the real worker-spawn ``--skills`` resolution path
    (``build_preloaded_skills_prompt`` -> ``_load_skill_payload`` ->
    ``skill_view``) against a temp HERMES_HOME — no mocks on the lookup
    chain. Before the fix, skill_view refused the name as ambiguous,
    every requested skill landed in ``missing``, and the CLI hard-raised
    ``ValueError('Unknown skill(s): ...')``, crash-looping spawned workers
    even though ``skills_list`` showed the skill as enabled.
    """

    def test_preloaded_skills_prompt_loads_duplicated_builtin(self, tmp_path):
        hermes_home = tmp_path / ".hermes"
        external_dir = tmp_path / "shared-skills"
        hermes_home.mkdir()
        local_skills = hermes_home / "skills"
        local_skills.mkdir()
        (hermes_home / "config.yaml").write_text(
            f"skills:\n  external_dirs:\n    - {external_dir}\n"
        )

        body = (
            "---\n"
            "name: kanban-worker\n"
            "description: Pitfalls and examples for Hermes Kanban workers.\n"
            "---\n"
            "\n"
            "# Kanban Worker\n\n"
            "Orient, work, heartbeat, complete.\n"
        )
        # The same skill exists in both tiers, byte-identical — mirroring
        # builtin seeds copied into a profile tree that also registers the
        # shared tree as an external_dir.
        for root in (local_skills, external_dir):
            skill_dir = root / "devops" / "kanban-worker"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(body)

        with (
            patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}),
            patch("tools.skills_tool.SKILLS_DIR", local_skills),
        ):
            from agent.skill_commands import build_preloaded_skills_prompt

            prompt, loaded, missing = build_preloaded_skills_prompt(
                ["kanban-worker"]
            )
            from tools.skills_tool import skills_list

            listed = json.loads(skills_list())

        assert missing == [], f"skill resolved as missing: {missing}"
        assert loaded == ["kanban-worker"]
        assert "Kanban Worker" in prompt
        # Enumeration and lookup must agree: the skill the list advertises
        # is the skill the loader serves.
        entry = [s for s in listed["skills"] if s["name"] == "kanban-worker"]
        assert len(entry) == 1
