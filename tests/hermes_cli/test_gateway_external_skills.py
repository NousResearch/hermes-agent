"""Tests that _collect_gateway_skill_entries includes external skills."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def hermes_home(tmp_path):
    """Create a minimal HERMES_HOME with skills directory."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "skills").mkdir()
    return home


@pytest.fixture
def external_skills_dir(tmp_path):
    """Create a temp dir with a sample external skill."""
    ext_dir = tmp_path / "external-skills"
    skill_dir = ext_dir / "morning-briefing"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: morning-briefing\n"
        "description: Daily morning briefing\n---\n\n"
        "# Morning Briefing\n\nBrief the user on the day ahead.\n"
    )
    return ext_dir


@pytest.fixture
def local_skill(hermes_home):
    """Create a local skill under HERMES_HOME/skills."""
    skill_dir = hermes_home / "skills" / "local-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: local-skill\n"
        "description: A local skill\n---\n\n"
        "# Local Skill\n\nDo local things.\n"
    )
    return skill_dir


class TestCollectGatewaySkillEntriesExternal:
    """Verify that external skills from skills.external_dirs appear in gateway
    slash command registration (Fixes #8110)."""

    def test_external_skill_included_in_gateway_entries(
        self, hermes_home, external_skills_dir, local_skill
    ):
        """External skills should be included alongside local skills."""
        (hermes_home / "config.yaml").write_text(
            f"skills:\n  external_dirs:\n    - {external_skills_dir}\n"
        )
        local_skills = hermes_home / "skills"

        with (
            patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}),
            patch("tools.skills_tool.SKILLS_DIR", local_skills),
        ):
            # Clear cached skill commands to force a fresh scan
            from agent import skill_commands as _sc_mod
            _sc_mod._skill_commands.clear()

            from hermes_cli.commands import _collect_gateway_skill_entries
            entries, hidden = _collect_gateway_skill_entries(
                platform="telegram",
                max_slots=100,
                reserved_names=set(),
                desc_limit=40,
            )

        names = [name for name, _desc, _key in entries]
        assert "morning-briefing" in names, (
            "External skill 'morning-briefing' should be registered as a "
            "gateway slash command"
        )
        assert "local-skill" in names, (
            "Local skill should still be registered"
        )

    def test_external_skill_excluded_without_fix(
        self, hermes_home, external_skills_dir
    ):
        """Without external_dirs config, only local skills appear."""
        # No config.yaml → no external dirs
        local_skills = hermes_home / "skills"
        local_dir = local_skills / "only-local"
        local_dir.mkdir(parents=True)
        (local_dir / "SKILL.md").write_text(
            "---\nname: only-local\n"
            "description: Only local\n---\n\nLocal.\n"
        )

        with (
            patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}),
            patch("tools.skills_tool.SKILLS_DIR", local_skills),
        ):
            from agent import skill_commands as _sc_mod
            _sc_mod._skill_commands.clear()

            from hermes_cli.commands import _collect_gateway_skill_entries
            entries, hidden = _collect_gateway_skill_entries(
                platform="telegram",
                max_slots=100,
                reserved_names=set(),
                desc_limit=40,
            )

        names = [name for name, _desc, _key in entries]
        assert "only-local" in names

    def test_hub_skills_still_excluded(self, hermes_home):
        """Hub-installed skills should remain excluded even from local dir."""
        local_skills = hermes_home / "skills"
        hub_dir = local_skills / ".hub" / "hub-skill"
        hub_dir.mkdir(parents=True)
        (hub_dir / "SKILL.md").write_text(
            "---\nname: hub-skill\n"
            "description: A hub skill\n---\n\nHub.\n"
        )

        with (
            patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}),
            patch("tools.skills_tool.SKILLS_DIR", local_skills),
        ):
            from agent import skill_commands as _sc_mod
            _sc_mod._skill_commands.clear()

            from hermes_cli.commands import _collect_gateway_skill_entries
            entries, hidden = _collect_gateway_skill_entries(
                platform="telegram",
                max_slots=100,
                reserved_names=set(),
                desc_limit=40,
            )

        names = [name for name, _desc, _key in entries]
        assert "hub-skill" not in names, (
            "Hub skills should remain excluded from gateway registration"
        )

    def test_path_prefix_boundary_not_confused(self, hermes_home, tmp_path):
        """A dir whose name is a prefix of another should not cause false matches.

        e.g. external_dir '/tmp/my-skills' must NOT match a skill under
        '/tmp/my-skills-extra/'.
        """
        # "my-skills" is the configured external dir
        real_ext = tmp_path / "my-skills"
        skill_dir = real_ext / "real-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: real-skill\n"
            "description: Real skill\n---\n\nReal.\n"
        )

        # "my-skills-extra" is NOT configured — a skill here should NOT appear
        imposter_ext = tmp_path / "my-skills-extra"
        imposter_dir = imposter_ext / "imposter-skill"
        imposter_dir.mkdir(parents=True)
        (imposter_dir / "SKILL.md").write_text(
            "---\nname: imposter-skill\n"
            "description: Imposter\n---\n\nImposter.\n"
        )

        (hermes_home / "config.yaml").write_text(
            f"skills:\n  external_dirs:\n    - {real_ext}\n"
        )
        local_skills = hermes_home / "skills"

        with (
            patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}),
            patch("tools.skills_tool.SKILLS_DIR", local_skills),
        ):
            from agent import skill_commands as _sc_mod
            _sc_mod._skill_commands.clear()

            from hermes_cli.commands import _collect_gateway_skill_entries
            entries, _ = _collect_gateway_skill_entries(
                platform="telegram",
                max_slots=100,
                reserved_names=set(),
                desc_limit=40,
            )

        names = [name for name, _desc, _key in entries]
        assert "real-skill" in names
        assert "imposter-skill" not in names, (
            "Skill from a dir whose name only shares a prefix should be excluded"
        )

    def test_multiple_external_dirs_all_included(self, hermes_home, tmp_path):
        """Skills from multiple configured external dirs should all appear."""
        ext_a = tmp_path / "ext-a"
        (ext_a / "skill-a").mkdir(parents=True)
        (ext_a / "skill-a" / "SKILL.md").write_text(
            "---\nname: skill-a\ndescription: Skill A\n---\n\nA.\n"
        )

        ext_b = tmp_path / "ext-b"
        (ext_b / "skill-b").mkdir(parents=True)
        (ext_b / "skill-b" / "SKILL.md").write_text(
            "---\nname: skill-b\ndescription: Skill B\n---\n\nB.\n"
        )

        (hermes_home / "config.yaml").write_text(
            f"skills:\n  external_dirs:\n    - {ext_a}\n    - {ext_b}\n"
        )
        local_skills = hermes_home / "skills"

        with (
            patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}),
            patch("tools.skills_tool.SKILLS_DIR", local_skills),
        ):
            from agent import skill_commands as _sc_mod
            _sc_mod._skill_commands.clear()

            from hermes_cli.commands import _collect_gateway_skill_entries
            entries, _ = _collect_gateway_skill_entries(
                platform="telegram",
                max_slots=100,
                reserved_names=set(),
                desc_limit=40,
            )

        names = [name for name, _desc, _key in entries]
        assert "skill-a" in names
        assert "skill-b" in names

    def test_empty_skill_md_path_skipped(self, hermes_home):
        """A skill entry with an empty skill_md_path should be silently skipped."""
        local_skills = hermes_home / "skills"
        # Create one valid local skill so scan produces something
        valid_dir = local_skills / "valid-skill"
        valid_dir.mkdir(parents=True)
        (valid_dir / "SKILL.md").write_text(
            "---\nname: valid-skill\n"
            "description: Valid\n---\n\nValid.\n"
        )

        with (
            patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}),
            patch("tools.skills_tool.SKILLS_DIR", local_skills),
        ):
            from agent import skill_commands as _sc_mod
            _sc_mod._skill_commands.clear()

            # Inject a bogus entry with empty skill_md_path after scan
            from agent.skill_commands import get_skill_commands
            cmds = get_skill_commands()
            cmds["/ghost-skill"] = {
                "name": "ghost-skill",
                "description": "Has no path",
                "skill_md_path": "",
                "skill_dir": "",
            }

            from hermes_cli.commands import _collect_gateway_skill_entries
            entries, _ = _collect_gateway_skill_entries(
                platform="telegram",
                max_slots=100,
                reserved_names=set(),
                desc_limit=40,
            )

        names = [name for name, _desc, _key in entries]
        assert "ghost-skill" not in names, (
            "Skill with empty skill_md_path should be skipped"
        )
        assert "valid-skill" in names

    def test_hub_sibling_dir_not_excluded(self, hermes_home):
        """A directory named '.hub-extra' should NOT be excluded by the hub filter.

        Only skills under the actual '.hub/' directory should be excluded.
        """
        local_skills = hermes_home / "skills"

        # This skill is under ".hub-extra", not ".hub" — it should pass
        sibling_dir = local_skills / ".hub-extra" / "sibling-skill"
        sibling_dir.mkdir(parents=True)
        (sibling_dir / "SKILL.md").write_text(
            "---\nname: sibling-skill\n"
            "description: Not a hub skill\n---\n\nSibling.\n"
        )

        # This skill is under the real ".hub" — it should be excluded
        hub_dir = local_skills / ".hub" / "real-hub-skill"
        hub_dir.mkdir(parents=True)
        (hub_dir / "SKILL.md").write_text(
            "---\nname: real-hub-skill\n"
            "description: A real hub skill\n---\n\nHub.\n"
        )

        with (
            patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}),
            patch("tools.skills_tool.SKILLS_DIR", local_skills),
        ):
            from agent import skill_commands as _sc_mod
            _sc_mod._skill_commands.clear()

            from hermes_cli.commands import _collect_gateway_skill_entries
            entries, _ = _collect_gateway_skill_entries(
                platform="telegram",
                max_slots=100,
                reserved_names=set(),
                desc_limit=40,
            )

        names = [name for name, _desc, _key in entries]
        assert "sibling-skill" in names, (
            "Skill under '.hub-extra/' should NOT be excluded by hub filter"
        )
        assert "real-hub-skill" not in names, (
            "Skill under '.hub/' should still be excluded"
        )
