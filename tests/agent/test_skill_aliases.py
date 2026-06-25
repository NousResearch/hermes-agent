"""Tests for skill alias registration in agent/skill_commands.py."""

from unittest.mock import patch

import pytest

from agent.skill_commands import scan_skill_commands, _get_core_command_names


def _make_skill(skills_dir, name, description="Description for skill.", body="Do the thing.", extra_frontmatter=""):
    """Helper to create a minimal skill directory with SKILL.md."""
    import re as _re
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    content = f"""\
---
name: {name}
description: {description}
{extra_frontmatter}---
# {name}

{body}
"""
    (skill_dir / "SKILL.md").write_text(content)
    return skill_dir


class TestSkillAliases:
    def test_alias_registers_additional_slash_command(self, tmp_path):
        """A SKILL.md with aliases: [pho] registers both /project-handoff and /pho."""
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "project-handoff",
                        extra_frontmatter="aliases: [pho]\n")
            result = scan_skill_commands()

        assert "/project-handoff" in result
        assert "/pho" in result
        # Both entries point to the same skill info
        for key in ("name", "skill_md_path", "skill_dir"):
            assert result["/project-handoff"][key] == result["/pho"][key]

    def test_multiple_aliases_all_registered(self, tmp_path):
        """Multiple aliases (pho, po) all register and point to the same skill."""
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "project-handoff",
                        extra_frontmatter="aliases: [pho, po]\n")
            result = scan_skill_commands()

        assert "/project-handoff" in result
        assert "/pho" in result
        assert "/po" in result
        for alias in ("/pho", "/po"):
            assert result[alias]["name"] == result["/project-handoff"]["name"]

    def test_empty_aliases_list_no_extra_entries(self, tmp_path):
        """aliases: [] — no extra entries registered beyond the main command."""
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "my-skill",
                        extra_frontmatter="aliases: []\n")
            result = scan_skill_commands()

        assert "/my-skill" in result
        assert len(result) == 1

    def test_alias_collision_with_core_command_skipped(self, tmp_path):
        """An alias that collides with a built-in command is skipped with warning."""
        core_names = _get_core_command_names()
        # Pick a real core command that's not normally a skill name
        bad_alias = "new"  # /new is a built-in command
        assert bad_alias in core_names, f"Expected '{bad_alias}' to be a core command"

        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "my-skill",
                        extra_frontmatter=f"aliases: [{bad_alias}]\n")
            with patch("agent.skill_commands.logger") as mock_logger:
                result = scan_skill_commands()

        assert "/my-skill" in result
        assert f"/{bad_alias}" not in result
        mock_logger.warning.assert_any_call(
            "Skill '%s' alias '/%s' collides with a core Hermes command. Skipping.",
            "my-skill", bad_alias,
        )

    def test_alias_collision_with_other_skill_skipped(self, tmp_path):
        """Two skills claiming the same alias: second is skipped with warning."""
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "skill-one",
                        extra_frontmatter="aliases: [shared]\n")
            _make_skill(tmp_path, "skill-two",
                        extra_frontmatter="aliases: [shared]\n")

            with patch("agent.skill_commands.logger") as mock_logger:
                result = scan_skill_commands()

        assert "/skill-one" in result
        assert "/skill-two" in result
        assert "/shared" in result
        # The alias belongs to whichever skill scanned first
        assert result["/shared"]["name"] in ("skill-one", "skill-two")
        mock_logger.warning.assert_any_call(
            "Skill '%s' alias '/%s' collides with existing command from '%s'. Skipping.",
            "skill-two", "shared", "skill-one",
        )

    def test_invalid_alias_entries_silently_skipped(self, tmp_path):
        """None, empty string, and non-string entries in the aliases list are skipped."""
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "my-skill",
                        extra_frontmatter="aliases: [valid, '', !!null, 42]\n")
            result = scan_skill_commands()

        assert "/my-skill" in result
        assert "/valid" in result
        # empty string, null, and integer entries should not create commands
        assert "/" not in result
        # 42 is not a string — skipped silently
        assert "/42" not in result

    def test_alias_description_includes_alias_note(self, tmp_path):
        """Alias entries describe themselves as aliases for the main command."""
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "project-handoff",
                        extra_frontmatter="aliases: [pho]\n")
            result = scan_skill_commands()

        assert "alias" in result["/pho"]["description"].lower()
        assert "/project-handoff" in result["/pho"]["description"]
