"""Tests for skill subcommand metadata extraction via metadata.hermes.ui.

Covers: agent/skill_commands.py scan_skill_commands() extraction of
subcommands dict from SKILL.md frontmatter.
"""

from pathlib import Path
from unittest.mock import patch

import tools.skills_tool as skills_tool_module
from agent.skill_commands import scan_skill_commands


def _make_skill(
    skills_dir, name, frontmatter_extra="", body="Do the thing.", category=None
):
    """Helper to create a minimal skill directory with SKILL.md."""
    if category:
        skill_dir = Path(skills_dir) / category / name
    else:
        skill_dir = Path(skills_dir) / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md = skill_dir / "SKILL.md"
    fm = [
        "---",
        f"name: {name}",
        f"description: Skill {name}.",
        "version: 1.0.0",
        "author: test",
        "license: MIT",
    ]
    if frontmatter_extra:
        fm.append(frontmatter_extra)
    fm.extend(["---", "", body, ""])
    skill_md.write_text("\n".join(fm) + "\n")
    return skill_md


class TestScanSkillCommandsUi:
    """metadata.hermes.ui.subcommands is extracted into _skill_commands dict."""

    def test_skill_without_ui_section_has_empty_subcommands(self, tmp_path):
        """Skills with no ui section should still register with empty subcommands."""
        with patch.object(skills_tool_module, "SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "plain-skill")
            result = scan_skill_commands()
        assert "/plain-skill" in result
        assert result["/plain-skill"]["subcommands"] == {}

    def test_skill_with_ui_subcommands_dict(self, tmp_path):
        """A skill declaring subcommands as a dict gets them into the dict entry."""
        ui_extra = (
            "metadata:\n"
            "  hermes:\n"
            "    ui:\n"
            "      subcommands:\n"
            "        list: Show all items in a table\n"
            "        find: Search by name or description\n"
        )
        with patch.object(skills_tool_module, "SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "my-cmd", frontmatter_extra=ui_extra)
            result = scan_skill_commands()
        assert "/my-cmd" in result
        sc = result["/my-cmd"]["subcommands"]
        assert sc == {"list": "Show all items in a table", "find": "Search by name or description"}

    def test_subcommands_empty_dict_when_ui_present_but_no_subcommands_key(
        self, tmp_path
    ):
        """Skill with metadata.hermes.ui but no subcommands key gets empty dict."""
        ui_extra = (
            "metadata:\n"
            "  hermes:\n"
            "    ui:\n"
            "      some_other_field: value\n"
        )
        with patch.object(skills_tool_module, "SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "no-subcommands", frontmatter_extra=ui_extra)
            result = scan_skill_commands()
        assert result["/no-subcommands"]["subcommands"] == {}

    def test_subcommands_deeply_nested_ui_path(self, tmp_path):
        """The ui key must be under metadata.hermes — correct nesting works."""
        ui_extra = (
            "metadata:\n"
            "  hermes:\n"
            "    ui:\n"
            "      subcommands:\n"
            "        run: Execute something\n"
        )
        with patch.object(skills_tool_module, "SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "nested-ui", frontmatter_extra=ui_extra)
            result = scan_skill_commands()
        assert "/nested-ui" in result
        assert result["/nested-ui"]["subcommands"] == {"run": "Execute something"}

    def test_subcommands_wrong_type_is_list_falls_back_to_empty(self, tmp_path):
        """subcommands declared as a list (not dict) should be handled gracefully."""
        ui_extra = (
            "metadata:\n"
            "  hermes:\n"
            "    ui:\n"
            "      subcommands:\n"
            "        - list\n"
            "        - find\n"
        )
        with patch.object(skills_tool_module, "SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "list-subcommands", frontmatter_extra=ui_extra)
            result = scan_skill_commands()
        assert result["/list-subcommands"]["subcommands"] == {}