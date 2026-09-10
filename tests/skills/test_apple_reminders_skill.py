"""Contract and workflow tests for the apple-reminders skill."""

import re
from pathlib import Path
import pytest

from tests.skills._skill_test_utils import parse_frontmatter_and_body, resolve_related_skills_in_repo

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_MD = REPO_ROOT / "skills" / "apple" / "apple-reminders" / "SKILL.md"

REQUIRED_SECTIONS = [
    "## Prerequisites",
    "## When to Use",
    "## When NOT to Use",
    "## Quick Reference",
    "## Due Time vs Alarm / Early Nudge",
    "## Date Formats",
    "## Rules",
]

REMINDCTL_COMMANDS = [
    "remindctl today",
    "remindctl tomorrow",
    "remindctl week",
    "remindctl overdue",
    "remindctl all",
    "remindctl list",
    "remindctl add",
    "remindctl complete",
    "remindctl delete",
]


class TestAppleRemindersContract:
    """Test that apple-reminders adheres to the project hardline standards."""

    def test_skill_file_exists(self):
        assert SKILL_MD.is_file()

    def test_frontmatter_required_fields(self):
        fm, _ = parse_frontmatter_and_body(SKILL_MD)
        for field in ("name", "description", "version", "author", "license", "platforms"):
            assert field in fm, f"missing frontmatter field: {field}"
        assert fm["name"] == "apple-reminders"
        assert fm["platforms"] == ["macos"], "skill is macOS-specific"
        hermes = fm["metadata"]["hermes"]
        assert hermes["tags"]

    def test_prerequisites_commands(self):
        fm, _ = parse_frontmatter_and_body(SKILL_MD)
        assert fm.get("prerequisites", {}).get("commands") == ["remindctl"]

    def test_description_hardline(self):
        fm, _ = parse_frontmatter_and_body(SKILL_MD)
        desc = fm["description"]
        assert len(desc) <= 60, f"description is {len(desc)} chars; max allowed is 60"
        assert desc.endswith("."), "description must end with a period"
        assert not re.search(
            r"\b(powerful|comprehensive|seamless|revolutionary|cutting-edge|state-of-the-art)\b",
            desc,
            re.I,
        )

    def test_no_machine_local_paths(self):
        content = SKILL_MD.read_text(encoding="utf-8")
        assert "/home/" not in content
        assert not re.search(r"[A-Z]:\\\\Users", content)


class TestAppleRemindersContentAndStructure:
    """Test documentation structure, due vs alarm distinction, and output formats."""

    def test_required_sections_present_and_ordered(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        positions = [body.index(section) for section in REQUIRED_SECTIONS]
        assert positions == sorted(positions), "sections must follow structured order"

    def test_remindctl_commands_documented(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        for cmd in REMINDCTL_COMMANDS:
            assert cmd in body, f"command {cmd} must be documented"

    def test_due_vs_alarm_early_nudge_distinction(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        assert "--due" in body
        assert "--alarm" in body
        assert "dueDate" in body
        assert "alarmDate" in body

    def test_output_formats_documented(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        assert "--json" in body
        assert "--plain" in body
        assert "--quiet" in body

    def test_rules_clarify_agent_cronjob_vs_reminders(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        assert "cronjob" in body.lower(), "must clarify agent cronjob vs Apple Reminders"
