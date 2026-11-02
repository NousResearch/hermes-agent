"""Contract and workflow tests for the imessage skill."""

import re
from pathlib import Path
import pytest

from tests.skills._skill_test_utils import parse_frontmatter_and_body, resolve_related_skills_in_repo

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_MD = REPO_ROOT / "skills" / "apple" / "imessage" / "SKILL.md"

REQUIRED_SECTIONS = [
    "## Prerequisites",
    "## When to Use",
    "## When NOT to Use",
    "## Quick Reference",
    "## Service Options",
    "## Rules",
    "## Example Workflow",
]

IMSG_COMMANDS = [
    "imsg chats",
    "imsg history",
    "imsg send",
    "imsg watch",
]

SERVICE_OPTIONS = [
    "--service imessage",
    "--service sms",
    "--service auto",
]


class TestIMessageContract:
    """Test that imessage adheres to the project hardline standards."""

    def test_skill_file_exists(self):
        assert SKILL_MD.is_file()

    def test_frontmatter_required_fields(self):
        fm, _ = parse_frontmatter_and_body(SKILL_MD)
        for field in ("name", "description", "version", "author", "license", "platforms"):
            assert field in fm, f"missing frontmatter field: {field}"
        assert fm["name"] == "imessage"
        assert fm["platforms"] == ["macos"], "skill is macOS-specific"
        hermes = fm["metadata"]["hermes"]
        assert hermes["tags"]

    def test_prerequisites_commands(self):
        fm, _ = parse_frontmatter_and_body(SKILL_MD)
        assert fm.get("prerequisites", {}).get("commands") == ["imsg"]

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


class TestIMessageContentAndStructure:
    """Test messaging commands, service options, and safety rules in imessage."""

    def test_required_sections_present_and_ordered(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        positions = [body.index(section) for section in REQUIRED_SECTIONS]
        assert positions == sorted(positions), "sections must follow structured order"

    def test_imsg_commands_documented(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        for cmd in IMSG_COMMANDS:
            assert cmd in body, f"command {cmd} must be documented"

    def test_service_options_documented(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        for opt in SERVICE_OPTIONS:
            assert opt in body, f"service option {opt} must be documented"

    def test_safety_and_confirmation_rules(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        assert "Always confirm recipient and message content" in body
        assert "Never send to unknown numbers" in body

    def test_workflow_has_confirmation_step(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        assert "Confirm with user" in body
        assert "Send after confirmation" in body
