"""Contract and workflow tests for the findmy skill."""

import re
from pathlib import Path
import pytest

from tests.skills._skill_test_utils import parse_frontmatter_and_body, resolve_related_skills_in_repo

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_MD = REPO_ROOT / "skills" / "apple" / "findmy" / "SKILL.md"

REQUIRED_SECTIONS = [
    "## Prerequisites",
    "## When to Use",
    "## Method 1: AppleScript + Screenshot (Basic)",
    "## Method 2: Peekaboo UI Automation (Recommended)",
    "## Workflow: Track AirTag Location Over Time",
    "## Limitations",
    "## Rules",
]


class TestFindMyContract:
    """Test that findmy adheres to the project hardline standards."""

    def test_skill_file_exists(self):
        assert SKILL_MD.is_file()

    def test_frontmatter_required_fields(self):
        fm, _ = parse_frontmatter_and_body(SKILL_MD)
        for field in ("name", "description", "version", "author", "license", "platforms"):
            assert field in fm, f"missing frontmatter field: {field}"
        assert fm["name"] == "findmy"
        assert fm["platforms"] == ["macos"], "skill is macOS-specific"
        hermes = fm["metadata"]["hermes"]
        assert hermes["tags"]

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


class TestFindMyContentAndStructure:
    """Test UI automation methods, AirTag tracking loop, and privacy rules in findmy."""

    def test_required_sections_present_and_ordered(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        positions = [body.index(section) for section in REQUIRED_SECTIONS]
        assert positions == sorted(positions), "sections must follow structured order"

    def test_both_automation_methods_documented(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        # Method 1
        assert "osascript" in body
        assert "screencapture" in body
        assert "vision_analyze" in body
        # Method 2
        assert "peekaboo" in body

    def test_airtag_foreground_tracking_discipline(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        assert "AirTags only update" in body or "AirTag only updates" in body
        assert "foreground" in body

    def test_privacy_rules_documented(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        assert "Respect privacy" in body
        assert "only track devices/items the user owns" in body
