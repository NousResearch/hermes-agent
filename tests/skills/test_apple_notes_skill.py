"""Contract and workflow tests for the apple-notes skill."""

import re
from pathlib import Path
import pytest

from tests.skills._skill_test_utils import parse_frontmatter_and_body, resolve_related_skills_in_repo

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_MD = REPO_ROOT / "skills" / "apple" / "apple-notes" / "SKILL.md"

REQUIRED_SECTIONS = [
    "## Prerequisites",
    "## When to Use",
    "## When NOT to Use",
    "## Quick Reference",
    "## Limitations",
    "## Rules",
]

MEMO_COMMANDS = [
    "memo notes",
    "memo notes -f",
    "memo notes -s",
    "memo notes -a",
    "memo notes -e",
    "memo notes -d",
    "memo notes -m",
    "memo notes -ex",
]


class TestAppleNotesContract:
    """Test that apple-notes adheres to the project hardline standards."""

    def test_skill_file_exists(self):
        assert SKILL_MD.is_file()

    def test_frontmatter_required_fields(self):
        fm, _ = parse_frontmatter_and_body(SKILL_MD)
        for field in ("name", "description", "version", "author", "license", "platforms"):
            assert field in fm, f"missing frontmatter field: {field}"
        assert fm["name"] == "apple-notes"
        assert fm["platforms"] == ["macos"], "skill is macOS-specific"
        hermes = fm["metadata"]["hermes"]
        assert hermes["tags"]
        assert "related_skills" in hermes

    def test_prerequisites_commands(self):
        fm, _ = parse_frontmatter_and_body(SKILL_MD)
        assert fm.get("prerequisites", {}).get("commands") == ["memo"]

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

    def test_related_skills_resolve_in_repo(self):
        fm, _ = parse_frontmatter_and_body(SKILL_MD)
        missing = resolve_related_skills_in_repo(fm["metadata"]["hermes"]["related_skills"])
        assert not missing, f"related_skills entries do not resolve in repo: {missing}"

    def test_no_machine_local_paths(self):
        content = SKILL_MD.read_text(encoding="utf-8")
        assert "/home/" not in content
        assert not re.search(r"[A-Z]:\\\\Users", content)


class TestAppleNotesContentAndStructure:
    """Test documentation structure, CLI commands, and limitations in apple-notes."""

    def test_required_sections_present_and_ordered(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        positions = [body.index(section) for section in REQUIRED_SECTIONS]
        assert positions == sorted(positions), "sections must follow structured order"

    def test_memo_subcommands_documented(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        for cmd in MEMO_COMMANDS:
            assert cmd in body, f"command {cmd} must be documented"

    def test_editor_environment_discipline(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        assert "$EDITOR" in body, "must document $EDITOR environment variable requirement for -a"

    def test_limitations_and_boundary_rules(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        assert "Limitations" in body
        assert "memory" in body, "must document contrast with internal memory tool"
        assert "obsidian" in body, "must document contrast with obsidian skill"
