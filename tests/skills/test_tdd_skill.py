"""Contract tests for the bundled test-driven-development skill."""

import re
from pathlib import Path
import pytest

from tests.skills._skill_test_utils import parse_frontmatter_and_body, resolve_related_skills_in_repo

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_MD = (
    REPO_ROOT
    / "skills"
    / "software-development"
    / "test-driven-development"
    / "SKILL.md"
)

REQUIRED_SECTIONS = [
    "## When to Use",
    "## The Iron Law",
    "## Red-Green-Refactor Cycle",
    "## Avoid Horizontal Slices",
    "## Why Order Matters",
    "## Common Rationalizations",
    "## Red Flags — STOP and Start Over",
    "## Verification Checklist",
    "## When Stuck",
    "## Hermes Agent Integration",
    "## Testing Anti-Patterns",
]

CYCLE_STEPS = [
    "### RED — Write Failing Test",
    "### Verify RED — Watch It Fail",
    "### GREEN — Minimal Code",
    "### Verify GREEN — Watch It Pass",
    "### REFACTOR — Clean Up",
]


class TestTDDSkillContract:
    """Test that SKILL.md adheres to the project hardline standards."""

    def test_skill_file_exists(self):
        assert SKILL_MD.is_file()

    def test_frontmatter_required_fields(self):
        fm, _ = parse_frontmatter_and_body(SKILL_MD)
        for field in ("name", "description", "version", "author", "license", "platforms"):
            assert field in fm, f"missing frontmatter field: {field}"
        assert fm["name"] == "test-driven-development"
        hermes = fm["metadata"]["hermes"]
        assert hermes["tags"]
        assert "related_skills" in hermes

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

    def test_required_sections_present_and_ordered(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        positions = [body.index(section) for section in REQUIRED_SECTIONS]
        assert positions == sorted(positions), "sections must follow structured sequence"

    def test_red_green_refactor_cycle_steps_complete(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        for step in CYCLE_STEPS:
            assert step in body, f"cycle step missing: {step}"

    def test_iron_law_of_tdd_documented(self):
        _, body = parse_frontmatter_and_body(SKILL_MD)
        assert "NO PRODUCTION CODE WITHOUT A FAILING TEST FIRST" in body
        assert "Tracer Bullets" in body or "tracer bullet" in body.lower()
        assert "Hermes Integration" in body or "Hermes Agent Integration" in body
