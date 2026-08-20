"""Contract tests for the Ponytail skill."""

import re
from pathlib import Path


SKILL_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "software-development"
    / "ponytail"
    / "SKILL.md"
)
CONTENT = SKILL_PATH.read_text(encoding="utf-8")


def _frontmatter_field(name: str) -> str:
    match = re.search(rf"^{name}: (.+)$", CONTENT, re.MULTILINE)
    assert match, f"missing {name} frontmatter"
    return match.group(1)


def test_frontmatter_is_listing_safe_and_credits_human_first():
    description = _frontmatter_field("description")
    assert len(description) <= 60
    assert description.endswith(".")
    assert "\n" not in description
    assert _frontmatter_field("author").startswith("SeoYeonKim (@westkite1201)")


def test_sections_follow_modern_skill_order():
    sections = [
        "# Ponytail Skill",
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]
    positions = [CONTENT.index(section) for section in sections]
    assert positions == sorted(positions)


def test_minimalism_is_explicit_opt_in_not_a_debugging_shortcut():
    assert "only when the user explicitly asks" in CONTENT
    assert "Do not auto-run Ponytail" in CONTENT
    assert "Do not use it to skip reading" in CONTENT
    assert "confirmed root cause" in CONTENT


def test_safety_and_verification_are_protected_from_simplification():
    protected = (
        "trust-boundary validation",
        "authentication",
        "data loss",
        "accessibility",
        "migrations",
        "tests for non-trivial",
    )
    for boundary in protected:
        assert boundary in CONTENT
    assert "Run the repository's smallest relevant test" in CONTENT

