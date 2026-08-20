"""Contract tests for the Ponytail Review skill."""

import re
from pathlib import Path


SKILL_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "software-development"
    / "ponytail-review"
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
    assert _frontmatter_field("author").startswith("SeoYeonKim (@westkite1201)")


def test_sections_follow_modern_skill_order():
    sections = [
        "# Ponytail Review Skill",
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


def test_review_is_read_only_and_requires_separate_change_authority():
    assert "This workflow is report-only" in CONTENT
    assert "requires a separate explicit request" in CONTENT
    assert "Do not apply any suggested change unless the user explicitly asks" in CONTENT


def test_review_stays_complexity_only_and_preserves_safety_controls():
    assert "not a correctness, security, or performance review" in CONTENT
    assert "label it `normal-review:`" in CONTENT
    for boundary in ("trust-boundary validation", "data-loss prevention", "accessibility"):
        assert boundary in CONTENT


def test_findings_require_location_tag_cut_and_replacement():
    assert "<file>:L<line>: <tag> <what to cut>. <replacement>." in CONTENT
    assert "one location, one tag, and one replacement" in CONTENT
    assert "No over-engineering findings in this scope" in CONTENT

