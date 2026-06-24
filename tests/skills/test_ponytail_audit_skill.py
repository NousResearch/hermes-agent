"""Contract tests for the Ponytail Audit skill."""

import re
from pathlib import Path


SKILL_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "software-development"
    / "ponytail-audit"
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
        "# Ponytail Audit Skill",
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


def test_audit_is_scoped_and_report_only():
    assert "keep the audit there instead of expanding" in CONTENT
    assert "This workflow is report-only" in CONTENT
    assert "Do not implement the findings unless the user explicitly asks" in CONTENT


def test_uncertain_external_surfaces_require_verification():
    assert "Use `verify:` whenever a candidate might be public API" in CONTENT
    assert "Do not present uncertain removal as safe" in CONTENT
    assert "extension points with real external consumers" in CONTENT


def test_audit_preserves_safety_and_ranks_evidence_backed_impact():
    for boundary in ("trust-boundary validation", "data-loss prevention", "accessibility"):
        assert boundary in CONTENT
    assert "Rank findings by likely removable maintenance burden" in CONTENT
    assert "Findings are ranked by likely impact and cite evidence" in CONTENT

