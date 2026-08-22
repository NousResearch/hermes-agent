"""Contract tests for the bundled skill-behavior-validation skill."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SKILL_MD = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "software-development"
    / "skill-behavior-validation"
    / "SKILL.md"
)
REQUIRED_SECTIONS = [
    "## When to Use",
    "## Prerequisites",
    "## How to Run",
    "## Quick Reference",
    "## Procedure",
    "## Pitfalls",
    "## Verification",
]
TRIGGER_CATEGORIES = [
    "priority",
    "safety",
    "classification",
    "evidence",
    "routing",
    "remediation",
    "completion",
]


@pytest.fixture(scope="module")
def skill_text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


def _frontmatter_value(text: str, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}:\s*(.+)$", text, re.MULTILINE)
    assert match, f"missing frontmatter field: {key}"
    return match.group(1).strip()


def test_frontmatter_meets_hardline_standard(skill_text: str) -> None:
    assert skill_text.startswith("---\n")
    assert _frontmatter_value(skill_text, "name") == "skill-behavior-validation"

    description = _frontmatter_value(skill_text, "description")
    assert len(description) <= 60
    assert description.endswith(".")

    for field in ("version", "author", "license", "platforms"):
        assert _frontmatter_value(skill_text, field)


def test_body_uses_required_modern_section_order(skill_text: str) -> None:
    assert "# Skill Behavior Validation Skill" in skill_text
    positions = [skill_text.index(section) for section in REQUIRED_SECTIONS]
    assert positions == sorted(positions)


def test_when_to_use_lists_behavioral_triggers_and_skip_rule(skill_text: str) -> None:
    when_to_use = skill_text.split("## When to Use", 1)[1].split(
        "## Prerequisites", 1
    )[0]
    for category in TRIGGER_CATEGORIES:
        assert category in when_to_use.lower(), f"missing trigger category: {category}"
    assert "Skip for" in when_to_use
    assert "typo" in when_to_use.lower()


def test_procedure_covers_all_eight_steps(skill_text: str) -> None:
    procedure = skill_text.split("## Procedure", 1)[1].split("## Pitfalls", 1)[0]
    expected_step_titles = [
        "State the failure and the intended change",
        "Search the full instruction surface for conflicts",
        "Replay the prior case chronologically",
        "Test a negative control",
        "Verify the decision changed, not just the words",
        "Get an independent adversarial review",
        "Patch and replay until clean",
        "Preserve a concise validation artifact",
    ]
    for i, title in enumerate(expected_step_titles, start=1):
        assert f"### {i}. {title}" in procedure


def test_review_verdicts_are_exhaustive(skill_text: str) -> None:
    procedure = skill_text.split("## Procedure", 1)[1].split("## Pitfalls", 1)[0]
    for verdict in ("PASS", "PARTIAL", "FAIL"):
        assert f"**{verdict}**" in procedure
    # Never claim effectiveness while status is PARTIAL.
    assert "PARTIAL" in procedure and "effective" in procedure.lower()


def test_uses_delegate_task_for_independent_review(skill_text: str) -> None:
    assert "`delegate_task`" in skill_text


def test_pitfalls_warn_against_tautological_replay(skill_text: str) -> None:
    pitfalls = skill_text.split("## Pitfalls", 1)[1].split("## Verification", 1)[0]
    assert "Tautological replay" in pitfalls
    assert "runtime enforcement" in pitfalls.lower()


def test_verification_checklist_matches_acceptance_criteria(skill_text: str) -> None:
    verification = skill_text.split("## Verification", 1)[1]
    assert "negative control" in verification.lower()
    assert "`PARTIAL`" in verification
    assert "`PASS`" in verification
    assert "validation artifact" in verification.lower()
