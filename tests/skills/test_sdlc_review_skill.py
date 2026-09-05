"""Contract tests for the bundled SDLC review skill."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SKILL_MD = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "devops"
    / "sdlc-review"
    / "SKILL.md"
)
REQUIRED_SECTIONS = [
    "## When to Use",
    "## Prerequisites",
    "## How to Run",
    "## Quick Reference",
    "## Review Lenses",
    "## Procedure",
    "## Pitfalls",
    "## Verification",
]
REVIEW_ACTIONS = {
    "kanban_show",
    "kanban_comment",
    "kanban_complete",
    "kanban_request_changes",
    "kanban_block",
}


@pytest.fixture(scope="module")
def skill_text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


def _frontmatter_value(text: str, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}:\s*(.+)$", text, re.MULTILINE)
    assert match, f"missing frontmatter field: {key}"
    return match.group(1).strip()


def test_frontmatter_meets_hardline_standard(skill_text: str) -> None:
    assert skill_text.startswith("---\n")
    assert _frontmatter_value(skill_text, "name") == "sdlc-review"

    description = _frontmatter_value(skill_text, "description")
    assert len(description) <= 60
    assert description.endswith(".")

    for field in ("version", "author", "license", "platforms"):
        assert _frontmatter_value(skill_text, field)
    assert not _frontmatter_value(skill_text, "author").startswith("Hermes Agent")


def test_body_uses_required_modern_section_order(skill_text: str) -> None:
    assert "# SDLC Review Skill" in skill_text
    positions = [skill_text.index(section) for section in REQUIRED_SECTIONS]
    assert positions == sorted(positions)


@pytest.mark.parametrize("tool_name", sorted(REVIEW_ACTIONS))
def test_skill_documents_native_review_actions(
    skill_text: str,
    tool_name: str,
) -> None:
    assert f"`{tool_name}`" in skill_text


def test_verdicts_route_through_distinct_terminal_actions(skill_text: str) -> None:
    quick_reference = skill_text.split("## Quick Reference", 1)[1].split(
        "## Review Lenses", 1
    )[0]
    assert "Approve" in quick_reference and "`kanban_complete`" in quick_reference
    assert "Request changes" in quick_reference
    assert "`kanban_request_changes`" in quick_reference
    assert "Escalate" in quick_reference and "`kanban_block`" in quick_reference


def test_review_lenses_vary_per_round(skill_text: str) -> None:
    lenses = skill_text.split("## Review Lenses", 1)[1].split("## Procedure", 1)[0]
    # Round derivation must key off history the reviewer actually sees.
    assert "`changes_requested`" in lenses
    assert "Prior attempts on this task" in lenses
    # One distinct lens per round.
    for lens in ("Artifact", "Execution", "Contract"):
        assert lens in lenses
    # Execution lens must direct empirical verification via the terminal.
    assert "`terminal`" in lenses
    # Fan-out note: parallel reviewers get different briefs.
    assert "`delegate_task`" in lenses
