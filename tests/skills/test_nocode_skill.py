"""Tests for the nocode answer-only mode skill.

Structural + internal-consistency checks only (stdlib + pytest, no network).
The behavior contract was validated live via /nocode on CLI and Telegram.
"""

import re
from pathlib import Path

import pytest

SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "autonomous-ai-agents"
    / "nocode"
)
SKILL_MD = SKILL_DIR / "SKILL.md"

REQUIRED_SECTIONS = (
    "## When to Use",
    "## Prerequisites",
    "## How to Run",
    "## Quick Reference",
    "## Procedure",
    "## Pitfalls",
    "## Verification",
)


@pytest.fixture(scope="module")
def skill_text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


def test_skill_file_exists():
    assert SKILL_MD.is_file(), f"missing {SKILL_MD}"


def test_frontmatter_present(skill_text: str):
    assert skill_text.startswith("---\n"), "SKILL.md must open with YAML frontmatter"
    assert skill_text.count("---") >= 2, "frontmatter must be delimited by two '---'"


def test_name_and_description_declared(skill_text: str):
    assert re.search(r"^name:\s*nocode\s*$", skill_text, re.MULTILINE), "name field"
    m = re.search(r'^description:\s*"([^"]+)"\s*$', skill_text, re.MULTILINE)
    assert m, "description field required"
    desc = m.group(1)
    assert len(desc) <= 60, "new-skill descriptions must fit the 60-char budget"
    assert desc.startswith("Use when"), "description must start with the trigger phrase"


def test_author_credits_human_contributor_first(skill_text: str):
    m = re.search(r"^author:\s*(.*)$", skill_text, re.MULTILINE)
    assert m, "author field required"
    assert "@BrunoBza" in m.group(1), "human contributor must be credited first"


def test_platforms_declared(skill_text: str):
    m = re.search(r"^platforms:\s*(.*)$", skill_text, re.MULTILINE)
    assert m, "platforms field required"
    for os_name in ("linux", "macos", "windows"):
        assert os_name in m.group(1), f"missing {os_name} in platforms"


def test_required_sections_present(skill_text: str):
    for heading in REQUIRED_SECTIONS:
        assert heading in skill_text, f"missing section: {heading}"


def test_plan_comparison_present(skill_text: str):
    """The /plan vs /nocode decision guidance must survive edits."""
    assert "/plan" in skill_text and "/nocode" in skill_text
    assert ".hermes/plans/" in skill_text


def test_related_skills_resolve_in_repo(skill_text: str):
    m = re.search(r"related_skills:\s*\[([^\]]*)\]", skill_text)
    assert m, "related_skills required"
    related = [name.strip() for name in m.group(1).split(",") if name.strip()]
    assert "plan" in related, "plan must be listed as a related skill"
    plan_skill = (
        Path(__file__).resolve().parents[2]
        / "skills"
        / "software-development"
        / "plan"
        / "SKILL.md"
    )
    assert plan_skill.is_file(), "related_skills must reference in-repo skills only"
