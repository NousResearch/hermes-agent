"""Hermetic contract tests for the AI red-team intelligence brief skill."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SKILL = (
    ROOT
    / "optional-skills"
    / "security"
    / "ai-red-team-intel-brief"
    / "SKILL.md"
)
OLD_SKILL = (
    ROOT
    / "skills"
    / "red-teaming"
    / "ai-red-team-intel-brief"
    / "SKILL.md"
)


def _skill_text() -> str:
    return SKILL.read_text(encoding="utf-8")


def _frontmatter() -> str:
    match = re.match(r"\A---\n(.*?)\n---\n", _skill_text(), re.DOTALL)
    assert match, "SKILL.md must begin with YAML frontmatter"
    return match.group(1)


def test_skill_is_optional_security_skill() -> None:
    assert SKILL.is_file()
    assert not OLD_SKILL.exists()


def test_frontmatter_meets_skill_requirements() -> None:
    metadata = _frontmatter()
    description_match = re.search(
        r"^description:\s*(.+)$", metadata, re.MULTILINE
    )
    assert description_match
    description = description_match.group(1)

    assert len(description) <= 60
    assert description.endswith(".")

    for field in ("name", "version", "author", "license", "platforms"):
        assert re.search(rf"^{field}:", metadata, re.MULTILINE)

    assert re.search(r"^metadata:\n\s+hermes:", metadata, re.MULTILINE)
    for field in ("tags", "category", "related_skills"):
        assert re.search(rf"^\s+{field}:", metadata, re.MULTILINE)


def test_required_sections_are_present_in_order() -> None:
    text = _skill_text()
    headings = [
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]

    positions = [text.index(heading) for heading in headings]
    assert positions == sorted(positions)
