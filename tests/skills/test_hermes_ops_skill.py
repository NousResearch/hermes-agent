"""Contract tests for the built-in hermes-ops skill."""

from __future__ import annotations

import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SKILL = REPO / "skills" / "autonomous-ai-agents" / "hermes-ops" / "SKILL.md"
REQUIRED_SECTIONS = [
    "## When to Use",
    "## Prerequisites",
    "## How to Run",
    "## Quick Reference",
    "## Procedure",
    "## Pitfalls",
    "## Verification",
]


def _frontmatter_and_body() -> tuple[dict[str, str], str, str]:
    content = SKILL.read_text(encoding="utf-8")
    match = re.match(r"\A---\n(?P<frontmatter>.*?)\n---\n(?P<body>.*)\Z", content, re.S)
    assert match, "SKILL.md must contain closed YAML frontmatter"
    frontmatter = {}
    for line in match.group("frontmatter").splitlines():
        if line and not line[0].isspace() and ":" in line:
            key, value = line.split(":", 1)
            frontmatter[key] = value.strip().strip('"')
    return frontmatter, match.group("frontmatter"), match.group("body")


def test_skill_file_and_frontmatter_contract():
    assert SKILL.is_file()
    frontmatter, raw_frontmatter, _ = _frontmatter_and_body()
    for field in ("name", "description", "version", "author", "license", "platforms"):
        assert field in frontmatter, f"missing frontmatter field: {field}"
    assert frontmatter["name"] == "hermes-ops"
    assert len(frontmatter["description"]) <= 60
    assert frontmatter["description"].endswith(".")
    assert "  hermes:" in raw_frontmatter
    for field in ("tags", "category", "related_skills", "config"):
        assert re.search(rf"^    {field}:", raw_frontmatter, re.M), (
            f"missing metadata.hermes.{field}"
        )


def test_required_sections_are_present_in_order():
    _, _, body = _frontmatter_and_body()
    positions = [body.find(section) for section in REQUIRED_SECTIONS]
    assert all(position >= 0 for position in positions)
    assert positions == sorted(positions)


def test_skill_teaches_profile_cli_boundary_and_grill_me():
    _, _, body = _frontmatter_and_body()
    assert "hermes -p NAME config set" in body
    assert "hermes -p NAME skills" in body
    assert "hermes -p hermes-ops" in body
    assert "grill-me" in body
    assert "Nunca uses" in body
