"""Frontmatter/structure smoke tests for the system-monitor skill."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

SKILL_DIR = (
    Path(__file__).resolve().parents[2] / "skills" / "devops" / "system-monitor"
)
SKILL_MD = SKILL_DIR / "SKILL.md"


@pytest.fixture(scope="module")
def skill_src() -> str:
    assert SKILL_MD.is_file(), f"missing {SKILL_MD}"
    return SKILL_MD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontmatter(skill_src: str) -> dict:
    m = re.search(r"^---\n(.*?)\n---", skill_src, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    data = yaml.safe_load(m.group(1))
    assert isinstance(data, dict)
    return data


def test_skill_dir_exists() -> None:
    assert SKILL_DIR.is_dir()


def test_name_matches_dir(frontmatter: dict) -> None:
    assert frontmatter["name"] == "system-monitor"


def test_description_under_60_chars(frontmatter: dict) -> None:
    desc = frontmatter["description"]
    assert isinstance(desc, str)
    assert len(desc) <= 60, f"description is {len(desc)} chars: {desc!r}"
    assert desc.endswith("."), f"description must end with a period: {desc!r}"


def test_platforms_linux_only(frontmatter: dict) -> None:
    assert frontmatter.get("platforms") == ["linux"]


def test_license_mit(frontmatter: dict) -> None:
    assert frontmatter.get("license") == "MIT"


def test_author_credits_contributor(frontmatter: dict) -> None:
    author = str(frontmatter.get("author") or "")
    assert "Alex Chen" in author
    assert "l46983284-cpu" in author


def test_related_skills_resolve(frontmatter: dict) -> None:
    related = (
        (frontmatter.get("metadata") or {})
        .get("hermes", {})
        .get("related_skills")
        or []
    )
    assert related, "expected related_skills"
    root = Path(__file__).resolve().parents[2]
    for name in related:
        dirs = [
            p.parent
            for p in (root / "skills").rglob("SKILL.md")
            if p.parent.name == name
        ] + [
            p.parent
            for p in (root / "optional-skills").rglob("SKILL.md")
            if p.parent.name == name
        ]
        assert dirs, f"related skill not found in tree: {name}"


def test_no_escaped_markdown_fences(skill_src: str) -> None:
    # original PR used backslash-escaped fences that render as literal text
    assert r"\`\`\`" not in skill_src
    assert "```bash" in skill_src


def test_modern_section_headings(skill_src: str) -> None:
    for heading in [
        "# System Monitor Skill",
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]:
        assert heading in skill_src, f"missing section: {heading}"


def test_points_at_terminal_tool(skill_src: str) -> None:
    assert "`terminal`" in skill_src
