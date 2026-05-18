from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / "software-development" / "karpeslop"


def _frontmatter() -> dict:
    src = (SKILL_DIR / "SKILL.md").read_text()
    m = re.search(r"^---\n(.*?)\n---", src, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


def _body() -> str:
    src = (SKILL_DIR / "SKILL.md").read_text()
    m = re.search(r"^---\n.*?\n---\n(.*)", src, re.DOTALL)
    assert m, "SKILL.md missing body after frontmatter"
    return m.group(1)


def test_skill_dir_exists() -> None:
    assert SKILL_DIR.is_dir(), f"missing skill dir: {SKILL_DIR}"


def test_skill_md_present() -> None:
    assert (SKILL_DIR / "SKILL.md").is_file()


def test_name_matches_dir() -> None:
    fm = _frontmatter()
    assert fm["name"] == "karpeslop"


def test_description_under_60_chars() -> None:
    fm = _frontmatter()
    desc = fm["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars (hardline ≤60): {desc!r}"


def test_description_ends_with_period() -> None:
    fm = _frontmatter()
    assert fm["description"].endswith("."), f"description must end with period: {fm['description']!r}"


def test_author_credits_contributor() -> None:
    fm = _frontmatter()
    assert "CodeDeficient" in fm["author"]


def test_license_mit() -> None:
    fm = _frontmatter()
    assert fm["license"] == "MIT"


def test_platforms_all_desktop() -> None:
    fm = _frontmatter()
    assert set(fm["platforms"]) == {"linux", "macos", "windows"}


def test_no_marketing_words_in_description() -> None:
    fm = _frontmatter()
    forbidden = {"powerful", "comprehensive", "seamless", "advanced"}
    desc_lower = fm["description"].lower()
    for word in forbidden:
        assert word not in desc_lower, f"description contains marketing word '{word}': {fm['description']!r}"


def test_related_skills_present() -> None:
    fm = _frontmatter()
    related = fm["metadata"]["hermes"]["related_skills"]
    assert "requesting-code-review" in related


@pytest.mark.parametrize("section", [
    "## When to Use",
    "## Prerequisites",
    "## How to Run",
    "## Quick Reference",
    "## Procedure",
    "## Pitfalls",
    "## Verification",
])
def test_required_sections_present(section: str) -> None:
    body = _body()
    assert section in body, f"missing required section: {section}"


def test_skill_body_not_empty() -> None:
    body = _body()
    assert len(body.strip()) > 50, "body too short"


def test_file_size_within_limit() -> None:
    src = (SKILL_DIR / "SKILL.md").read_text()
    assert len(src) <= 100_000, f"SKILL.md is {len(src)} chars (hardline ≤100000)"


def test_no_powerful_in_body() -> None:
    src = (SKILL_DIR / "SKILL.md").read_text()
    assert "powerful" not in src.lower(), "SKILL.md body contains marketing word 'powerful'"


def test_no_seamless_in_body() -> None:
    src = (SKILL_DIR / "SKILL.md").read_text()
    assert "seamless" not in src.lower(), "SKILL.md body contains marketing word 'seamless'"
