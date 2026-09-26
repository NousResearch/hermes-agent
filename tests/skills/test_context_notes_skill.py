"""Tests for the context-notes bundled skill."""
import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / "skills" / "software-development" / "context-notes"
SKILL_MD = SKILL_DIR / "SKILL.md"


def _parse_frontmatter(content: str) -> dict:
    assert content.startswith("---"), "SKILL.md must start with ---"
    m = re.search(r"\n---\s*\n", content[3:])
    assert m, "unclosed frontmatter"
    fm = yaml.safe_load(content[3 : m.start() + 3])
    assert isinstance(fm, dict), "frontmatter must be a YAML mapping"
    return fm


@pytest.fixture(scope="module")
def skill_text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontmatter(skill_text: str) -> dict:
    return _parse_frontmatter(skill_text)


def test_skill_file_exists():
    assert SKILL_MD.exists(), f"SKILL.md not found at {SKILL_MD}"


def test_frontmatter_has_required_fields(frontmatter: dict):
    for field in ("name", "description"):
        assert field in frontmatter, f"Missing required frontmatter field: {field}"


def test_name_matches_directory(frontmatter: dict):
    assert frontmatter["name"] == "context-notes"


def test_description_hardline(frontmatter: dict):
    desc = str(frontmatter.get("description") or "")
    assert len(desc) <= 60, f"description is {len(desc)} chars (hardline 60)"
    assert desc.rstrip().endswith("."), "description must end with a period"


def test_non_empty_body(skill_text: str):
    m = re.search(r"\n---\s*\n", skill_text[3:])
    body = skill_text[m.end() :]
    assert body.strip(), "SKILL.md body must not be empty"
