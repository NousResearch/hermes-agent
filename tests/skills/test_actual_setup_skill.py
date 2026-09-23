"""
Smoke tests for the actual-setup optional skill.

Validates:
  - SKILL.md frontmatter conforms to the ≤60-char description standard
  - Frontmatter has required fields
  - The skill references the first-class ``actual`` provider (not the
    legacy custom-provider config path that conflicts with it)
  - The bundled OpenCode reference exists
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "devops"
    / "actual-setup"
)


@pytest.fixture(scope="module")
def skill_source() -> str:
    return (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontmatter(skill_source) -> dict:
    m = re.search(r"^---\n(.*?)\n---", skill_source, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


def test_skill_dir_exists() -> None:
    assert SKILL_DIR.is_dir(), f"missing skill dir: {SKILL_DIR}"


def test_description_under_60_chars(frontmatter) -> None:
    desc = frontmatter["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars (limit ≤60): {desc!r}"


def test_has_required_frontmatter_fields(frontmatter) -> None:
    for field in ("name", "description", "version", "license"):
        assert field in frontmatter, f"missing required field: {field}"


def test_credits_contributor(frontmatter) -> None:
    assert "shl0ms" in str(frontmatter.get("author", "")), (
        "author must credit the human contributor first"
    )


def test_uses_first_class_provider_not_custom_provider(skill_source) -> None:
    # The legacy setup configured Actual as providers.actual.* custom entries,
    # which now collides with the built-in provider of the same name.
    assert "--provider actual" in skill_source
    assert "hermes config set providers.actual.api " not in skill_source
    assert "key_env" not in skill_source


def test_opencode_reference_exists() -> None:
    assert (SKILL_DIR / "references" / "opencode.md").is_file()
