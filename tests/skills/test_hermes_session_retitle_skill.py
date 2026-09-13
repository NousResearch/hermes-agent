"""
Tests for the hermes-session-retitle skill.

Validates:
  - SKILL.md frontmatter conforms to the ≤60-char description standard
  - Frontmatter credits the human contributor and has required fields
  - scripts/retitle.py loads and its pure helpers behave as documented
    (normalization test, title round-trip, conflict precheck)
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest
import yaml

SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "productivity"
    / "hermes-session-retitle"
)


@pytest.fixture(scope="module")
def skill_source() -> str:
    return (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontmatter(skill_source) -> dict:
    m = re.search(r"^---\n(.*?)\n---", skill_source, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


@pytest.fixture(scope="module")
def retitle():
    spec = importlib.util.spec_from_file_location(
        "retitle_skill_script", SKILL_DIR / "scripts" / "retitle.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_skill_dir_exists() -> None:
    assert SKILL_DIR.is_dir(), f"missing skill dir: {SKILL_DIR}"


def test_description_under_60_chars(frontmatter) -> None:
    desc = frontmatter["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars (limit ≤60): {desc!r}"


def test_has_required_frontmatter_fields(frontmatter) -> None:
    for field in ("name", "description", "version", "license"):
        assert field in frontmatter, f"missing required field: {field}"


def test_credits_contributor(frontmatter) -> None:
    assert "caesarjue" in str(frontmatter.get("author", "")), (
        "author must credit the human contributor first"
    )


def test_script_exists() -> None:
    assert (SKILL_DIR / "scripts" / "retitle.py").is_file()


def test_is_normalized_matches_prefix_only(retitle) -> None:
    assert retitle.is_normalized("📅 0911 | 🔧 修复 | x")
    assert not retitle.is_normalized("修复 dsh desktop")
    assert not retitle.is_normalized("")
    assert not retitle.is_normalized(None)


def test_build_and_parse_title_round_trip(retitle) -> None:
    title = retitle.build_title("0911", "🔧", "修复", "排查 DSH")
    assert title.startswith("📅 0911 | 🔧 修复 | ")
    parts = retitle.parse_title(title)
    assert parts == {"date": "0911", "emoji": "🔧", "type": "修复", "topic": "排查 DSH"}
    assert retitle.parse_title("not a normalized title") is None


def test_strip_skip_prefixes_reaches_user_text(retitle) -> None:
    raw = "[IMPORTANT: Background process x completed]\nReal user request here"
    assert retitle.strip_skip_prefixes(raw) == "Real user request here"
    assert retitle.strip_skip_prefixes("plain message") == "plain message"


def test_precheck_rejects_duplicates_and_collisions(retitle) -> None:
    plan = [
        {"id": "a", "title": "📅 0911 | 🔧 修复 | one"},
        {"id": "b", "title": "📅 0911 | 🔧 修复 | one"},  # internal duplicate
        {"id": "c", "title": "📅 0912 | ✍️ 内容 | two"},
    ]
    problems = retitle.precheck_plan(plan, existing={"📅 0912 | ✍️ 内容 | two"})
    assert any("internal duplicate" in p for p in problems)
    assert any("collides with existing" in p for p in problems)


def test_precheck_accepts_clean_plan(retitle) -> None:
    plan = [{"id": "a", "title": "📅 0911 | 🔧 修复 | one"}]
    assert retitle.precheck_plan(plan, existing={"📅 0910 | ✍️ 内容 | zero"}) == []
