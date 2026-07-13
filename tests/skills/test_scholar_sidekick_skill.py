"""
Smoke tests for the scholar-sidekick optional skill.

The skill is pure documentation over a public REST API — there are no shipped
scripts and we make no live network calls in CI. These tests verify:
  - SKILL.md frontmatter conforms to the hardline format
  - the body uses the modern section order
  - the documented request bodies stay in sync with the real API contract
    (the per-endpoint field names are the thing agents get wrong most often)
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "research"
    / "scholar-sidekick"
)


@pytest.fixture(scope="module")
def skill_md() -> str:
    return (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontmatter(skill_md: str) -> dict:
    m = re.search(r"^---\n(.*?)\n---", skill_md, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


def test_skill_dir_exists() -> None:
    assert SKILL_DIR.is_dir(), f"missing skill dir: {SKILL_DIR}"


def test_skill_md_present() -> None:
    assert (SKILL_DIR / "SKILL.md").is_file()


def test_description_under_60_chars(frontmatter) -> None:
    desc = frontmatter["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars (hardline <=60): {desc!r}"


def test_description_is_one_sentence_ending_in_period(frontmatter) -> None:
    desc = frontmatter["description"]
    assert desc.endswith("."), f"description must end with a period: {desc!r}"
    assert desc.count(".") == 1, f"description must be one sentence: {desc!r}"


def test_description_has_no_marketing_words(frontmatter) -> None:
    banned = {"powerful", "comprehensive", "seamless", "advanced"}
    words = set(re.findall(r"[a-z]+", frontmatter["description"].lower()))
    assert not (words & banned), f"marketing words in description: {words & banned}"


def test_name_matches_dir(frontmatter) -> None:
    assert frontmatter["name"] == SKILL_DIR.name


def test_author_credits_contributor(frontmatter) -> None:
    author = frontmatter["author"]
    assert "Mark Lavercombe" in author, f"author must credit the human first: {author!r}"
    assert "mlava" in author, f"author must carry the GitHub handle: {author!r}"


def test_license_mit(frontmatter) -> None:
    assert frontmatter["license"] == "MIT"


def test_modern_section_order(skill_md: str) -> None:
    required = [
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]
    positions = []
    for heading in required:
        idx = skill_md.find(heading)
        assert idx != -1, f"SKILL.md missing required section: {heading}"
        positions.append(idx)
    assert positions == sorted(positions), "sections are out of the mandated order"


def test_no_api_key_required(skill_md: str) -> None:
    """Anonymous access is the whole point — a key must never be a prerequisite."""
    prereqs = skill_md.split("## Prerequisites", 1)[1].split("## How to Run", 1)[0]
    assert "None." in prereqs, "Prerequisites must state that no setup is required"


def test_documents_optional_first_party_key(skill_md: str) -> None:
    """Anonymous is the default, but ssk_ keys exist — don't claim they don't."""
    assert "Bearer ssk_" in skill_md
    assert "X-RapidAPI-Key" in skill_md


def test_endpoint_bodies_match_api_contract(skill_md: str) -> None:
    """Field names differ per endpoint; drift here silently breaks every call."""
    # /api/format and /api/export take `text`.
    assert re.search(r"/api/format\S*\"[^`]*?\"text\"", skill_md, re.DOTALL)
    assert re.search(r"/api/export\S*\"[^`]*?\"text\"", skill_md, re.DOTALL)
    # /api/retraction-check and /api/oa-check take a single `id`.
    assert re.search(r"/api/retraction-check\S*\"[^`]*?\"id\"", skill_md, re.DOTALL)
    assert re.search(r"/api/oa-check\S*\"[^`]*?\"id\"", skill_md, re.DOTALL)
    # /api/verify nests the citation under `claimed`.
    assert re.search(r"/api/verify\S*\"[^`]*?\"claimed\"", skill_md, re.DOTALL)


def test_verify_verdicts_are_the_real_enum(skill_md: str) -> None:
    verify_section = skill_md.split("### 5.", 1)[1].split("## Pitfalls", 1)[0]
    for verdict in ("matched", "mismatch", "ambiguous", "not_found"):
        assert f"`{verdict}`" in verify_section, f"missing verify verdict: {verdict}"


def test_warns_against_fabricating_fallbacks(skill_md: str) -> None:
    pitfalls = skill_md.split("## Pitfalls", 1)[1].split("## Verification", 1)[0]
    assert "never invent" in pitfalls.lower()


def test_disclaims_paper_search(skill_md: str) -> None:
    """This skill cites papers you already have; `arxiv` finds new ones."""
    assert "`arxiv`" in skill_md
