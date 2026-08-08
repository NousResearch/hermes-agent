"""Guard the SKILL.md against the in-repo frontmatter validator contract.

Mirrors tools/skill_manager_tool.py::_validate_frontmatter so a bad edit to
the skill's own metadata fails here instead of at PR review.
"""

from __future__ import annotations

import re
from pathlib import Path

from _common import parse_frontmatter

SKILL_MD = Path(__file__).resolve().parent.parent / "SKILL.md"
MAX_NAME = 64
MAX_DESCRIPTION = 60
MAX_CONTENT = 100_000
EXPECTED_SECTIONS = [
    "## When to Use",
    "## Prerequisites",
    "## How to Run",
    "## Quick Reference",
    "## Procedure",
    "## Pitfalls",
    "## Verification",
]


def test_skill_md_exists():
    assert SKILL_MD.exists()


def test_starts_and_closes_with_delimiters():
    content = SKILL_MD.read_text(encoding="utf-8")
    assert content.startswith("---")
    assert re.search(r"\n---\s*\n", content[3:])


def test_required_fields_and_limits():
    content = SKILL_MD.read_text(encoding="utf-8")
    fm, body = parse_frontmatter(content)
    assert "name" in fm and fm["name"]
    assert re.fullmatch(r"[a-z0-9-]{1,%d}" % MAX_NAME, fm["name"])
    assert "description" in fm and fm["description"]
    assert len(fm["description"]) <= MAX_DESCRIPTION
    assert fm["description"].endswith(".")
    assert len(content) <= MAX_CONTENT
    assert body.strip()


def test_peer_matched_metadata_present():
    fm, _ = parse_frontmatter(SKILL_MD.read_text(encoding="utf-8"))
    for key in ("version", "author", "license", "metadata"):
        assert key in fm, f"missing peer-standard field: {key}"
    assert fm["author"].startswith("Benlloyd Goldstein (benlloydg)")


def test_modern_section_order():
    content = SKILL_MD.read_text(encoding="utf-8")
    headings = re.findall(r"^## .+$", content, flags=re.MULTILINE)
    assert [h for h in headings if h in EXPECTED_SECTIONS] == EXPECTED_SECTIONS
