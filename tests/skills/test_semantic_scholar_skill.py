"""Tests for skills/research/semantic-scholar/SKILL.md — structure and content checks."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_MD = REPO_ROOT / "skills" / "research" / "semantic-scholar" / "SKILL.md"


def _read_skill():
    return SKILL_MD.read_text(encoding="utf-8")


def _parse_frontmatter(text):
    match = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    assert match, "SKILL.md must start with YAML frontmatter"
    return match.group(1)


def test_skill_file_exists():
    assert SKILL_MD.is_file(), f"Expected {SKILL_MD} to exist"


def test_frontmatter_has_required_fields():
    fm = _parse_frontmatter(_read_skill())
    for field in ("name:", "description:", "version:", "platforms:"):
        assert field in fm, f"Frontmatter missing {field}"


def test_name_is_semantic_scholar():
    fm = _parse_frontmatter(_read_skill())
    assert "name: semantic-scholar" in fm


def test_description_within_60_chars():
    fm = _parse_frontmatter(_read_skill())
    match = re.search(r'description:\s*"([^"]+)"', fm)
    assert match, "description must be a quoted string"
    desc = match.group(1)
    assert len(desc) <= 60, f"Description is {len(desc)} chars, max 60: {desc!r}"


def test_platforms_include_all_three():
    fm = _parse_frontmatter(_read_skill())
    for platform in ("linux", "macos", "windows"):
        assert platform in fm, f"platforms should include {platform}"


def test_related_skills_includes_arxiv():
    fm = _parse_frontmatter(_read_skill())
    assert "arxiv" in fm, "related_skills should reference arxiv"


def test_required_sections_present():
    text = _read_skill()
    required = [
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]
    for section in required:
        assert section in text, f"Missing required section: {section}"


def test_all_urls_point_to_semantic_scholar():
    text = _read_skill()
    api_urls = re.findall(r"https://api\.[a-z.]+/", text)
    for url in api_urls:
        assert "semanticscholar.org" in url, f"Unexpected API URL: {url}"


def test_no_hardcoded_api_keys():
    text = _read_skill()
    assert "x-api-key: " not in text.replace(
        "x-api-key: $SEMANTIC_SCHOLAR_API_KEY", ""
    ), "SKILL.md should not contain hardcoded API keys"


def test_line_count_within_target():
    lines = _read_skill().splitlines()
    assert len(lines) <= 250, f"SKILL.md is {len(lines)} lines, target ≤200 (hard cap 250)"
