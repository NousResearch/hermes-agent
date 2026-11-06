"""Authoring-standards tests for the global-business-culture skill.

Validates the skill against the bundled-skill HARDLINE standards from
``skills/AGENTS.md``: frontmatter rules, section order, reference-file
contract, and authoring-language rules. Stdlib + pytest only, no network.
"""

from __future__ import annotations

import re
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / "business" / "global-business-culture"

EXPECTED_REFERENCES = [
    "frameworks.md",
    "western.md",
    "east-asia.md",
    "southeast-asia.md",
    "south-asia.md",
    "middle-east-north-africa.md",
    "sub-saharan-africa.md",
    "latin-america.md",
    "ldc-playbook.md",
]

EXPECTED_SECTIONS = [
    "## When to Use",
    "## Prerequisites",
    "## How to Run",
    "## Quick Reference",
    "## Procedure",
    "## Pitfalls",
    "## Verification",
]

REGION_SCHEMA_HEADERS = [
    "Snapshot",
    "Communication",
    "Trust",
    "Decisions",
    "Negotiation & contracts",
    "Etiquette",
    "Taboos",
]


def _skill_md() -> str:
    path = SKILL_DIR / "SKILL.md"
    assert path.is_file(), f"missing {path}"
    return path.read_text(encoding="utf-8")


def _frontmatter(text: str) -> dict:
    match = re.match(r"\A---\n(.*?)\n---\n", text, re.DOTALL)
    assert match, "SKILL.md must open with a YAML frontmatter block"
    fm = {}
    for line in match.group(1).splitlines():
        if ":" in line and not line.startswith((" ", "-", "\t")):
            key, _, value = line.partition(":")
            fm[key.strip()] = value.strip().strip('"').strip("'")
    return fm


def test_skill_dir_exists() -> None:
    assert SKILL_DIR.is_dir(), f"missing skill dir {SKILL_DIR}"


def test_description_standard() -> None:
    fm = _frontmatter(_skill_md())
    description = fm.get("description", "")
    assert description, "description required"
    assert len(description) <= 60, f"description must be <=60 chars, got {len(description)}: {description!r}"
    assert description.endswith("."), "description must end with a period"
    lowered = description.lower()
    for banned in ("powerful", "comprehensive", "seamless", "advanced"):
        assert banned not in lowered, f"marketing word {banned!r} banned in description"


def test_author_credits_human_first() -> None:
    fm = _frontmatter(_skill_md())
    author = fm.get("author", "")
    assert "(" in author and ")" in author, "author must include a GitHub handle in parentheses"
    assert not author.startswith("Hermes Agent"), "human contributor must be credited first"


def test_section_order() -> None:
    text = _skill_md()
    positions = [text.find(section) for section in EXPECTED_SECTIONS]
    assert all(pos != -1 for pos in positions), f"missing sections: {[s for s, p in zip(EXPECTED_SECTIONS, positions) if p == -1]}"
    assert positions == sorted(positions), "sections must appear in the standard order"


def test_references_exist_and_substantive() -> None:
    for name in EXPECTED_REFERENCES:
        path = SKILL_DIR / "references" / name
        assert path.is_file(), f"missing reference {name}"
        content = path.read_text(encoding="utf-8")
        assert len(content) > 2000, f"{name} too thin ({len(content)} chars)"
        assert "## Sources" in content or "Sources" in content, f"{name} must cite sources"


def test_region_files_cover_schema() -> None:
    region_files = [n for n in EXPECTED_REFERENCES if n not in ("frameworks.md", "ldc-playbook.md")]
    for name in region_files:
        content = (SKILL_DIR / "references" / name).read_text(encoding="utf-8")
        lowered = content.lower()
        for token in ("snapshot", "communication", "trust", "decision", "etiquette", "taboo"):
            assert token in lowered, f"{name} missing schema element {token!r}"


def test_country_coverage_documented() -> None:
    for country in ("Laos", "China", "Japan", "India", "Brazil", "Nigeria", "Germany", "Saudi"):
        found = any(
            country.lower() in (SKILL_DIR / "references" / n).read_text(encoding="utf-8").lower()
            for n in EXPECTED_REFERENCES
        )
        assert found, f"country {country} not covered in any reference file"


def test_hermes_tool_names_in_prose() -> None:
    """The skill should name native tools (web_search) not shell utilities."""
    text = _skill_md()
    assert "`web_search`" in text, "skill should reference web_search in backticks"
    shell_util = re.compile(r"\b(grep|cat|head|tail|sed|awk)\b")
    match = shell_util.search(text)
    assert match is None, f"shell utility {match.group(0)!r} must not appear in SKILL.md prose"


def test_no_hard_prompts_or_secrets() -> None:
    token_like = re.compile(
        r"sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_|xai-[A-Za-z0-9]{10,}|AKIA[0-9A-Z]{16}"
    )
    assignment_like = re.compile(r"(?i)\b(api_key|password|secret)\s*[:=]\s*\S")
    for path in [SKILL_DIR / "SKILL.md", *(SKILL_DIR / "references" / n for n in EXPECTED_REFERENCES)]:
        content = path.read_text(encoding="utf-8")
        assert not token_like.search(content), f"secret-like token found in {path.name}"
        assert not assignment_like.search(content), f"credential assignment pattern found in {path.name}"


def test_quick_reference_lists_all_reference_files() -> None:
    text = _skill_md()
    for name in EXPECTED_REFERENCES:
        assert f"`references/{name}`" in text, f"Quick Reference must point to {name}"
