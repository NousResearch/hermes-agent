"""
Tests for the git-workflow optional skill.

No network and no live git. These verify the SKILL.md meets the hardline skill
standards (AGENTS.md) and that the reviewer feedback on PR #40778 stuck:
  - description is a single sentence <= 60 chars, ending with a period
  - the modern section order is present
  - related_skills point only at skills that exist in this repo (the old
    github-pr-reviewer / changelog-generator references are gone)
  - every `git reset --hard` sits next to a confirmation instruction
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / "optional-skills" / "software-development" / "git-workflow"
SKILL_MD = SKILL_DIR / "SKILL.md"

# Sections that MUST appear, in this order (AGENTS.md hardline section order).
REQUIRED_SECTIONS = [
    "# Git Workflow Skill",
    "## When to Use",
    "## Prerequisites",
    "## How to Run",
    "## Quick Reference",
    "## Procedure",
    "## Pitfalls",
    "## Verification",
]

# related_skills must resolve to a real skill directory on disk.
KNOWN_SKILLS = {
    "github-code-review": REPO_ROOT / "skills" / "github" / "github-code-review",
    "github-pr-workflow": REPO_ROOT / "skills" / "github" / "github-pr-workflow",
}

# Names the reviewer flagged as non-existent — they must never reappear.
FORBIDDEN_SKILL_REFS = ["github-pr-reviewer", "changelog-generator"]


@pytest.fixture(scope="module")
def source() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontmatter(source: str) -> str:
    m = re.search(r"^---\n(.*?)\n---", source, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return m.group(1)


def _field(frontmatter: str, key: str) -> str:
    """Pull a scalar frontmatter field without a YAML dependency (stdlib only)."""
    m = re.search(rf"^\s*{re.escape(key)}:\s*(.+)$", frontmatter, re.MULTILINE)
    assert m, f"frontmatter missing '{key}'"
    return m.group(1).strip().strip('"').strip("'")


def test_skill_files_exist() -> None:
    assert SKILL_DIR.is_dir(), f"missing skill dir: {SKILL_DIR}"
    assert SKILL_MD.is_file(), f"missing SKILL.md: {SKILL_MD}"


def test_name_matches_dir(frontmatter: str) -> None:
    assert _field(frontmatter, "name") == "git-workflow"


def test_description_is_short_single_sentence(frontmatter: str) -> None:
    desc = _field(frontmatter, "description")
    assert len(desc) <= 60, f"description is {len(desc)} chars (hardline <=60): {desc!r}"
    assert desc.endswith("."), "description must end with a period"
    assert desc.count(".") == 1, "description must be a single sentence"
    assert "git-workflow" not in desc.lower(), "description must not repeat the skill name"


def test_author_credits_human_first(frontmatter: str) -> None:
    author = _field(frontmatter, "author")
    assert author.startswith("Burak Koç"), f"human contributor must be credited first: {author!r}"
    assert "@HeLLGURD" in author


def test_license_mit(frontmatter: str) -> None:
    assert _field(frontmatter, "license") == "MIT"


def test_platforms_are_cross_platform(frontmatter: str) -> None:
    m = re.search(r"^\s*platforms:\s*\[(.*?)\]", frontmatter, re.MULTILINE)
    assert m, "frontmatter missing platforms"
    platforms = {p.strip() for p in m.group(1).split(",") if p.strip()}
    assert platforms == {"linux", "macos", "windows"}, platforms


def test_required_sections_present_in_order(source: str) -> None:
    last = -1
    for section in REQUIRED_SECTIONS:
        idx = source.find(section)
        assert idx != -1, f"missing required section: {section!r}"
        assert idx > last, f"section out of order: {section!r}"
        last = idx


def test_related_skills_reference_real_skills(frontmatter: str) -> None:
    m = re.search(r"related_skills:\s*\[(.*?)\]", frontmatter)
    assert m, "frontmatter missing related_skills"
    refs = [r.strip() for r in m.group(1).split(",") if r.strip()]
    assert set(refs) == set(KNOWN_SKILLS), (
        f"related_skills must be exactly {sorted(KNOWN_SKILLS)}: {refs}"
    )
    for ref in refs:
        assert KNOWN_SKILLS[ref].is_dir(), f"related skill does not exist on disk: {ref}"


def test_no_dangling_skill_references(source: str) -> None:
    for bad in FORBIDDEN_SKILL_REFS:
        assert bad not in source, f"SKILL.md still references non-existent skill: {bad}"


def test_every_hard_reset_is_confirmation_gated(source: str) -> None:
    """The reviewer's core fix: no unguarded `git reset --hard`. Every mention
    must sit next to a confirmation instruction."""
    lines = source.splitlines()
    hits = [i for i, line in enumerate(lines) if "reset --hard" in line]
    assert hits, "expected the skill to still document `git reset --hard`"
    for i in hits:
        window = "\n".join(lines[max(0, i - 8): i + 5]).lower()
        assert "confirm" in window, (
            f"`git reset --hard` on line {i + 1} is not gated by a nearby "
            f"confirmation instruction: {lines[i]!r}"
        )
