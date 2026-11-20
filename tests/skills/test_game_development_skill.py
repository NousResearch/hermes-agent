from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / "optional-skills" / "gaming" / "game-development"
SKILL_MD = SKILL_DIR / "SKILL.md"

# Modern section order required by AGENTS.md (skill authoring standards).
REQUIRED_SECTIONS = [
    "# Game Development Skill",
    "## When to Use",
    "## Prerequisites",
    "## How to Run",
    "## Quick Reference",
    "## Procedure",
    "## Pitfalls",
    "## Verification",
]

REQUIRED_REFERENCES = [
    "engine-selection.md",
    "core-systems.md",
    "unity.md",
    "unreal.md",
    "godot.md",
    "shipping.md",
]


def _read_skill() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


def _frontmatter(text: str) -> str:
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert match, "SKILL.md must open with a YAML frontmatter block"
    return match.group(1)


def _frontmatter_field(text: str, key: str) -> str:
    match = re.search(rf"^{key}: (.*)$", _frontmatter(text), re.MULTILINE)
    assert match, f"frontmatter must define {key}"
    return match.group(1)


def test_skill_md_exists():
    assert SKILL_MD.is_file(), f"missing SKILL.md at {SKILL_MD}"


def test_description_is_single_line_and_within_limit():
    description = _frontmatter_field(_read_skill(), "description")
    # A block scalar ("|" / ">") would leave only the indicator on this line.
    assert description.strip() not in {"|", ">", "|-", ">-", "|+", ">+"}, (
        "description must be a single inline line, not a multi-line block scalar"
    )
    assert "\n" not in description
    assert len(description) <= 60, f"description is {len(description)} chars: {description!r}"
    assert description.endswith("."), "description must end with a period"


def test_author_credits_human_before_hermes_agent():
    author = _frontmatter_field(_read_skill(), "author")
    assert "@HeLLGURD" in author, "author must include the contributor handle"
    assert "Hermes Agent" in author, "author must credit Hermes Agent as collaborator"
    assert author.index("HeLLGURD") < author.index("Hermes Agent"), (
        "the human contributor must be credited before Hermes Agent"
    )


def test_required_sections_present_and_in_order():
    text = _read_skill()
    last_index = -1
    for section in REQUIRED_SECTIONS:
        index = text.find(section)
        assert index != -1, f"missing required section: {section!r}"
        assert index > last_index, f"section out of order: {section!r}"
        last_index = index


def test_referenced_reference_files_exist_and_are_linked():
    text = _read_skill()
    references_dir = SKILL_DIR / "references"
    for name in REQUIRED_REFERENCES:
        path = references_dir / name
        assert path.is_file(), f"missing reference file: {path}"
        assert f"references/{name}" in text, f"SKILL.md does not link references/{name}"
