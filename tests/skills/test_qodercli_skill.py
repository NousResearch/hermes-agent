from __future__ import annotations

import re
from pathlib import Path

import pytest

SKILL_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "autonomous-ai-agents"
    / "qodercli"
    / "SKILL.md"
)


@pytest.fixture
def skill_text() -> str:
    assert SKILL_PATH.exists(), f"SKILL.md not found at {SKILL_PATH}"
    return SKILL_PATH.read_text(encoding="utf-8")


@pytest.fixture
def frontmatter(skill_text: str) -> dict[str, str]:
    match = re.match(r"^---\n(.*?)\n---", skill_text, re.DOTALL)
    assert match, "No YAML frontmatter found"
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line and not line.startswith(" ") and not line.startswith("-"):
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip().strip('"')
    return fields


def test_description_length(frontmatter: dict[str, str]):
    desc = frontmatter.get("description", "")
    assert len(desc) <= 60, f"Description is {len(desc)} chars (max 60)"
    assert desc.endswith("."), "Description must end with a period"


def test_required_sections(skill_text: str):
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
        assert section in skill_text, f"Missing required section: {section}"


def test_section_order(skill_text: str):
    required = [
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]
    positions = [skill_text.index(s) for s in required]
    assert positions == sorted(positions), "Sections are out of order"


def test_platforms_valid(frontmatter: dict[str, str]):
    platforms = frontmatter.get("platforms", "")
    valid = {"linux", "macos", "windows"}
    listed = {p.strip().strip("[]") for p in platforms.split(",")}
    assert listed.issubset(valid), f"Invalid platforms: {listed - valid}"


def test_line_count(skill_text: str):
    lines = skill_text.count("\n")
    assert lines <= 250, f"Skill is {lines} lines (target ~200, hard cap 250)"


def test_no_marketing_words(frontmatter: dict[str, str]):
    desc = frontmatter.get("description", "").lower()
    banned = ["powerful", "comprehensive", "seamless", "advanced", "robust"]
    for word in banned:
        assert word not in desc, f"Marketing word '{word}' in description"


def test_author_not_hermes_agent(frontmatter: dict[str, str]):
    author = frontmatter.get("author", "")
    assert author != "Hermes Agent", "Author must credit the human contributor first"


def test_required_env_vars_declared(skill_text: str):
    assert "required_environment_variables" in skill_text
    assert "QODER_PERSONAL_ACCESS_TOKEN" in skill_text


def test_uses_terminal_tool(skill_text: str):
    assert "terminal(" in skill_text, "Skill must reference the terminal tool"


def test_no_raw_shell_utilities_in_prose(skill_text: str):
    in_fence = False
    prose_lines = []
    for line in skill_text.splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence and not line.startswith(("terminal(", "|", "#")):
            prose_lines.append(line)
    prose = " ".join(prose_lines)
    assert "grep " not in prose, "Use search_files instead of grep"
    assert "cat " not in prose, "Use read_file instead of cat"


def _parse_related_skills(text: str) -> list[str]:
    match = re.search(r"related_skills:\s*\[([^\]]*)\]", text)
    if not match:
        return []
    return [s.strip() for s in match.group(1).split(",") if s.strip()]


def test_related_skills_bidirectional(skill_text: str):
    related = _parse_related_skills(skill_text)
    assert related, "qodercli must declare related_skills"
    skills_dir = SKILL_PATH.parent.parent
    for name in related:
        sibling = skills_dir / name / "SKILL.md"
        assert sibling.exists(), f"Related skill '{name}' not found at {sibling}"
        sibling_text = sibling.read_text(encoding="utf-8")
        sibling_related = _parse_related_skills(sibling_text)
        assert "qodercli" in sibling_related, (
            f"Skill '{name}' does not list qodercli in its related_skills"
        )
