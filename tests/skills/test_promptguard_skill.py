"""Contract tests for the optional PromptGuard skill."""

from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[2]
SKILL_MD = ROOT / "optional-skills" / "devops" / "promptguard" / "SKILL.md"
REQUIRED_HEADINGS = [
    "# PromptGuard Skill",
    "## When to Use",
    "## Prerequisites",
    "## How to Run",
    "## Quick Reference",
    "## Procedure",
    "## Pitfalls",
    "## Verification",
]


@pytest.fixture(scope="module")
def content() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontmatter(content: str) -> str:
    match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
    assert match, "SKILL.md missing YAML frontmatter"
    return match.group(1)


def _scalar(frontmatter: str, field: str) -> str:
    match = re.search(rf"^{re.escape(field)}:\s*(.+)$", frontmatter, re.MULTILINE)
    assert match, f"frontmatter missing {field}"
    return match.group(1).strip()


def test_description_is_compact_sentence(frontmatter: str) -> None:
    description = _scalar(frontmatter, "description")
    assert description not in {">", "|"}, "description must be one line"
    assert len(description) <= 60
    assert description.endswith(".")


def test_author_credits_human_first(frontmatter: str) -> None:
    author = _scalar(frontmatter, "author")
    contributors = [part.strip() for part in author.split(",")]
    assert re.fullmatch(r".+\s+\(@[A-Za-z0-9-]+\)", contributors[0])
    assert "Hermes Agent" in contributors[1:]


def test_platforms_match_posix_workflow(frontmatter: str) -> None:
    platforms = _scalar(frontmatter, "platforms")
    declared = set(re.findall(r"[A-Za-z0-9_-]+", platforms))
    assert {"linux", "macos"} <= declared


def test_modern_section_order(content: str) -> None:
    positions = [content.find(heading) for heading in REQUIRED_HEADINGS]
    assert all(position >= 0 for position in positions)
    assert positions == sorted(positions)


def test_uses_native_terminal_surface(content: str) -> None:
    assert "`terminal`" in content


def test_avoids_ungated_temp_path(content: str) -> None:
    assert "/tmp" not in content
