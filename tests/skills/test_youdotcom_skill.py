"""Smoke tests for the You.com research skill."""

from __future__ import annotations

import re
from pathlib import Path

import pytest


SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / "research" / "youdotcom"
SKILL_MD = SKILL_DIR / "SKILL.md"


@pytest.fixture(scope="module")
def skill_text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontmatter(skill_text: str) -> dict[str, object]:
    match = re.search(r"^---\n(.*?)\n---", skill_text, re.DOTALL)
    assert match, "SKILL.md missing YAML frontmatter"
    data: dict[str, object] = {}
    for line in match.group(1).splitlines():
        if ": " in line and not line.startswith("  "):
            key, value = line.split(": ", 1)
            data[key] = value
    return data


def test_skill_dir_exists() -> None:
    assert SKILL_DIR.is_dir()
    assert SKILL_MD.is_file()


def test_description_matches_skill_standard(frontmatter: dict[str, object]) -> None:
    description = str(frontmatter["description"])
    assert len(description) <= 60
    assert description.endswith(".")


def test_required_api_key_setup_metadata_present(skill_text: str) -> None:
    assert "hermes mcp install youdotcom" in skill_text
    assert "YDC_API_KEY" in skill_text
    assert "YDC_ALLOWED_TOOLS" in skill_text
    assert "free search-only mode" in skill_text


def test_modern_section_order(skill_text: str) -> None:
    headings = [
        "# You.com MCP Research Skill",
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]
    positions = [skill_text.index(heading) for heading in headings]
    assert positions == sorted(positions)


def test_terminal_tool_is_the_direct_api_surface(skill_text: str) -> None:
    assert "Use the registered MCP tools directly" in skill_text
    assert "do not require the `ydc` CLI" in skill_text
    assert "curl" not in skill_text


def test_mcp_tool_names_documented(skill_text: str) -> None:
    assert "mcp_youdotcom_you_search" in skill_text
    assert "mcp_youdotcom_you_contents" in skill_text
    assert "mcp_youdotcom_you_research" in skill_text
    assert "mcp_youdotcom_you_finance" in skill_text
    assert "you-search" in skill_text
    assert "you-finance" in skill_text
