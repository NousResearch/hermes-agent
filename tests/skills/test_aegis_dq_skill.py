"""Offline metadata and contract tests for the aegis-dq data quality skill."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / "skills" / "data-quality" / "aegis-dq"
SKILL_MD = SKILL_DIR / "SKILL.md"

MCP_PREFIX = "mcp__aegis__"
EXPECTED_TOOLS = [
    "run_validation",
    "list_runs",
    "get_run_report",
    "get_trajectory",
    "search_decisions",
    "compare_reports",
    "summarize_reports",
    "check_consistency",
    "load_pipeline",
]


def _parse_frontmatter(content: str) -> dict:
    from agent.skill_utils import parse_frontmatter

    frontmatter, _ = parse_frontmatter(content)
    return frontmatter


@pytest.fixture(scope="module")
def skill_text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontmatter(skill_text: str) -> dict:
    return _parse_frontmatter(skill_text)


def test_skill_file_exists():
    assert SKILL_MD.is_file(), f"SKILL.md not found at {SKILL_MD}"


def test_description_length(frontmatter: dict):
    description = frontmatter.get("description", "")
    assert isinstance(description, str) and description.strip()
    assert len(description) <= 60, (
        f"description is {len(description)} chars (max 60): {description!r}"
    )
    assert description.endswith("."), "description must end with a period"


def test_author_credits_contributor_first(frontmatter: dict):
    author = str(frontmatter.get("author", ""))
    assert "koreddi" in author.lower() or "shiva" in author.lower(), (
        f"author field must credit the human contributor first, got: {author!r}"
    )


def test_license(frontmatter: dict):
    assert frontmatter.get("license") == "Apache-2.0"


def test_platforms(frontmatter: dict):
    platforms = frontmatter.get("platforms", [])
    assert isinstance(platforms, list)
    assert set(platforms) == {"linux", "macos", "windows"}


def test_install_command_includes_mcp_extra(skill_text: str):
    assert "aegis-dq[mcp]" in skill_text, (
        "Prerequisites must install aegis-dq[mcp] — bare aegis-dq omits the MCP dependency"
    )


def test_all_tools_use_mcp_prefix(skill_text: str):
    for tool in EXPECTED_TOOLS:
        prefixed = f"{MCP_PREFIX}{tool}"
        assert prefixed in skill_text, (
            f"Tool {tool!r} must be referenced as {prefixed!r} in SKILL.md"
        )


def test_no_bare_tool_names_in_tool_table(skill_text: str):
    for tool in EXPECTED_TOOLS:
        # bare name inside backticks but NOT preceded by mcp__aegis__
        bare = re.compile(rf"(?<!mcp__aegis__)`{re.escape(tool)}`")
        assert not bare.search(skill_text), (
            f"Bare tool name `{tool}` found — use `{MCP_PREFIX}{tool}` instead"
        )


def test_modern_section_order(skill_text: str):
    sections = re.findall(r"^##\s+(.+)$", skill_text, re.MULTILINE)
    required = [
        "When to Use",
        "Prerequisites",
        "How to Run",
        "Quick Reference",
        "Procedure",
        "Pitfalls",
        "Verification",
    ]
    for section in required:
        assert any(section.lower() in s.lower() for s in sections), (
            f"Required section '## {section}' missing from SKILL.md"
        )
