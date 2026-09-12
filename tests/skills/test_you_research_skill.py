"""Smoke tests for the You.com research skill (skills/research/you-research)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest


SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / "research" / "you-research"
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


def test_old_youdotcom_skill_is_gone() -> None:
    assert not (SKILL_DIR.parent / "youdotcom").exists()


def test_description_matches_skill_standard(frontmatter: dict[str, object]) -> None:
    description = str(frontmatter["description"])
    assert len(description) <= 60
    assert description.endswith(".")


def test_required_api_key_setup_metadata_present(skill_text: str) -> None:
    assert "hermes mcp install youdotcom" in skill_text
    assert "YDC_API_KEY" in skill_text
    assert "free search-only mode" in skill_text


def test_no_deprecated_scoped_tools_mechanism(skill_text: str) -> None:
    """you-research/you-finance moved to dedicated endpoints; the bundled
    search-only entry has no per-tool opt-in mechanism to document."""
    assert "YDC_ALLOWED_TOOLS" not in skill_text
    assert "you-finance" not in skill_text


def test_documents_sanitized_mcp_tool_names(skill_text: str) -> None:
    """Hermes exposes MCP tools with sanitized names; the skill must use them."""
    assert "mcp_youdotcom_you_search" in skill_text
    assert "mcp_youdotcom_you_contents" in skill_text


def test_free_mode_degradation_is_documented(skill_text: str) -> None:
    """Free installs expose search only; the skill must tell the model how to
    detect that state and what to do when contents is unavailable."""
    assert "free search-only mode" in skill_text
    assert "mcp_youdotcom_you_contents" in skill_text


def test_search_pipeline_and_citations_present(skill_text: str) -> None:
    assert "## Search Pipeline" in skill_text
    assert "## Evidence Rules" in skill_text
    assert "## Citation Quality Rules" in skill_text


def test_untrusted_content_warning_present(skill_text: str) -> None:
    assert "untrusted external data" in skill_text
