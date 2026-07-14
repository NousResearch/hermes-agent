"""
Smoke tests for the home-assistant optional MCP skill.

The skill talks to a live Home Assistant install over MCP behind OAuth, so none
of that is exercisable in CI. These tests instead pin the things that silently
rot:
  - SKILL.md frontmatter conforms to the hardline format
  - documented tool names match how Hermes actually registers MCP tools
  - the MCP endpoint URL is not mangled by a tool-name rename
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

SKILL_DIR = Path(__file__).resolve().parents[2] / "optional-skills" / "mcp" / "home-assistant"

# Mirrors tools/mcp_tool.py: sanitize_mcp_name_component() + mcp_prefixed_tool_name().
# The skill documents `hermes mcp add home-assistant`, which sanitizes to home_assistant.
TOOL_PREFIX = "mcp__home_assistant__"

# The Home Assistant route the MCP server is mounted at. It is not a tool, so it
# must survive any prefixing of the selora_* tool names.
ENDPOINT_PATH = "/api/selora_ai/mcp"


@pytest.fixture(scope="module")
def skill_md() -> str:
    return (SKILL_DIR / "SKILL.md").read_text()


@pytest.fixture(scope="module")
def frontmatter(skill_md: str) -> dict:
    m = re.search(r"^---\n(.*?)\n---", skill_md, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


def test_skill_dir_exists() -> None:
    assert SKILL_DIR.is_dir(), f"missing skill dir: {SKILL_DIR}"


def test_skill_md_present() -> None:
    assert (SKILL_DIR / "SKILL.md").is_file()


def test_description_under_60_chars(frontmatter) -> None:
    desc = frontmatter["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars (hardline ≤60): {desc!r}"


def test_description_is_one_terminated_sentence(frontmatter) -> None:
    desc = frontmatter["description"]
    assert desc.endswith("."), f"description must end with a period: {desc!r}"
    assert desc.count(".") == 1, f"description must be a single sentence: {desc!r}"


def test_name_matches_dir(frontmatter) -> None:
    assert frontmatter["name"] == "home-assistant"


def test_author_credits_human_contributor(frontmatter) -> None:
    # AGENTS.md: the human contributor comes first, not the org or the tooling.
    author = frontmatter["author"]
    assert "Gunnar Beck Nelson" in author, f"author should credit the contributor: {author!r}"
    assert "GChief117" in author, f"author should carry the GitHub handle: {author!r}"


def test_license_mit(frontmatter) -> None:
    assert frontmatter["license"] == "MIT"


def test_title_has_skill_suffix(skill_md: str) -> None:
    m = re.search(r"^# (.+)$", skill_md, re.MULTILINE)
    assert m, "SKILL.md has no H1 title"
    assert m.group(1).endswith(" Skill"), f"title must use '# <Skill> Skill': {m.group(1)!r}"


def test_section_order_matches_hardline(skill_md: str) -> None:
    expected = [
        "When to Use",
        "Prerequisites",
        "How to Run",
        "Quick Reference",
        "Procedure",
        "Pitfalls",
        "Verification",
    ]
    assert re.findall(r"^## (.+)$", skill_md, re.MULTILINE) == expected


def test_tool_references_are_mcp_prefixed(skill_md: str) -> None:
    """Hermes exposes MCP tools as mcp__<server>__<tool>; bare selora_* names are not callable."""
    body = skill_md.replace(ENDPOINT_PATH, "")  # endpoint URL is not a tool reference
    bare = re.findall(r"(?<!__)\bselora_(?!ai\b)[a-z_]+", body)
    assert not bare, f"bare tool names are not callable, prefix with {TOOL_PREFIX}: {sorted(set(bare))}"


def test_documented_tools_all_carry_prefix(skill_md: str) -> None:
    tools = re.findall(r"`" + re.escape(TOOL_PREFIX) + r"(selora_[a-z_]+)`", skill_md)
    assert tools, "expected the Quick Reference to document prefixed tool names"
    assert "selora_get_home_snapshot" in tools, "snapshot tool should stay documented"


def test_endpoint_url_not_prefixed(skill_md: str) -> None:
    """Guards the obvious bad fix: a blind selora_* rename would corrupt the MCP URL."""
    assert ENDPOINT_PATH in skill_md, "MCP endpoint path should remain documented"
    assert f"{TOOL_PREFIX}selora_ai" not in skill_md, "endpoint URL must not be tool-prefixed"


def test_mutating_tools_flagged(skill_md: str) -> None:
    # Mutations are gated on a Selora Connect role; the skill must keep saying so.
    assert f"`{TOOL_PREFIX}selora_create_automation` 🔒" in skill_md
    assert f"`{TOOL_PREFIX}selora_delete_automation` 🔒" in skill_md
