"""
Structural tests for the vortex-notes optional skill.

The skill talks to a paired, possibly end-to-end-encrypted vault over an MCP
server, so we can't exercise it in CI. These tests verify the SKILL.md conforms
to the hardline authoring standards:
  - frontmatter shape + ≤60-char description
  - the human contributor is credited
  - modern section order
  - prose points at native Hermes tools / MCP, not raw shell utilities
stdlib + pytest + pyyaml only; no network.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

SKILL_DIR = Path(__file__).resolve().parents[2] / "optional-skills" / "productivity" / "vortex-notes"

MARKETING = ("powerful", "comprehensive", "seamless", "advanced")
REQUIRED_SECTIONS = (
    "## When to Use",
    "## Prerequisites",
    "## How to Run",
    "## Quick Reference",
    "## Procedure",
    "## Pitfalls",
    "## Verification",
)


@pytest.fixture(scope="module")
def source() -> str:
    return (SKILL_DIR / "SKILL.md").read_text()


@pytest.fixture(scope="module")
def frontmatter(source) -> dict:
    m = re.search(r"^---\n(.*?)\n---", source, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


@pytest.fixture(scope="module")
def body(source) -> str:
    return source.split("\n---\n", 1)[1]


def test_skill_dir_exists() -> None:
    assert SKILL_DIR.is_dir(), f"missing skill dir: {SKILL_DIR}"


def test_skill_md_present() -> None:
    assert (SKILL_DIR / "SKILL.md").is_file()


def test_description_under_60_chars(frontmatter) -> None:
    desc = frontmatter["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars (hardline ≤60): {desc!r}"
    assert desc.endswith("."), "description must be one sentence ending with a period"
    assert "vortex" not in desc.lower(), "description must not repeat the skill name"
    for word in MARKETING:
        assert word not in desc.lower(), f"marketing word {word!r} in description"


def test_required_frontmatter_fields(frontmatter) -> None:
    for field in ("name", "version", "author", "license", "platforms"):
        assert frontmatter.get(field), f"missing frontmatter field: {field}"


def test_author_credits_human(frontmatter) -> None:
    author = str(frontmatter["author"])
    assert "@vortex-303" in author, "author must credit the human contributor + handle"
    assert "Hermes Agent" not in author, "credit the human, not the tool"


def test_modern_section_order(body) -> None:
    positions = []
    for section in REQUIRED_SECTIONS:
        idx = body.find(section)
        assert idx != -1, f"missing required section: {section}"
        positions.append(idx)
    assert positions == sorted(positions), "sections are out of the required order"


def test_no_raw_shell_utilities_headlined(body) -> None:
    for tool in ("`grep`", "`cat`", "`sed`", "`awk`", "`find`"):
        assert tool not in body, f"{tool} should map to a native Hermes tool, not be headlined"


def test_points_at_native_tools_or_mcp(body) -> None:
    assert "MCP" in body and "vortex-notes" in body, "must name the expected MCP server"
    assert any(t in body for t in ("`read_file`", "`search_files`", "`patch`", "`terminal`")), (
        "must point at native Hermes tools by name"
    )
