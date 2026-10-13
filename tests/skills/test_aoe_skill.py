"""
Smoke tests for the aoe (Agent of Empires) skill.

The skill drives an external `aoe` CLI over tmux, which we can't exercise in CI
(needs tmux + real agent processes). These tests instead verify that SKILL.md
conforms to the hardline authoring contract in AGENTS.md:

  - frontmatter shape (name, ≤60-char description, platforms, license, author)
  - correct category home (skills/autonomous-ai-agents/aoe/)
  - the modern section order is present
  - the documented JSON field names stay accurate (no project_path/group_path)

No network, no subprocess — stdlib + pytest + yaml only.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / "autonomous-ai-agents" / "aoe"
SKILL_MD = SKILL_DIR / "SKILL.md"


@pytest.fixture(scope="module")
def body() -> str:
    return SKILL_MD.read_text()


@pytest.fixture(scope="module")
def frontmatter(body: str) -> dict:
    m = re.search(r"^---\n(.*?)\n---", body, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


def test_skill_lives_in_autonomous_ai_agents() -> None:
    assert SKILL_DIR.is_dir(), f"missing skill dir: {SKILL_DIR}"
    assert SKILL_MD.is_file()


def test_name_matches_dir(frontmatter) -> None:
    assert frontmatter["name"] == "aoe"


def test_description_under_60_chars(frontmatter) -> None:
    desc = frontmatter["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars (hardline ≤60): {desc!r}"
    assert desc.endswith("."), "description must end with a period"


def test_license_mit(frontmatter) -> None:
    assert frontmatter["license"] == "MIT"


def test_author_credits_contributor_first(frontmatter) -> None:
    author = frontmatter["author"]
    assert "njbrake" in author, f"author should credit the contributor: {author!r}"


def test_platforms_gated_to_posix(frontmatter) -> None:
    # aoe + tmux are POSIX-only (tmux has no native Windows build).
    platforms = frontmatter["platforms"]
    assert "windows" not in platforms, "tmux is unavailable on Windows"
    assert set(platforms) == {"linux", "macos"}


def test_modern_section_order(body: str) -> None:
    required = [
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]
    positions = []
    for section in required:
        idx = body.find(section)
        assert idx != -1, f"missing required section: {section}"
        positions.append(idx)
    assert positions == sorted(positions), "sections are out of the modern order"


def test_documented_json_fields_are_accurate(body: str) -> None:
    # These are the real field names emitted by upstream list.rs / status.rs.
    for field in ('"path"', '"group"', '"tool"', '"command"', '"waiting"', '"running"'):
        assert field in body, f"expected documented JSON field {field} in SKILL.md"
    # Guard against regressing to the wrong field names as JSON keys. (The
    # Verification prose intentionally names them as anti-examples, so only
    # reject the quoted-key form that would appear in a documented shape.)
    assert '"project_path"' not in body
    assert '"group_path"' not in body


def test_references_native_terminal_tool(body: str) -> None:
    # Per the authoring contract, prose should point at the native `terminal`
    # tool as the interaction surface, not raw shell utilities.
    assert "`terminal`" in body
