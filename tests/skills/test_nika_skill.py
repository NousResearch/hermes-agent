"""
Smoke tests for the nika optional skill.

We can't run real workflows in CI (the nika binary is an external install),
so these tests verify the hardline conformance surface:
  - SKILL.md frontmatter conforms to the hardline format
  - platform gating matches what the skill actually teaches (POSIX shell)
  - the modern section order is present
  - the upstream engine license (AGPL) is disclosed in the body
  - the human contributor is credited first
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

SKILL_DIR = Path(__file__).resolve().parents[2] / "optional-skills" / "autonomous-ai-agents" / "nika"


@pytest.fixture(scope="module")
def raw() -> str:
    return (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontmatter(raw: str) -> dict:
    m = re.match(r"\A---\n(.*?)\n---\n", raw, re.S)
    assert m, "SKILL.md must open with a YAML frontmatter block"
    return yaml.safe_load(m.group(1))


def test_skill_dir_exists() -> None:
    assert SKILL_DIR.is_dir(), f"missing skill dir: {SKILL_DIR}"


def test_skill_md_present() -> None:
    assert (SKILL_DIR / "SKILL.md").is_file()


def test_description_under_60_chars(frontmatter) -> None:
    desc = frontmatter["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars (hardline <=60): {desc!r}"
    assert desc.endswith("."), "description must be one sentence ending with a period"
    assert "nika" not in desc.lower(), "description must not repeat the skill name"


def test_name_matches_dir(frontmatter) -> None:
    assert frontmatter["name"] == "nika"


def test_platforms_excludes_windows(frontmatter) -> None:
    # The skill teaches POSIX shell invocations (brew install path, $()
    # substitutions) and the engine ships macOS/Linux release binaries;
    # gated [linux, macos]. If a Windows build ships, update this test.
    assert "windows" not in frontmatter["platforms"]
    assert set(frontmatter["platforms"]) >= {"linux", "macos"}


def test_author_credits_contributor_first(frontmatter) -> None:
    author = frontmatter["author"]
    assert author.startswith("Thibaut Melen"), (
        f"author should credit the human contributor first: {author!r}"
    )
    assert "@ThibautMelen" in author


def test_upstream_license_disclosed(raw: str, frontmatter) -> None:
    # The skill itself is MIT; the engine it drives is AGPL — the body
    # must disclose the upstream license so users aren't surprised.
    assert frontmatter["license"] == "MIT"
    assert "AGPL" in raw, "upstream engine license (AGPL) must be disclosed in the body"


def test_modern_title(raw: str) -> None:
    assert re.search(r"^# Nika Skill$", raw, re.M), "title must be '# Nika Skill'"


def test_section_order(raw: str) -> None:
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
        m = re.search(rf"^{re.escape(section)}$", raw, re.M)
        assert m, f"missing required section: {section}"
        positions.append(m.start())
    assert positions == sorted(positions), "sections are out of the hardline order"


def test_no_env_example_touched() -> None:
    # This skill needs no env vars; it must not ship .env.example edits.
    assert not (SKILL_DIR / ".env.example").exists()


def test_teaches_native_tool_surface(raw: str) -> None:
    # The headline interaction surface must be the Hermes `terminal` tool.
    assert "`terminal`" in raw
