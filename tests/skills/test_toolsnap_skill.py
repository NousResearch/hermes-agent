"""
Smoke tests for the toolsnap optional skill.

The skill is prose-only (it drives a remote MCP server, no shipped scripts),
so these tests verify — without any network calls — that:
  - SKILL.md frontmatter conforms to the hardline format
  - the body follows the modern title/section sequence
  - the skill declares its MCP prerequisite and matches the catalog manifest
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / "optional-skills" / "web" / "toolsnap"
MANIFEST = REPO_ROOT / "optional-mcps" / "toolsnap" / "manifest.yaml"


@pytest.fixture(scope="module")
def skill_src() -> str:
    return (SKILL_DIR / "SKILL.md").read_text()


@pytest.fixture(scope="module")
def frontmatter(skill_src) -> dict:
    m = re.search(r"^---\n(.*?)\n---", skill_src, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


def test_skill_md_present() -> None:
    assert (SKILL_DIR / "SKILL.md").is_file()


def test_description_hardline(frontmatter) -> None:
    desc = frontmatter["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars (hardline ≤60): {desc!r}"
    assert desc.endswith("."), "description must end with a period"
    assert "toolsnap" not in desc.lower(), "description must not repeat the skill name"


def test_name_matches_dir(frontmatter) -> None:
    assert frontmatter["name"] == "toolsnap"


def test_platforms_all(frontmatter) -> None:
    # Remote MCP server over HTTP — nothing platform-bound runs locally.
    assert set(frontmatter["platforms"]) == {"linux", "macos", "windows"}


def test_author_credits_contributor(frontmatter) -> None:
    assert "icosaedro" in frontmatter["author"].lower()


def test_license_mit(frontmatter) -> None:
    assert frontmatter["license"] == "MIT"


def test_declares_mcp_prerequisite(frontmatter) -> None:
    assert frontmatter["prerequisites"]["mcps"] == ["toolsnap"]


def test_modern_section_sequence(skill_src) -> None:
    body = skill_src.split("---", 2)[2]
    assert body.lstrip().startswith("# ToolSnap Skill"), "modern '# <Skill> Skill' title required"
    headings = [
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]
    positions = [body.find(h) for h in headings]
    assert all(p >= 0 for p in positions), (
        f"missing sections: {[h for h, p in zip(headings, positions) if p < 0]}"
    )
    assert positions == sorted(positions), "sections out of the mandated order"


def test_references_precise_native_tools(skill_src) -> None:
    # Handoffs must name the precise native tools, not a generic "browser".
    assert "`browser_navigate`" in skill_src
    assert "`web_search`" in skill_src
    assert "`browser`" not in skill_src


def test_manifest_present_and_matches() -> None:
    manifest = yaml.safe_load(MANIFEST.read_text())
    assert manifest["name"] == "toolsnap"
    assert manifest["transport"]["type"] == "http"
    assert manifest["transport"]["url"].startswith("https://")
