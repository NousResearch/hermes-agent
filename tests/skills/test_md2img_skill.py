"""Contract tests for the optional md2img skill."""

from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / "optional-skills" / "creative" / "md2img"
SKILL_PATH = SKILL_DIR / "SKILL.md"
REQUIRED_SECTIONS = [
    "## When to Use",
    "## Prerequisites",
    "## How to Run",
    "## Quick Reference",
    "## Procedure",
    "## Pitfalls",
    "## Verification",
]


@pytest.fixture(scope="module")
def skill_text() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontmatter(skill_text: str) -> str:
    match = re.match(r"\A---\n(.*?)\n---\n", skill_text, re.DOTALL)
    assert match, "SKILL.md must start with YAML frontmatter"
    return match.group(1)


def _scalar(frontmatter: str, field: str) -> str:
    match = re.search(rf"^{re.escape(field)}:\s*(.+)$", frontmatter, re.MULTILINE)
    assert match, f"missing frontmatter field: {field}"
    return match.group(1).strip().strip('"\'')


def test_skill_uses_optional_creative_location() -> None:
    assert SKILL_PATH.is_file()
    assert not (REPO_ROOT / "skills" / "md2img").exists()


def test_frontmatter_matches_skill_contract(frontmatter: str) -> None:
    description = _scalar(frontmatter, "description")

    assert _scalar(frontmatter, "name") == "md2img"
    assert len(description) <= 60
    assert description.endswith(".")
    assert _scalar(frontmatter, "version")
    assert _scalar(frontmatter, "license") == "MIT"
    assert "Juan Macias (@jmaciasluque)" in _scalar(frontmatter, "author")
    assert _scalar(frontmatter, "platforms") == "[linux, macos, windows]"
    assert re.search(r"^metadata:\n  hermes:\n", frontmatter, re.MULTILINE)
    assert re.search(r"^    tags:\s*\[[^]]+\]$", frontmatter, re.MULTILINE)
    assert re.search(r"^    category:\s*creative$", frontmatter, re.MULTILINE)
    assert re.search(r"^    related_skills:", frontmatter, re.MULTILINE)
    assert re.search(r"^    config:", frontmatter, re.MULTILINE)


def test_body_uses_modern_section_order(skill_text: str) -> None:
    assert "# md2img Skill" in skill_text

    positions = [skill_text.index(section) for section in REQUIRED_SECTIONS]
    assert positions == sorted(positions)
    assert all(skill_text.count(section) == 1 for section in REQUIRED_SECTIONS)


def test_workflow_uses_native_tools_without_renderer_internals(skill_text: str) -> None:
    for tool in ("terminal", "read_file", "patch", "vision_analyze"):
        assert f"`{tool}`" in skill_text

    assert "cat " not in skill_text
    assert "| md2img" not in skill_text

    for implementation_detail in (
        "goldmark",
        "Ghostscript",
        "RenderWithConfig",
        "drawString",
        "Benchmark baseline",
    ):
        assert implementation_detail not in skill_text
