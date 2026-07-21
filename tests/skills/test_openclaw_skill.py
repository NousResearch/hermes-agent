"""Smoke tests for the bundled OpenClaw operational skill."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / "autonomous-ai-agents" / "openclaw"


@pytest.fixture(scope="module")
def skill_text() -> str:
    return (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontmatter(skill_text: str) -> dict:
    match = re.search(r"^---\n(.*?)\n---", skill_text, re.DOTALL)
    assert match, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(match.group(1))


def test_skill_md_present() -> None:
    assert (SKILL_DIR / "SKILL.md").is_file()


def test_frontmatter_hardline_shape(frontmatter: dict) -> None:
    assert frontmatter["name"] == "openclaw"
    assert frontmatter["license"]
    assert set(frontmatter["platforms"]) >= {"linux", "macos"}

    description = frontmatter["description"]
    assert len(description) <= 60
    assert description.endswith(".")


def test_metadata_points_to_official_source(frontmatter: dict) -> None:
    hermes = frontmatter["metadata"]["hermes"]
    assert hermes["homepage"] == "https://github.com/openclaw/openclaw"
    assert "OpenClaw" in hermes["tags"]
    assert "hermes-agent" in hermes["related_skills"]


def test_official_docs_referenced(skill_text: str) -> None:
    assert "docs.openclaw.ai" in skill_text
    assert "official Hermes OpenClaw migration docs" in skill_text


def test_no_migration_apply_recipes(skill_text: str) -> None:
    blocked_fragments = [
        "hermes claw migrate",
        "openclaw migrate apply",
        "--include-secrets",
    ]

    for fragment in blocked_fragments:
        assert fragment not in skill_text


def test_openclaw_cli_flow_uses_documented_surfaces(skill_text: str) -> None:
    required_fragments = [
        "openclaw config file",
        "openclaw config validate --json",
        "openclaw config patch --file ./openclaw.patch.json5 --dry-run",
        "openclaw config set --batch-file ./openclaw-config-set.batch.json --dry-run",
        "openclaw gateway status",
        "openclaw skills check",
    ]

    for fragment in required_fragments:
        assert fragment in skill_text


def test_modern_sections_present(skill_text: str) -> None:
    for heading in [
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]:
        assert heading in skill_text
