"""Hermetic contract tests for the Doubleword optional skill."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.skill_utils import parse_frontmatter
from tools import skills_tool
from tools.skills_hub import OptionalSkillSource


REPO_ROOT = Path(__file__).resolve().parents[2]
OPTIONAL_SKILLS_DIR = REPO_ROOT / "optional-skills"
SKILL_DIR = OPTIONAL_SKILLS_DIR / "mlops" / "doubleword-ai"
SKILL_MD = SKILL_DIR / "SKILL.md"
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
def parsed_skill() -> tuple[dict, str]:
    frontmatter, body = parse_frontmatter(SKILL_MD.read_text(encoding="utf-8"))
    return frontmatter, body


def test_hardline_frontmatter_contract(parsed_skill: tuple[dict, str]) -> None:
    frontmatter, _ = parsed_skill
    description = frontmatter["description"]

    assert frontmatter["name"] == "doubleword"
    assert len(description) <= 60
    assert description.endswith(".")
    assert frontmatter["author"].startswith("aschkandw")
    assert frontmatter["license"] == "MIT"


def test_secret_uses_secure_environment_metadata(
    parsed_skill: tuple[dict, str],
) -> None:
    frontmatter, _ = parsed_skill
    entries = frontmatter["required_environment_variables"]

    assert [entry["name"] for entry in entries] == ["DOUBLEWORD_API_KEY"]
    assert entries[0]["prompt"]
    assert entries[0]["help"]
    assert frontmatter["prerequisites"]["commands"] == ["dw"]
    assert "config" not in frontmatter["metadata"]["hermes"]


def test_body_uses_required_section_order(parsed_skill: tuple[dict, str]) -> None:
    _, body = parsed_skill

    assert body.lstrip().startswith("# Doubleword Skill")
    positions = [body.index(section) for section in REQUIRED_SECTIONS]
    assert positions == sorted(positions)


def test_headless_login_references_environment_not_literal_key() -> None:
    skill_text = SKILL_MD.read_text(encoding="utf-8")
    recipes_text = (SKILL_DIR / "references" / "cli-recipes.md").read_text(
        encoding="utf-8"
    )
    combined = f"{skill_text}\n{recipes_text}"

    assert 'dw login --api-key "$DOUBLEWORD_API_KEY"' in combined
    assert "dw login --api-key <key>" not in combined


def test_official_source_discovers_doubleword() -> None:
    source = OptionalSkillSource()
    source._optional_dir = OPTIONAL_SKILLS_DIR

    matches = source.search("doubleword")
    doubleword = next(meta for meta in matches if meta.name == "doubleword")

    assert doubleword.identifier == "official/mlops/doubleword-ai"
    assert doubleword.source == "official"
    assert doubleword.trust_level == "builtin"


def _load_skill(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[str, dict]:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    with (
        patch.object(skills_tool, "SKILLS_DIR", OPTIONAL_SKILLS_DIR),
        patch.object(skills_tool, "_secret_capture_callback", None),
    ):
        raw = skills_tool.skill_view("mlops/doubleword-ai", preprocess=False)
    return raw, json.loads(raw)


def test_loader_reports_missing_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("DOUBLEWORD_API_KEY", raising=False)

    _, loaded = _load_skill(monkeypatch, tmp_path)

    assert loaded["success"] is True
    assert loaded["setup_needed"] is True
    assert loaded["missing_required_environment_variables"] == [
        "DOUBLEWORD_API_KEY"
    ]


def test_loader_accepts_key_without_exposing_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    secret = "dw-test-secret-must-not-render"
    monkeypatch.setenv("DOUBLEWORD_API_KEY", secret)

    raw, loaded = _load_skill(monkeypatch, tmp_path)

    assert loaded["success"] is True
    assert loaded["setup_needed"] is False
    assert loaded["missing_required_environment_variables"] == []
    assert secret not in raw
