"""Discoverability contract for the optional grimdall-guard security skill.

The plugin's enforcement behavior is tested in its own repo
(github.com/grimdalltech/hermes-grimdall). This in-tree test guards only the
skill's discoverability contract — frontmatter parses, the description stays
within the authoring limit, the skill is classified under ``security``, and the
CLI command / config path the skill documents are actually present — so the
skill remains loadable without duplicating the plugin's behavioral tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.skill_utils import parse_frontmatter

SKILL_DIR = Path(__file__).resolve().parents[2] / "optional-skills" / "security" / "grimdall-guard"
SKILL_MD = SKILL_DIR / "SKILL.md"


@pytest.fixture(scope="module")
def skill_text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontmatter(skill_text: str) -> dict:
    parsed, _ = parse_frontmatter(skill_text)
    return parsed


def test_frontmatter_parses_and_names_the_skill(frontmatter: dict) -> None:
    assert frontmatter.get("name") == "grimdall-guard"


def test_description_is_within_authoring_limit(frontmatter: dict) -> None:
    description = frontmatter.get("description")
    assert isinstance(description, str) and description.strip()
    assert len(description) <= 60
    assert description.endswith(".")


def test_skill_loads_under_security_category(frontmatter: dict) -> None:
    assert frontmatter["metadata"]["hermes"]["category"] == "security"


def test_enable_command_is_documented(skill_text: str) -> None:
    assert "hermes plugins enable" in skill_text


def test_config_path_is_documented(skill_text: str) -> None:
    assert "plugins.entries.grimdall.settings" in skill_text
