"""Tests for the scientific-skills gateway skill."""

import yaml
import pytest
from pathlib import Path


SKILL_PATH = Path(__file__).parent.parent.parent / "optional-skills" / "research" / "scientific-skills" / "SKILL.md"


@pytest.fixture
def skill_content():
    if not SKILL_PATH.exists():
        pytest.skip("scientific-skills skill not found")
    return SKILL_PATH.read_text()


def test_skill_file_exists():
    assert SKILL_PATH.exists(), f"Expected skill at {SKILL_PATH}"


def test_description_length(skill_content):
    parts = skill_content.split("---", 2)
    if len(parts) < 3:
        pytest.skip("No YAML frontmatter")
    fm = yaml.safe_load(parts[1])
    desc = fm.get("description", "")
    assert len(desc) <= 60, f"Description '{desc}' is {len(desc)} chars (max 60)"


def test_description_ends_with_period(skill_content):
    parts = skill_content.split("---", 2)
    if len(parts) < 3:
        pytest.skip("No YAML frontmatter")
    fm = yaml.safe_load(parts[1])
    assert fm.get("description", "").endswith("."), "Description must end with '.'"


def test_author_field(skill_content):
    parts = skill_content.split("---", 2)
    if len(parts) < 3:
        pytest.skip("No YAML frontmatter")
    fm = yaml.safe_load(parts[1])
    assert "author" in fm, "Missing 'author' field"


def test_hermes_tool_framing(skill_content):
    assert "`terminal`" in skill_content, "Must reference terminal tool"
    assert "`read_file`" in skill_content, "Must reference read_file tool"


def test_upstream_path(skill_content):
    assert "skills/<skill-name>/SKILL.md" in skill_content, "Must use correct upstream path format"


def test_domain_count(skill_content):
    lines = [l for l in skill_content.split("\n") if l.strip().startswith("-") or "anndata" in l.lower()]
    assert len(lines) > 0, "Must have indexed domains"
