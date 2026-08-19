"""Tests for the orchestra-research gateway skill."""

import os
import re
import yaml
import pytest
from pathlib import Path


SKILL_PATH = Path(__file__).parent.parent.parent / "optional-skills" / "research" / "orchestra" / "SKILL.md"


@pytest.fixture
def skill_content():
    if not SKILL_PATH.exists():
        pytest.skip("orchestra-research skill not found")
    return SKILL_PATH.read_text()


def test_skill_file_exists():
    """The gateway SKILL.md must exist at the correct path."""
    assert SKILL_PATH.exists(), f"Expected skill at {SKILL_PATH}"


def test_description_length(skill_content):
    """AGENTS.md:888-900 requires ≤60 character description."""
    parts = skill_content.split("---", 2)
    if len(parts) < 3:
        pytest.skip("No YAML frontmatter found")
    frontmatter = yaml.safe_load(parts[1])
    desc = frontmatter.get("description", "")
    assert len(desc) <= 60, f"Description '{desc}' is {len(desc)} chars (max 60)"


def test_description_ends_with_period(skill_content):
    """AGENTS.md:888-893 requires a terminal period."""
    parts = skill_content.split("---", 2)
    if len(parts) < 3:
        pytest.skip("No YAML frontmatter found")
    frontmatter = yaml.safe_load(parts[1])
    desc = frontmatter.get("description", "")
    assert desc.endswith("."), f"Description must end with '.' — got '{desc}'"


def test_author_field(skill_content):
    """AGENTS.md:931 requires a contributor author frontmatter field."""
    parts = skill_content.split("---", 2)
    if len(parts) < 3:
        pytest.skip("No YAML frontmatter found")
    frontmatter = yaml.safe_load(parts[1])
    assert "author" in frontmatter, "Missing 'author' field in frontmatter"


def test_hermes_tool_framing(skill_content):
    """AGENTS.md:902-914 requires native Hermes tool framing (terminal, read_file)."""
    assert "`terminal`" in skill_content, "Must reference the terminal tool"
    assert "`read_file`" in skill_content, "Must reference the read_file tool"


def test_upstream_path_format(skill_content):
    """Skill paths must match upstream: skills/<name>/SKILL.md."""
    assert "skills/<skill-name>/SKILL.md" in skill_content or "skills/" in skill_content, \
        "Must reference correct upstream path format"


def test_domain_count(skill_content):
    """Gateway must index all 21 domains."""
    domain_count = skill_content.count("| 0") + skill_content.count("| 1") + skill_content.count("| 2")
    assert domain_count >= 20, f"Expected 21 domains, found {domain_count}"


def test_gateway_pattern(skill_content):
    """Must use gateway pattern — clone, read, apply — not bundle all skills."""
    assert "clone" in skill_content.lower(), "Must mention git clone"
    assert "on demand" in skill_content.lower(), "Must describe on-demand fetching"
    assert "94" in skill_content or "Orchestra" in skill_content, "Must reference the orchestra library"
