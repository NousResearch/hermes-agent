"""Tests for the squirrelscan optional skill's SKILL.md."""

import re
from pathlib import Path

import pytest
import yaml

_REL = "optional-skills/web-development/squirrelscan/SKILL.md"


def _skill_path() -> Path:
    staging = Path(__file__).resolve().parents[2] / _REL
    if staging.exists():
        return staging
    return Path.home() / ".hermes/hermes-agent" / _REL


@pytest.fixture(scope="module")
def skill_text() -> str:
    path = _skill_path()
    assert path.exists(), f"SKILL.md not found at {path}"
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def parts(skill_text):
    assert skill_text.startswith("---\n"), "missing frontmatter"
    _, fm, body = skill_text.split("---\n", 2)
    return yaml.safe_load(fm), body


def test_frontmatter_required_fields(parts):
    fm, _ = parts
    for field in ("name", "description", "version", "author", "license", "platforms"):
        assert field in fm, f"missing frontmatter field: {field}"
    assert fm["name"] == "squirrelscan"
    assert fm["version"] == "0.1.0"


def test_description_length_and_period(parts):
    fm, _ = parts
    desc = fm["description"]
    assert isinstance(desc, str)
    assert len(desc) <= 60, f"description too long: {len(desc)} chars"
    assert desc.endswith("."), "description must end with a period"


def test_license_mit(parts):
    fm, _ = parts
    assert fm["license"] == "MIT"


def test_platforms_present(parts):
    fm, _ = parts
    platforms = fm["platforms"]
    assert isinstance(platforms, list) and platforms
    assert "linux" in platforms


def test_hermes_metadata_tags(parts):
    fm, _ = parts
    tags = fm["metadata"]["hermes"]["tags"]
    assert isinstance(tags, list) and tags


def test_real_binary_name_documented(parts):
    _, body = parts
    # The npm package installs a binary named `squirrel`, not `squirrelscan`.
    assert "squirrel audit" in body
    assert "node_modules/.bin/squirrel" in body


def test_wrong_invocation_form_absent(parts):
    _, body = parts
    assert "squirrelscan audit" not in body, (
        "wrong binary form: package is 'squirrelscan' but the binary is 'squirrel'"
    )


def test_hermes_tool_framing(parts):
    _, body = parts
    assert "terminal" in body, "body must frame invocations via the Hermes terminal tool"


def test_no_claude_residue(skill_text):
    assert not re.search(r"(?i)claude", skill_text), "upstream Claude residue found"
