"""Tests for the video-shotcraft optional skill (staged port)."""

import re
from pathlib import Path

import pytest

REL = "optional-skills/creative/video-shotcraft/SKILL.md"


def _skill_path() -> Path:
    primary = Path(__file__).resolve().parents[2] / REL
    if primary.exists():
        return primary
    fallback = Path.home() / ".hermes/hermes-agent" / REL
    if fallback.exists():
        return fallback
    pytest.skip(f"SKILL.md not found at {primary} or {fallback}")


@pytest.fixture(scope="module")
def skill_path() -> Path:
    return _skill_path()


@pytest.fixture(scope="module")
def skill_text(skill_path: Path) -> str:
    return skill_path.read_text(encoding="utf-8")


def _split(text: str):
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    assert m, "SKILL.md must start with a YAML frontmatter block"
    return m.group(1), m.group(2)


def _parse_frontmatter(raw: str) -> dict:
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(raw)
        assert isinstance(data, dict)
        return data
    except ImportError:
        # Minimal fallback parser for the flat keys asserted below.
        data: dict = {}
        for line in raw.splitlines():
            m = re.match(r"^(\w[\w-]*):\s*(.*)$", line)
            if m:
                val = m.group(2).strip().strip("'\"")
                data[m.group(1)] = val
        return data


@pytest.fixture(scope="module")
def frontmatter(skill_text: str) -> dict:
    raw, _ = _split(skill_text)
    return _parse_frontmatter(raw)


@pytest.fixture(scope="module")
def body(skill_text: str) -> str:
    _, b = _split(skill_text)
    return b


def test_required_frontmatter_fields(frontmatter):
    for field in ("name", "description", "version", "author", "license"):
        assert field in frontmatter, f"missing frontmatter field: {field}"
    assert frontmatter["name"] == "video-shotcraft"
    assert frontmatter["version"] == "0.1.0"


def test_description_length_and_period(frontmatter):
    desc = str(frontmatter["description"])
    assert len(desc) <= 60, f"description is {len(desc)} chars (max 60)"
    assert desc.endswith("."), "description must end with a period"


def test_license_is_apache(frontmatter):
    assert str(frontmatter["license"]) == "Apache-2.0"


def test_no_claude_residue(skill_text):
    assert not re.search(r"(?i)claude", skill_text)


def test_mentions_shallow_clone_of_upstream(body):
    assert "git clone --depth 1" in body
    assert "https://github.com/Vincentwei1021/video-shotcraft" in body


def test_mentions_terminal_tool(body):
    assert "`terminal`" in body


def test_vendoring_invariant(skill_path, skill_text):
    """Every references/*.md path mentioned must exist locally or be framed
    as living in the upstream clone."""
    skill_dir = skill_path.parent
    for line in skill_text.splitlines():
        for ref in re.findall(r"references/[\w./-]+\.md", line):
            local = skill_dir / ref
            lowered = line.lower()
            assert local.exists() or "upstream" in lowered or "clone" in lowered, (
                f"dangling reference path {ref!r} on line: {line.strip()!r}"
            )
