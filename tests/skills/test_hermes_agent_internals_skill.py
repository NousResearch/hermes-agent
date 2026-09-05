"""Frontmatter checks for skills/autonomous-ai-agents/hermes-agent-internals."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

SKILL_MD = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "autonomous-ai-agents"
    / "hermes-agent-internals"
    / "SKILL.md"
)


def _frontmatter() -> dict:
    content = SKILL_MD.read_text(encoding="utf-8")
    assert content.startswith("---")
    m = re.search(r"\n---\s*\n", content[3:])
    assert m
    return yaml.safe_load(content[3 : m.start() + 3])


def test_skill_file_exists() -> None:
    assert SKILL_MD.is_file()


def test_frontmatter_peer_shape() -> None:
    fm = _frontmatter()
    assert fm["name"] == "hermes-agent-internals"
    assert fm["license"] == "MIT"
    assert "hermes" in fm["metadata"]
    assert fm["metadata"]["hermes"]["related_skills"] == ["hermes-agent"]


def test_description_use_when_under_60() -> None:
    desc = _frontmatter()["description"]
    assert desc.startswith("Use when ")
    assert len(desc) <= 60


def test_body_has_required_sections() -> None:
    body = SKILL_MD.read_text(encoding="utf-8").split("---", 2)[2]
    for heading in (
        "## Overview",
        "## When to Use",
        "## Common Pitfalls",
        "## Verification Checklist",
    ):
        assert heading in body