"""Dual Gate contracts for the optional subagent-driven-development skill.

Leaf implementers do not commit. Task size is a reviewable deliverable,
not a 2-5 minute keystroke. This skill must not re-teach the factory
plan-skill anti-pattern the Dual Gate removed.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_PATH = (
    REPO_ROOT
    / "optional-skills"
    / "software-development"
    / "subagent-driven-development"
    / "SKILL.md"
)


def _frontmatter_and_body() -> tuple[dict, str]:
    content = SKILL_PATH.read_text(encoding="utf-8")
    assert content.startswith("---")
    m = re.search(r"\n---\s*\n", content[3:])
    assert m, "frontmatter must close with ---"
    fm = yaml.safe_load(content[3 : m.start() + 3])
    body = content[m.end() + 3 :]
    return fm, body


def test_skill_file_exists():
    assert SKILL_PATH.is_file()


def test_does_not_teach_two_to_five_minute_keystroke_tasks():
    _, body = _frontmatter_and_body()
    assert "Each task = 2-5 minutes of focused work." not in body


def test_implementer_example_does_not_tell_the_child_to_commit():
    _, body = _frontmatter_and_body()
    assert "6. Commit" not in body
    assert "git commit" not in body
