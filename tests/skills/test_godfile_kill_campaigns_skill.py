"""Tests for the godfile-kill-campaigns skill: frontmatter and structure
per the repo's authoring standards (tests/skills/test_authoring_standards.py
enforces the same rules parametrically; these pin the skill's own contract).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

SKILL = Path(__file__).resolve().parents[2] / "skills/software-development/godfile-kill-campaigns/SKILL.md"


def _load():
    content = SKILL.read_text(encoding="utf-8")
    assert content.startswith("---"), "SKILL.md must start with ---"
    m = re.search(r"\n---\s*\n", content[3:])
    assert m, "unclosed frontmatter"
    return content, content[3 : m.start() + 3], content[m.end():]


def test_description_hardline():
    _, fm, _ = _load()
    import hermes_yaml as yaml

    data = yaml.safe_load(fm)
    desc = str(data.get("description") or "")
    assert len(desc) <= 60, f"description {len(desc)} chars (hardline 60)"
    assert desc.endswith(".")
    assert not re.search(r"\b(powerful|comprehensive|seamless|revolutionary|cutting-edge|state-of-the-art)\b", desc, re.I)


def test_name_matches_directory():
    _, fm, _ = _load()
    import hermes_yaml as yaml

    data = yaml.safe_load(fm)
    assert data["name"] == SKILL.parent.name


def test_related_skills_resolve():
    _, fm, _ = _load()
    import hermes_yaml as yaml

    data = yaml.safe_load(fm)
    related = ((data.get("metadata") or {}).get("hermes") or {}).get("related_skills") or []
    skills_root = SKILL.parents[3]
    all_names = {
        p.parent.name
        for pattern in ("skills/**/SKILL.md", "optional-skills/**/SKILL.md")
        for p in skills_root.glob(pattern)
    }
    for name in related:
        assert name in all_names, f"dangling related_skills entry: {name}"


def test_no_machine_local_paths():
    _, _, body = _load()
    assert not re.search(r"/home/(?!runner\b)[a-z0-9_-]+/|[A-Z]:\\+Users\+", body)


def test_body_structure_and_size():
    content, _, body = _load()
    assert len(content) <= 100_000
    for section in ("## When to Use", "## Procedure", "## Pitfalls", "## Verification"):
        assert section in body, f"missing section {section}"


def test_steps_have_completion_criteria():
    _, _, body = _load()
    # Every numbered procedure step ends with a "Done when" clause.
    steps = re.findall(r"^\d+\.\s", body, re.M)
    done_clauses = re.findall(r"Done when", body)
    assert steps, "no numbered steps found"
    assert len(done_clauses) == len(steps), "each step must end with a 'Done when' completion criterion"


def test_epic_and_tracker_interlock_documented():
    _, _, body = _load()
    assert "Part of" in body, "interlock keyword documented"
    assert "cross-referenced" in body, "timeline cross-reference check documented"
