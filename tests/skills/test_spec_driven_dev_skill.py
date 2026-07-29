from __future__ import annotations

import re
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_PATH = REPO_ROOT / "skills" / "software-development" / "spec-driven-dev" / "SKILL.md"
REQUIRED_SECTION_ORDER = [
    "When to Use",
    "Prerequisites",
    "How to Run",
    "Quick Reference",
    "Procedure",
    "Pitfalls",
    "Verification",
]


def _skill_text() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


def _frontmatter() -> dict[str, object]:
    text = _skill_text()
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert match is not None
    parsed = yaml.safe_load(match.group(1))
    assert isinstance(parsed, dict)
    return parsed


def test_spec_driven_dev_frontmatter_credits_external_contributor_first():
    author = str(_frontmatter()["author"])

    assert author.startswith("Juan Ramon Gros (@jrgros-ops)")


def test_spec_driven_dev_title_ends_with_skill():
    text = _skill_text()
    title = next(line for line in text.splitlines() if line.startswith("# "))

    assert title == "# Spec-Driven Development Skill"


def test_spec_driven_dev_required_section_order():
    headings = re.findall(r"^## (.+)$", _skill_text(), flags=re.MULTILINE)

    assert headings == REQUIRED_SECTION_ORDER


def test_spec_driven_dev_uses_project_scoped_constitution():
    text = _skill_text()

    assert ".spec/constitution.md" in text
    assert ".spec/<feature>/constitution.md" not in text
    assert ".spec/<feature>/spec.md" in text
    assert ".spec/<feature>/plan.md" in text
    assert ".spec/<feature>/tasks.md" in text


def test_spec_driven_dev_description_matches_skill_contract():
    description = str(_frontmatter()["description"])

    assert len(description) <= 60
    assert description.endswith(".")
