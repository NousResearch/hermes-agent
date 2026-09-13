"""Standards tests for the leaf-brief-standards skill.

Mirrors the mechanical subset of the authoring standards for this skill:
frontmatter shape, description length, machine-local paths, marketing words.
"""

import re
from pathlib import Path

import pytest

SKILL = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "software-development"
    / "leaf-brief-standards"
    / "SKILL.md"
)


@pytest.fixture(scope="module")
def text() -> str:
    return SKILL.read_text(encoding="utf-8")


def test_skill_file_exists():
    assert SKILL.is_file(), f"missing {SKILL}"


def test_frontmatter_parses(text):
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert m, "SKILL.md must open with a YAML frontmatter block"


@pytest.fixture(scope="module")
def frontmatter(text) -> str:
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert m, "SKILL.md must open with a YAML frontmatter block"
    return m.group(1)


def test_description_within_budget(frontmatter):
    m = re.search(r'^description: "?([^"\n]*)"?$', frontmatter, re.M)
    assert m, "description required in frontmatter"
    desc = m.group(1).strip()
    assert len(desc) <= 60, f"description is {len(desc)} chars (<=60 required)"


def test_description_ends_with_period(frontmatter):
    m = re.search(r'^description: "?([^"\n]*)"?$', frontmatter, re.M)
    desc = m.group(1).strip()
    assert desc.endswith("."), "description must end with a period"


def test_no_marketing_words(text):
    body = text.split("---", 2)[-1]
    bad = re.findall(
        r"\b(powerful|comprehensive|seamless|revolutionary|cutting-edge|state-of-the-art)\b",
        body,
        re.I,
    )
    assert not bad, f"marketing words: {bad}"


def test_no_machine_local_paths(text):
    bad = re.findall(r"/home/(?!runner\b)[a-z0-9_-]+/", text)
    assert not bad, f"machine-local paths leak environment: {bad}"


def test_no_project_specific_names(text):
    """The skill is generic doctrine — project/incident names stay out of doctrine.

    The `author:` frontmatter line is exempt: attribution rules require naming
    where the skill was field-verified.
    """
    body = "\n".join(line for line in text.splitlines() if not line.startswith("author:"))
    bad = re.findall(r"\b(HermesForge|hermesforge|physics-playground|PHYSICS-DEPTH)\b", body)
    assert not bad, f"project-specific names in doctrine: {bad}"


def test_required_sections_present(text):
    for section in ("## When to Use", "## Pitfalls", "## Verification"):
        assert section in text, f"missing section: {section}"


def test_version_and_author_present(text):
    assert re.search(r"^version: \d+\.\d+\.\d+$", text, re.M), "version required"
    assert re.search(r"^author: .+", text, re.M), "author required"
