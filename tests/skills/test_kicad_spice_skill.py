"""Portable content contracts for the kicad-spice optional skill."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "research"
    / "kicad-spice"
)


@pytest.fixture(scope="module")
def skill():
    path = SKILL_DIR / "SKILL.md"
    assert path.is_file(), f"missing skill: {path}"
    source = path.read_text(encoding="utf-8")
    match = re.match(r"\A---\n(.*?)\n---\n(.*)\Z", source, re.DOTALL)
    assert match, "SKILL.md missing or unclosed frontmatter"

    # This skill deliberately uses flat YAML scalars and inline lists.
    # Parse that limited form with stdlib, without adding a YAML dependency.
    fields = {}
    for line in match[1].splitlines():
        entry = re.fullmatch(r"([a-z][a-z_]*):[ \t]+(.+)", line)
        assert entry, f"unsupported frontmatter entry: {line!r}"
        key, value = entry.groups()
        assert key not in fields, f"duplicate frontmatter field: {key}"
        fields[key] = value.strip().strip("\"'")
    return fields, match[2]


def test_portable_skill_metadata_and_resources(skill):
    fields, _ = skill
    for key in ("name", "description", "version", "author", "license", "platforms"):
        assert fields.get(key), f"missing required field: {key}"
    assert fields["name"] == "kicad-spice"
    description = fields["description"]
    assert len(description) <= 60
    assert description.endswith(".")
    assert not fields["author"].casefold().startswith("hermes agent")
    assert fields["author"].startswith("Snehal")

    reference = SKILL_DIR / "references" / "filter_math.md"
    assert reference.is_file(), f"missing reference: {reference}"
    assert "references/filter_math.md" in (SKILL_DIR / "SKILL.md").read_text(
        encoding="utf-8"
    )


def test_workflow_structure_and_portability(skill):
    _, body = skill
    assert re.search(r"^# .+ Skill$", body, re.MULTILINE)
    headings = re.findall(r"^## (.+)$", body, re.MULTILINE)
    required = [
        "When to Use",
        "Prerequisites",
        "How to Run",
        "Quick Reference",
        "Procedure",
        "Pitfalls",
        "Verification",
    ]
    assert all(heading in headings for heading in required)
    positions = [headings.index(heading) for heading in required]
    assert positions == sorted(positions)
    assert "terminal" in body
    # Guard against the private application dependency without embedding
    # its full identifier in the portable skill's patch.
    private_module = "faraday" + "_pipeline"
    assert private_module not in body
    reference = (SKILL_DIR / "references" / "filter_math.md").read_text(
        encoding="utf-8"
    )
    assert private_module not in reference
