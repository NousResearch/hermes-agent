"""Tests for the Obsidian skill bundle under skills/note-taking/.

Invariant-style checks (no data snapshots): frontmatter contracts from the
skill authoring standards, routing coherence between the generic wrapper and
the specialized skills, reference-link and attribution integrity, catalog
coverage, and unit tests for the bundled JSON Canvas validator script.

The bundle members are enumerated explicitly so that adding an unrelated
note-taking skill later does not drag it into bundle-specific contracts
(authorship, routing, attribution).
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_ROOT = REPO_ROOT / "skills"
BUNDLE_ROOT = SKILLS_ROOT / "note-taking"
SKILLS_CATALOG = REPO_ROOT / "website" / "docs" / "reference" / "skills-catalog.md"
THIRD_PARTY_NOTICES = BUNDLE_ROOT / "THIRD_PARTY_NOTICES.md"
VALIDATOR_PATH = BUNDLE_ROOT / "json-canvas" / "scripts" / "validate_canvas.py"

# The Obsidian bundle salvaged from PR #15848. Explicit on purpose — see
# module docstring.
SPECIALIZED_SKILLS = [
    "json-canvas",
    "obsidian-bases",
    "obsidian-cli",
    "obsidian-markdown",
]
BUNDLE_SKILLS = ["obsidian", *SPECIALIZED_SKILLS]

MARKETING_WORDS = {"powerful", "comprehensive", "seamless", "advanced"}


def _frontmatter_field(text: str, field: str) -> str | None:
    match = re.search(rf"^{field}: (.*)$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def _skill_text(name: str) -> str:
    return (BUNDLE_ROOT / name / "SKILL.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("name", BUNDLE_SKILLS)
def test_bundle_skill_exists(name):
    assert (BUNDLE_ROOT / name / "SKILL.md").is_file(), f"missing skill {name}"


@pytest.mark.parametrize("name", BUNDLE_SKILLS)
def test_description_meets_authoring_standard(name):
    description = _frontmatter_field(_skill_text(name), "description")
    assert description, f"{name}: missing description"
    assert len(description) <= 60, (
        f"{name}: description is {len(description)} chars (limit 60)"
    )
    assert description.endswith("."), f"{name}: description must end with a period"
    assert ". " not in description[:-1], f"{name}: description must be one sentence"
    lowered = description.lower()
    hits = sorted(w for w in MARKETING_WORDS if w in lowered)
    assert not hits, f"{name}: marketing words in description: {hits}"


@pytest.mark.parametrize("name", SPECIALIZED_SKILLS)
def test_description_does_not_repeat_skill_name(name):
    description = _frontmatter_field(_skill_text(name), "description")
    spaced = name.replace("-", " ").lower()
    assert spaced not in description.lower(), (
        f"{name}: description repeats the skill name"
    )


@pytest.mark.parametrize("name", BUNDLE_SKILLS)
def test_platforms_declared(name):
    platforms = _frontmatter_field(_skill_text(name), "platforms")
    assert platforms, f"{name}: missing platforms frontmatter"
    entries = {p.strip() for p in platforms.strip("[]").split(",")}
    assert entries <= {"linux", "macos", "windows"}, (
        f"{name}: unknown platform entries {entries}"
    )


@pytest.mark.parametrize("name", SPECIALIZED_SKILLS)
def test_adapted_skills_credit_human_contributor_first(name):
    author = _frontmatter_field(_skill_text(name), "author")
    assert author, f"{name}: missing author"
    # Contract: the agent is never the primary author of an adapted skill.
    assert not author.lower().startswith("hermes agent"), (
        f"{name}: author must credit the human contributor first, got {author!r}"
    )


@pytest.mark.parametrize("name", SPECIALIZED_SKILLS)
def test_adapted_skills_carry_upstream_attribution(name):
    text = _skill_text(name)
    assert "kepano/obsidian-skills" in text, f"{name}: missing source attribution"
    assert "THIRD_PARTY_NOTICES.md" in text, (
        f"{name}: missing pointer to the bundled MIT notice"
    )


def test_third_party_notice_preserves_upstream_mit_text():
    notice = THIRD_PARTY_NOTICES.read_text(encoding="utf-8")
    assert "Copyright (c) 2026 Steph Ango (@kepano)" in notice
    assert "Permission is hereby granted, free of charge" in notice


@pytest.mark.parametrize("name", SPECIALIZED_SKILLS)
def test_modern_section_order(name):
    text = _skill_text(name)
    required = [
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]
    positions = [text.find(section) for section in required]
    missing = [s for s, pos in zip(required, positions) if pos == -1]
    assert not missing, f"{name}: missing sections {missing}"
    assert positions == sorted(positions), (
        f"{name}: sections out of order: {list(zip(required, positions))}"
    )


def test_wrapper_routes_to_every_specialized_skill():
    wrapper = _skill_text("obsidian")
    missing = [name for name in SPECIALIZED_SKILLS if name not in wrapper]
    assert not missing, f"obsidian wrapper missing routing guidance for: {missing}"


@pytest.mark.parametrize("name", BUNDLE_SKILLS)
def test_related_skills_resolve_to_bundled_skills(name):
    related = _frontmatter_field(_skill_text(name), "    related_skills")
    assert related, f"{name}: missing related_skills"
    entries = [e.strip() for e in related.strip("[]").split(",")]
    all_bundled = {p.parent.name for p in SKILLS_ROOT.glob("*/*/SKILL.md")}
    unresolved = [e for e in entries if e not in all_bundled]
    assert not unresolved, f"{name}: related_skills do not exist: {unresolved}"


@pytest.mark.parametrize("name", BUNDLE_SKILLS)
def test_relative_links_resolve(name):
    skill_dir = BUNDLE_ROOT / name
    text = _skill_text(name)
    targets = re.findall(r"\]\(((?:\.\./|references/|scripts/)[^)#]+)\)", text)
    broken = [t for t in targets if not (skill_dir / t).is_file()]
    assert not broken, f"{name}: broken relative links: {broken}"


def test_catalog_lists_every_bundle_skill():
    catalog = SKILLS_CATALOG.read_text(encoding="utf-8")
    missing = [
        name
        for name in BUNDLE_SKILLS
        if f"note-taking/{name}`" not in catalog
    ]
    assert not missing, f"skills-catalog.md is missing rows for: {missing}"


# --- validate_canvas.py -----------------------------------------------------


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_canvas", VALIDATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _valid_canvas() -> dict:
    return {
        "nodes": [
            {
                "id": "6f0ad84f44ce9c17",
                "type": "text",
                "x": 0,
                "y": 0,
                "width": 300,
                "height": 150,
                "text": "# Hello",
            },
            {
                "id": "a1b2c3d4e5f67890",
                "type": "link",
                "x": 400,
                "y": 0,
                "width": 300,
                "height": 100,
                "url": "https://obsidian.md",
                "color": "4",
            },
            {
                "id": "feedfacefeedface",
                "type": "group",
                "x": -50,
                "y": -50,
                "width": 900,
                "height": 400,
                "label": "Everything",
            },
        ],
        "edges": [
            {
                "id": "0123456789abcdef",
                "fromNode": "6f0ad84f44ce9c17",
                "toNode": "a1b2c3d4e5f67890",
                "fromSide": "right",
                "toSide": "left",
                "toEnd": "arrow",
                "label": "leads to",
            }
        ],
    }


def _errors_for(canvas) -> list[str]:
    mod = _load_validator()
    return mod.validate_canvas_text(json.dumps(canvas))


def test_validator_accepts_valid_canvas():
    assert _errors_for(_valid_canvas()) == []


def test_validator_accepts_empty_canvas():
    mod = _load_validator()
    assert mod.validate_canvas_text('{"nodes": [], "edges": []}') == []
    assert mod.validate_canvas_text("{}") == []


@pytest.mark.parametrize(
    "color", ["#F00", "#F00A", "#FF0000", "#FF0000AA", "1", "6"]
)
def test_validator_accepts_spec_valid_colors(color):
    canvas = _valid_canvas()
    canvas["nodes"][0]["color"] = color
    assert _errors_for(canvas) == []


@pytest.mark.parametrize("color", ["#FF00 0", "#FF000", "0", "7", "red", 4, ["#F00"]])
def test_validator_rejects_invalid_colors(color):
    canvas = _valid_canvas()
    canvas["nodes"][0]["color"] = color
    assert any("color must be a preset" in e for e in _errors_for(canvas))


def test_validator_rejects_malformed_json():
    mod = _load_validator()
    errors = mod.validate_canvas_text('{"nodes": [')
    assert len(errors) == 1 and "invalid JSON" in errors[0]


def test_validator_rejects_duplicate_ids():
    canvas = _valid_canvas()
    canvas["nodes"][1]["id"] = canvas["nodes"][0]["id"]
    assert any("duplicate id" in e for e in _errors_for(canvas))


def test_validator_rejects_dangling_edge_reference():
    canvas = _valid_canvas()
    canvas["edges"][0]["toNode"] = "0000000000000000"
    assert any(
        "does not reference an existing node" in e for e in _errors_for(canvas)
    )


def test_validator_rejects_unknown_node_type_and_bad_enum():
    canvas = _valid_canvas()
    canvas["nodes"][0]["type"] = "widget"
    canvas["edges"][0]["fromSide"] = "diagonal"
    errors = _errors_for(canvas)
    assert any("'type' must be one of" in e for e in errors)
    assert any("'fromSide'" in e for e in errors)


@pytest.mark.parametrize("hostile", [[], {}, 7, True, ["top"]])
def test_validator_reports_non_string_enums_without_crashing(hostile):
    # Unhashable or otherwise non-string enum values must produce validation
    # errors, not a TypeError from set membership.
    canvas = _valid_canvas()
    canvas["nodes"][0]["type"] = hostile
    canvas["nodes"][2]["backgroundStyle"] = hostile
    canvas["edges"][0]["toEnd"] = hostile
    errors = _errors_for(canvas)
    assert any("'type' must be one of" in e for e in errors)
    assert any("'backgroundStyle'" in e for e in errors)
    assert any("'toEnd'" in e for e in errors)


def test_validator_rejects_missing_type_specific_field():
    canvas = _valid_canvas()
    del canvas["nodes"][1]["url"]
    assert any(
        "link node requires string field 'url'" in e for e in _errors_for(canvas)
    )


@pytest.mark.parametrize("bad_value", ["0", 1.5, True, None])
def test_validator_rejects_non_integer_geometry(bad_value):
    canvas = _valid_canvas()
    canvas["nodes"][0]["x"] = bad_value
    assert any(
        "non-integer required field 'x'" in e for e in _errors_for(canvas)
    )


def test_validator_rejects_non_string_labels():
    canvas = _valid_canvas()
    canvas["nodes"][2]["label"] = 42
    canvas["edges"][0]["label"] = ["not", "a", "string"]
    errors = _errors_for(canvas)
    assert sum("'label' must be a string" in e for e in errors) == 2


def test_validator_cli_exit_codes(tmp_path):
    mod = _load_validator()
    good = tmp_path / "good.canvas"
    good.write_text(json.dumps(_valid_canvas()), encoding="utf-8")
    bad = tmp_path / "bad.canvas"
    bad.write_text("not json", encoding="utf-8")
    assert mod.main([str(good)]) == 0
    assert mod.main([str(bad)]) == 1
    assert mod.main([]) == 2
