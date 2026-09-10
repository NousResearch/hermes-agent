"""Contract tests for the bundled Humanizer skill."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from agent.skill_utils import parse_frontmatter


REPO = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO / "skills" / "creative" / "humanizer"
SKILL_PATH = SKILL_DIR / "SKILL.md"
GENERATED_PAGE = (
    REPO
    / "website"
    / "docs"
    / "user-guide"
    / "skills"
    / "bundled"
    / "creative"
    / "creative-humanizer.md"
)
GENERATOR = REPO / "website" / "scripts" / "generate-skill-docs.py"


@pytest.fixture(scope="module")
def skill_text() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def generator_module():
    spec = importlib.util.spec_from_file_location("generate_skill_docs", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_fidelity_contract_precedes_pattern_catalog(skill_text):
    catalog_start = skill_text.index("## CONTENT PATTERNS")
    safeguards = (
        "Keep every substantive claim",
        "Do not invent",
        "The sample takes priority",
        "add no substantive proposition",
        "unsupported addition",
        "lost or distorted substantive claim",
    )

    for safeguard in safeguards:
        assert 0 <= skill_text.index(safeguard) < catalog_start


def test_evidence_boundary_and_modes_are_explicit(skill_text):
    for expected in (
        "not proof of authorship",
        "A valid outcome is to leave the text unchanged",
        "Pasted text (default)",
        "File mode",
        "Embedded mode",
        "Do not expose an intermediate draft",
    ):
        assert expected in skill_text


def test_pattern_catalog_stays_at_34_entries(skill_text):
    numbers = [
        int(number)
        for number in re.findall(r"^### (\d+)\.", skill_text, flags=re.MULTILINE)
    ]
    assert numbers == list(range(1, 35))


def test_unsafe_legacy_guidance_does_not_return(skill_text):
    forbidden = (
        "Add soul.",
        "Report the facts, then react to them",
        'Use "I" when it fits',
        "default behavior (natural, varied, opinionated voice",
        "ChatGPT uses curly quotes",
    )

    for phrase in forbidden:
        assert phrase.casefold() not in skill_text.casefold()


def test_fabricated_worked_example_is_removed(skill_text):
    assert "## Full Example" not in skill_text
    for artifact in (
        "a 2024 study by Google",
        "55% faster",
        "Mira, an engineer",
        "Jake, a senior dev",
        "The 2024 Uplevel study",
    ):
        assert artifact not in skill_text


def test_known_catalog_fabrications_do_not_return(skill_text):
    for artifact in (
        "collect and publish regional statistics independently",
        "In a 2024 New York Times interview",
        "The architect said",
        "weekly market and 18th-century church",
        "Chinese Academy of Sciences",
        "three new IT parks",
        "rooms totaling 3,000 square feet",
        "considered a delicacy",
        "speeds up load times",
        "founded in 1994",
        "two more locations next year",
        "including request memoization",
        "features users are asking for",
        "flat monthly fee",
        "regressions begin costing real time",
    ):
        assert artifact not in skill_text


def test_skill_remains_a_loadable_instruction_only_package(skill_text):
    frontmatter, body = parse_frontmatter(skill_text)
    assert frontmatter["name"] == "humanizer"
    assert frontmatter["description"] == "Humanize text without changing its meaning."
    assert body.startswith("# Humanizer:")

    executable_suffixes = {
        ".bash",
        ".cjs",
        ".js",
        ".mjs",
        ".py",
        ".sh",
        ".ts",
    }
    runtime_files = [
        path.relative_to(SKILL_DIR)
        for path in SKILL_DIR.rglob("*")
        if path.is_file() and path.suffix in executable_suffixes
    ]
    assert runtime_files == []


def test_generated_humanizer_page_matches_skill_source(generator_module):
    entries = generator_module.discover_skills()
    skill_index = {}
    target = None
    for meta, parsed in entries:
        name = parsed["frontmatter"].get("name", meta["slug"])
        if name not in skill_index or meta["source_kind"] == "bundled":
            skill_index[name] = meta
        if meta["source_kind"] == "bundled" and name == "humanizer":
            target = (meta, parsed)

    assert target is not None
    meta, parsed = target
    rendered = generator_module.render_skill_page(
        meta,
        parsed["frontmatter"],
        parsed["body"],
        skill_index=skill_index,
    )
    assert GENERATED_PAGE.read_text(encoding="utf-8") == rendered
