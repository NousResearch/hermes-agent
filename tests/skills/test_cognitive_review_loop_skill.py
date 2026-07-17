"""Standards tests for the cognitive-review-loop optional skill.

The skill is prose-only (no scripts), so these tests pin the contributed
SKILL.md to the hardline authoring standards in AGENTS.md: frontmatter shape,
the 60-char description budget, the modern section order, and wrapper-only
pytest invocations.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "software-development"
    / "cognitive-review-loop"
)

MARKETING_WORDS = ("powerful", "comprehensive", "seamless", "advanced")

REQUIRED_SECTIONS = [
    "## When to Use",
    "## Prerequisites",
    "## How to Run",
    "## Quick Reference",
    "## Procedure",
    "## Pitfalls",
    "## Verification",
]


@pytest.fixture(scope="module")
def skill_text() -> str:
    return (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontmatter(skill_text: str) -> dict:
    m = re.search(r"^---\n(.*?)\n---", skill_text, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


def test_skill_md_present() -> None:
    assert (SKILL_DIR / "SKILL.md").is_file()


def test_name_matches_dir(frontmatter: dict) -> None:
    assert frontmatter["name"] == "cognitive-review-loop"


def test_description_hardline(frontmatter: dict) -> None:
    desc = frontmatter["description"]
    assert isinstance(desc, str), "description must be a plain string, not a folded block"
    assert len(desc) <= 60, f"description is {len(desc)} chars (hardline <=60): {desc!r}"
    assert desc.endswith("."), "description must end with a period"
    assert ". " not in desc, "description must be a single sentence"
    lowered = desc.lower()
    assert not any(w in lowered for w in MARKETING_WORDS), "no marketing words in description"
    assert "cognitive-review-loop" not in lowered, "description must not repeat the skill name"


def test_platforms_all_three(frontmatter: dict) -> None:
    # Prose-only skill: nothing platform-bound, so all three are declared.
    assert set(frontmatter["platforms"]) == {"linux", "macos", "windows"}


def test_author_credits_contributor(frontmatter: dict) -> None:
    assert "TheSmokeDev" in frontmatter["author"]


def test_license_mit(frontmatter: dict) -> None:
    assert frontmatter["license"] == "MIT"


def test_related_skills_exist_in_repo(frontmatter: dict) -> None:
    repo_root = SKILL_DIR.parents[2]
    for related in frontmatter["metadata"]["hermes"]["related_skills"]:
        matches = list(repo_root.glob(f"skills/**/{related}/SKILL.md")) + list(
            repo_root.glob(f"optional-skills/**/{related}/SKILL.md")
        )
        assert matches, f"related skill does not exist in repo: {related!r}"


def test_modern_section_order(skill_text: str) -> None:
    positions = [skill_text.find(h) for h in REQUIRED_SECTIONS]
    missing = [h for h, p in zip(REQUIRED_SECTIONS, positions) if p == -1]
    assert not missing, f"missing required sections: {missing}"
    assert positions == sorted(positions), "sections out of the AGENTS.md order"


def test_no_direct_pytest_invocation(skill_text: str) -> None:
    # AGENTS.md: always scripts/run_tests.sh, never bare pytest.
    assert "python -m pytest" not in skill_text
    assert "scripts/run_tests.sh" in skill_text


def test_line_budget(skill_text: str) -> None:
    # ~100 lines for a simple skill, ~200 for a complex one; this is a simple one.
    assert len(skill_text.splitlines()) <= 200
