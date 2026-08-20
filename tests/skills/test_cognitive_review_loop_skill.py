"""Standards tests for the cognitive-review-loop optional skill.

The skill is prose-only (no scripts), so these tests pin the contributed
SKILL.md to the hardline authoring standards in AGENTS.md: frontmatter shape,
the 60-char description budget, the modern section order, and wrapper-only
pytest invocations.

Stdlib + pytest only, per the skill-test rule in AGENTS.md, so the frontmatter
is read by the narrow parser below instead of PyYAML.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

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


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _parse_value(raw: str, lineno: int) -> str | list[str]:
    value = raw.strip()
    if value in (">", "|", ">-", "|-", ">+", "|+"):
        raise AssertionError(
            f"line {lineno}: folded/literal blocks are not allowed in skill frontmatter"
        )
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [_unquote(part.strip()) for part in inner.split(",")] if inner else []
    return _unquote(value)


def _parse_frontmatter(block: str) -> dict:
    """Parse the fixed frontmatter subset: scalars, inline lists, nested maps.

    Deliberately narrow. Anything outside the supported subset raises rather
    than silently mis-parsing, so a malformed SKILL.md fails loudly here
    instead of quietly passing the standards assertions below.
    """
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]

    for lineno, line in enumerate(block.splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        if stripped.startswith("- "):
            raise AssertionError(f"line {lineno}: block sequences unsupported, use [a, b]")
        if ":" not in stripped:
            raise AssertionError(f"line {lineno}: unparsable frontmatter line {line!r}")

        key, _, raw = stripped.partition(":")
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if raw.strip():
            parent[key.strip()] = _parse_value(raw, lineno)
        else:
            child: dict = {}
            parent[key.strip()] = child
            stack.append((indent, child))

    return root


@pytest.fixture(scope="module")
def skill_text() -> str:
    return (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontmatter(skill_text: str) -> dict:
    m = re.search(r"^---\n(.*?)\n---", skill_text, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return _parse_frontmatter(m.group(1))


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
