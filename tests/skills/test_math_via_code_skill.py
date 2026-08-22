"""Smoke tests for the math-via-code bundled skill.

No live code execution — these assert SKILL.md frontmatter and body
contracts called out in AGENTS.md skill authoring standards and the
PR review for this skill.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "software-development"
    / "math-via-code"
)
SKILL_MD = SKILL_DIR / "SKILL.md"

REQUIRED_SECTIONS = (
    "## When to Use",
    "## Prerequisites",
    "## How to Run",
    "## Quick Reference",
    "## Procedure",
    "## Pitfalls",
    "## Verification",
)


@pytest.fixture(scope="module")
def skill_src() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontmatter(skill_src: str) -> dict:
    m = re.search(r"^---\n(.*?)\n---", skill_src, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


def test_skill_dir_exists() -> None:
    assert SKILL_DIR.is_dir(), f"missing skill dir: {SKILL_DIR}"


def test_skill_md_present() -> None:
    assert SKILL_MD.is_file()


def test_description_under_60_chars(frontmatter: dict) -> None:
    desc = frontmatter["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars (hardline ≤60): {desc!r}"


def test_description_ends_with_period(frontmatter: dict) -> None:
    desc = frontmatter["description"]
    assert desc.endswith("."), f"description must end with a period: {desc!r}"


def test_name_matches_dir(frontmatter: dict) -> None:
    assert frontmatter["name"] == "math-via-code"


def test_platforms_include_windows(frontmatter: dict) -> None:
    platforms = set(frontmatter["platforms"])
    assert platforms >= {"linux", "macos", "windows"}


def test_author_credits_contributor(frontmatter: dict) -> None:
    author = frontmatter["author"]
    assert "Tom Mulkins" in author, f"author should credit the contributor: {author!r}"


def test_license_mit(frontmatter: dict) -> None:
    assert frontmatter["license"] == "MIT"


def test_modern_sections_present(skill_src: str) -> None:
    for heading in REQUIRED_SECTIONS:
        assert heading in skill_src, f"missing required section: {heading}"


def test_title_uses_skill_suffix(skill_src: str) -> None:
    assert re.search(r"^# Math Via Code Skill\s*$", skill_src, re.MULTILINE)


def test_assign_and_assert_examples_execute(skill_src: str) -> None:
    """Self-contained examples (assignments + asserts) must run without NameError."""
    fences = re.findall(r"```python\n(.*?)```", skill_src, re.DOTALL)
    runnable = [
        block
        for block in fences
        if re.search(r"^\w+\s*=", block, re.MULTILINE) and "assert " in block
    ]
    assert runnable, "expected at least one assign+assert example fence"
    for i, block in enumerate(runnable):
        exec(compile(block, f"<math-via-code-example-{i}>", "exec"), {})


def test_no_hardcoded_tmp_paths(skill_src: str) -> None:
    """Windows is declared; guidance must use tempfile.gettempdir()."""
    assert "tempfile.gettempdir()" in skill_src
    assert "/tmp" not in skill_src


def test_references_native_tools(skill_src: str) -> None:
    assert "`execute_code`" in skill_src
    assert "`terminal`" in skill_src
