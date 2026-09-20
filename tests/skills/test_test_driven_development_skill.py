"""Dual Gate contracts for the bundled test-driven-development skill.

TDD starts at the design phase: the plan names cases; Coder writes the
test files in RED; Planner writes test files into the target repo only
for M+ public contracts when the brief asks. A plan that defers tests is
not a plan. These tests pin that section so a factory copy-paste of the
pre-Dual-Gate skill cannot silently return.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_PATH = (
    REPO_ROOT / "skills" / "software-development" / "test-driven-development" / "SKILL.md"
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


def test_design_phase_section_is_present():
    _, body = _frontmatter_and_body()
    assert "## TDD Starts at the Design Phase" in body


def test_plan_names_cases_coder_writes_files():
    _, body = _frontmatter_and_body()
    assert "Coder writes those" in body and "test files" in body
    assert "Planner writes test files into" in body
    assert "no new test: reason" in body


def test_deferred_tests_are_not_a_plan():
    _, body = _frontmatter_and_body()
    assert "tests will be written later" in body


def test_does_not_require_test_files_to_exist_when_the_plan_is_saved():
    """AGENTS.md once claimed the RED files exist at plan-save time.

    Dual Gate forbids that: the plan names cases; Coder writes the files.
    The bundled TDD skill must not reintroduce the sabotaging sentence.
    """
    content = SKILL_PATH.read_text(encoding="utf-8")
    assert "when the plan is saved, the test files exist" not in content
    assert "test files exist in this repo" not in content


def test_delegate_example_does_not_tell_the_child_to_commit():
    _, body = _frontmatter_and_body()
    # Git is Orchestrator-owned; a leaf Coder example must not teach commits.
    assert "6. Commit" not in body
