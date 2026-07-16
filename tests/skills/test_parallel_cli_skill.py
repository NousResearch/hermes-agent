"""Static contract tests for the optional Parallel CLI skill."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml


SKILL_MD = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "research"
    / "parallel-cli"
    / "SKILL.md"
)


@pytest.fixture(scope="module")
def source() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontmatter(source: str) -> dict:
    match = re.search(r"^---\n(.*?)\n---", source, re.DOTALL)
    assert match, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(match.group(1))


def test_frontmatter_matches_hermes_rules(frontmatter: dict) -> None:
    assert frontmatter["name"] == "parallel-cli"
    assert len(frontmatter["description"]) <= 60
    assert frontmatter["description"].endswith(".")
    assert frontmatter["author"].split(",", 1)[0] == "George Pickett (@grp06)"
    assert set(frontmatter["platforms"]) == {"linux", "macos", "windows"}


def test_modern_section_order(source: str) -> None:
    sections = [
        "# Parallel CLI Skill",
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]
    offsets = [source.index(section) for section in sections]
    assert offsets == sorted(offsets)
    assert len(source.splitlines()) <= 500


@pytest.mark.parametrize(
    "required",
    [
        "parallel-cli login --no-browser --json",
        'terminal(command="parallel-cli login --no-browser --json", background=true, notify_on_complete=true)',
        'process(action="poll", session_id="...")',
        "--excerpt-max-chars-total 27000",
        "parallel-cli findall entity-search",
        "parallel-cli monitor cancel",
        "parallel-cli monitor trigger",
        "`run_id`, `interaction_id`",
        "`taskgroup_id`",
        "`findall_id`",
        "`monitor_id`, `event_group_id`, `next_cursor`",
    ],
)
def test_current_agent_workflows_are_documented(source: str, required: str) -> None:
    assert required in source


@pytest.mark.parametrize(
    "stale",
    [
        "login --device",
        "--mode one-shot",
        "--mode agentic",
        "monitor delete",
        "monitor event-group",
        "monitor simulate",
        "<task_group_id>",
        "findall status <run_id>",
        "/tmp/",
    ],
)
def test_stale_or_platform_specific_forms_are_absent(source: str, stale: str) -> None:
    assert stale not in source
