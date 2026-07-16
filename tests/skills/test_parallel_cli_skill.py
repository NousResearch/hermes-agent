"""Static contract tests for the optional Parallel CLI skill."""

from __future__ import annotations

import re
from pathlib import Path

import pytest


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
def frontmatter(source: str) -> str:
    match = re.search(r"^---\n(.*?)\n---", source, re.DOTALL)
    assert match, "SKILL.md missing YAML frontmatter"
    return match.group(1)


def _frontmatter_value(frontmatter: str, field: str) -> str:
    match = re.search(rf"^{re.escape(field)}:\s*(.+)$", frontmatter, re.MULTILINE)
    assert match, f"SKILL.md missing {field!r} frontmatter"
    return match.group(1).strip()


def test_frontmatter_matches_hermes_rules(frontmatter: str) -> None:
    assert _frontmatter_value(frontmatter, "name") == "parallel-cli"
    description = _frontmatter_value(frontmatter, "description")
    assert len(description) <= 60
    assert description.endswith(".")
    assert description.startswith("Optional ")
    assert (
        _frontmatter_value(frontmatter, "author").split(",", 1)[0]
        == "George Pickett (@grp06)"
    )
    platforms = _frontmatter_value(frontmatter, "platforms").strip("[]").split(",")
    assert {platform.strip() for platform in platforms} == {"linux", "macos", "windows"}


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


def test_bash_blocks_do_not_use_shell_redirection_placeholders(source: str) -> None:
    blocks = re.findall(r"```bash\n(.*?)\n```", source, re.DOTALL)
    for block in blocks:
        assert not re.search(r"<[a-z_][a-z0-9_-]*>", block, re.IGNORECASE), block


def test_saved_search_and_extract_results_use_the_file_as_source_of_truth(
    source: str,
) -> None:
    blocks = re.findall(r"```bash\n(.*?)\n```", source, re.DOTALL)
    saved_result_blocks = [
        block
        for block in blocks
        if re.search(r"parallel-cli (search|extract)\b", block)
        and re.search(r"(^|\s)-o\s", block)
    ]
    assert saved_result_blocks
    for block in saved_result_blocks:
        assert "--json" not in block, block


def test_research_examples_keep_run_and_interaction_ids_distinct(source: str) -> None:
    assert "research status trun_run_xxx" in source
    assert "research poll trun_run_xxx" in source
    assert "--previous-interaction-id trun_interaction_xxx" in source


def test_monitor_read_and_mutation_examples_are_separate(source: str) -> None:
    blocks = re.findall(r"```bash\n(.*?)\n```", source, re.DOTALL)
    monitor_blocks = "\n".join(
        block for block in blocks if "parallel-cli monitor" in block
    )
    assert monitor_blocks
    for mutation in ("update", "trigger", "cancel"):
        assert f"parallel-cli monitor {mutation}" not in monitor_blocks


@pytest.mark.parametrize(
    "required",
    [
        "parallel-cli login --no-browser --json",
        'terminal(command="parallel-cli login --no-browser --json", background=true, notify_on_complete=true)',
        'process(action="poll", session_id="...")',
        "parallel-cli findall entity-search",
        "parallel-cli monitor cancel",
        "parallel-cli monitor trigger",
        "`run_id`, `interaction_id`",
        "`taskgroup_id`",
        "`findall_id`",
        "`monitor_id`, `event_group_id`, `next_cursor`",
        "JSON-designated output file",
        "text-schema research report",
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
