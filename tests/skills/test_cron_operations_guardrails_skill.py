"""Structural tests for the cron-operations-guardrails skill.

Stdlib + pytest only, per AGENTS.md §7 ("stdlib + pytest + unittest.mock").
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


SKILL_MD = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "autonomous-ai-agents"
    / "cron-operations-guardrails"
    / "SKILL.md"
)


@pytest.fixture(scope="module")
def content() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontmatter(content: str) -> dict[str, str]:
    match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
    assert match, "SKILL.md must start with YAML frontmatter"

    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if line.startswith((" ", "\t")) or ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()
    return fields


def test_skill_file_exists() -> None:
    assert SKILL_MD.is_file()


def test_frontmatter_invariants(frontmatter: dict[str, str]) -> None:
    assert frontmatter["name"] == "cron-operations-guardrails"

    description = frontmatter["description"]
    assert len(description) <= 60
    assert description.endswith(".")
    assert description.count(".") == 1

    author = frontmatter["author"]
    assert "Community Contributor" not in author
    assert "goodchang77" in author
    assert "Hermes Agent" in author
    assert author.index("goodchang77") < author.index("Hermes Agent"), (
        "human contributor must be credited before Hermes Agent"
    )

    assert frontmatter["platforms"] == "[macos, linux, windows]"


def test_category_matches_directory(content: str) -> None:
    match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
    assert match
    category = re.search(r"^\s+category:\s*(\S+)\s*$", match.group(1), re.MULTILINE)
    assert category, "frontmatter must declare metadata.hermes.category"
    assert category.group(1) == "autonomous-ai-agents"
    assert SKILL_MD.parent.parent.name == category.group(1)


def test_modern_section_order(content: str) -> None:
    expected = [
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]
    positions = []
    for section in expected:
        match = re.search(rf"^{re.escape(section)}$", content, re.MULTILINE)
        assert match, f"missing required section: {section}"
        positions.append(match.start())
    assert positions == sorted(positions)
    for section in expected:
        assert len(re.findall(rf"^{re.escape(section)}$", content, re.MULTILINE)) == 1


def test_intro_states_scope_and_safety(content: str) -> None:
    title = "# Hermes Cron Operations Guardrails Skill"
    assert title in content
    intro = content.split(title, 1)[1].split("## When to Use", 1)[0]
    assert "Diagnose" in intro
    assert "read-only by default" in intro
    assert "explicit approval" in intro


def test_required_operational_contracts(content: str) -> None:
    for command in (
        "hermes cron status",
        "hermes cron list --all",
        "hermes status --all",
    ):
        assert command in content

    assert "Healthy | Empty | 0" in content
    assert "Alert generated | One concise alert | 0" in content
    assert "Monitor implementation failed | Diagnostic only | Non-zero" in content
    assert "Exclude the monitor itself" in content


def test_reporting_template_under_verification(content: str) -> None:
    template = content.split("## Verification", 1)[1]
    assert "- Severity: [LOW|MEDIUM|HIGH|CRITICAL]" in template
    for heading in (
        "### Incident",
        "### Evidence",
        "### Root cause",
        "### Remediation",
        "### Verification",
    ):
        assert heading in template


def test_no_obvious_secret_placeholders(content: str) -> None:
    forbidden = ("sk-", "ghp_", "xoxb-", "BEGIN PRIVATE KEY")
    assert not any(marker in content for marker in forbidden)
