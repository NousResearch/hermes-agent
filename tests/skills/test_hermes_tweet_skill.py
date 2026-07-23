"""Focused contract tests for the Hermes Tweet optional skill."""

from __future__ import annotations

import re
from pathlib import Path


SKILL_PATH = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "social-media"
    / "hermes-tweet"
    / "SKILL.md"
)


def _skill_text() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


def _frontmatter_value(key: str) -> str:
    match = re.search(rf"^{re.escape(key)}: (.+)$", _skill_text(), re.MULTILINE)
    assert match is not None, f"missing {key!r} frontmatter"
    return match.group(1)


def test_description_matches_hardline_contract() -> None:
    description = _frontmatter_value("description")

    assert len(description) <= 60
    assert description.endswith(".")
    assert description.count(".") == 1


def test_author_credits_human_contributor_first() -> None:
    assert _frontmatter_value("author").startswith("Burak Bayır (kriptoburak)")


def test_required_sections_use_modern_order() -> None:
    text = _skill_text()
    sections = [
        "# Hermes Tweet Skill",
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]

    positions = [text.index(section) for section in sections]
    assert positions == sorted(positions)


def test_environment_refresh_guidance_distinguishes_surfaces() -> None:
    text = _skill_text()

    assert "active CLI session, run `/reload`" in text
    assert "gateway use, run `hermes gateway restart`, then start a new session" in text


def test_read_and_action_gates_remain_explicit() -> None:
    text = _skill_text()

    assert "`tweet_explore` does not need it" in text
    assert "`XQUIK_API_KEY` for `tweet_read`" in text
    assert "`HERMES_TWEET_ENABLE_ACTIONS=true`" in text
    assert "user approves the exact operation" in text
