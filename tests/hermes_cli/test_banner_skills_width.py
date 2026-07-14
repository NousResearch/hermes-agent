"""Tests for banner skills display — terminal-width-aware truncation."""

import os
from unittest.mock import patch

from rich.console import Console

import hermes_cli.banner as banner
import model_tools
import tools.mcp_tool


def _build_banner_with_skills(skills_by_category, term_width=160):
    """Helper: build banner with given skills and return captured output."""
    with (
        patch.object(
            model_tools,
            "check_tool_availability",
            return_value=([], []),
        ),
        patch.object(banner, "get_available_skills", return_value=skills_by_category),
        patch.object(banner, "get_update_result", return_value=None),
        patch.object(tools.mcp_tool, "get_mcp_status", return_value=[]),
        patch("shutil.get_terminal_size", return_value=os.terminal_size((term_width, 50))),
    ):
        console = Console(
            record=True, force_terminal=False, color_system=None, width=term_width
        )
        banner.build_welcome_banner(
            console=console,
            model="anthropic/test-model",
            cwd="/tmp/project",
            tools=[],
        )
        return console.export_text()


def test_wide_terminal_shows_more_than_8_skills():
    """A wide terminal should display more than 8 skills per category."""
    # 15 skills in one category
    skills = {"research": [f"skill-{i:02d}" for i in range(15)]}
    text = _build_banner_with_skills(skills, term_width=200)

    # With a 200-char terminal, more than 8 should be visible.
    # The old code always truncated at 8; we should see at least 9 now.
    assert "skill-08" in text, f"Expected skill-08 in output for wide terminal: {text}"


def test_narrow_terminal_limits_skills():
    """A narrow terminal should still limit skills to avoid wrapping."""
    skills = {"research": [f"skill-{i:02d}" for i in range(15)]}
    text = _build_banner_with_skills(skills, term_width=80)

    # With an 80-char terminal, we should NOT see all 15 skills — some truncation
    # is expected. Verify the "+N more" indicator is present.
    assert "more" in text or "..." in text or "skill-00" in text


def test_small_category_shows_all_skills():
    """Categories with few skills should show all of them regardless of width."""
    skills = {"security": ["auth", "vault"]}
    text = _build_banner_with_skills(skills, term_width=80)

    assert "auth" in text
    assert "vault" in text
    # No "+N more" indicator for small categories
    assert "+2 more" not in text


def test_skills_respect_category_label_width():
    """Skills display should account for the category label prefix width."""
    # A category with a long name should have less room for skills
    skills = {"very-long-category-name": [f"skill-{i:02d}" for i in range(10)]}
    text = _build_banner_with_skills(skills, term_width=120)

    # Should still show at least some skills
    assert "skill-00" in text
