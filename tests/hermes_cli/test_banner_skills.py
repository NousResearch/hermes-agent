"""Tests for banner get_available_skills() — disabled and platform filtering."""

from unittest.mock import patch

import pytest


_MOCK_SKILLS = [
    {"name": "skill-a", "description": "A skill", "category": "tools"},
    {"name": "skill-b", "description": "B skill", "category": "tools"},
    {"name": "skill-c", "description": "C skill", "category": "creative"},
]


def test_get_available_skills_delegates_to_find_all_skills():
    """get_available_skills should call _find_all_skills (which handles filtering)."""
    with patch("tools.skills_tool._find_all_skills", return_value=list(_MOCK_SKILLS)):
        from hermes_cli.banner import get_available_skills
        result = get_available_skills()

    assert "tools" in result
    assert "creative" in result
    assert sorted(result["tools"]) == ["skill-a", "skill-b"]
    assert result["creative"] == ["skill-c"]


def test_get_available_skills_excludes_disabled():
    """Disabled skills should not appear in the banner count."""
    # _find_all_skills already filters disabled skills, so if we give it
    # a filtered list, get_available_skills should reflect that.
    filtered = [s for s in _MOCK_SKILLS if s["name"] != "skill-b"]
    with patch("tools.skills_tool._find_all_skills", return_value=filtered):
        from hermes_cli.banner import get_available_skills
        result = get_available_skills()

    all_names = [n for names in result.values() for n in names]
    assert "skill-b" not in all_names
    assert "skill-a" in all_names
    assert len(all_names) == 2


def test_get_available_skills_empty_when_no_skills():
    """No skills installed returns empty dict."""
    with patch("tools.skills_tool._find_all_skills", return_value=[]):
        from hermes_cli.banner import get_available_skills
        result = get_available_skills()

    assert result == {}


def test_get_available_skills_handles_import_failure():
    """If _find_all_skills import fails, return empty dict gracefully."""
    with patch("tools.skills_tool._find_all_skills", side_effect=ImportError("boom")):
        from hermes_cli.banner import get_available_skills
        result = get_available_skills()

    assert result == {}


def test_get_available_skills_null_category_becomes_general():
    """Skills with None category should be grouped under 'general'."""
    skills = [{"name": "orphan-skill", "description": "No cat", "category": None}]
    with patch("tools.skills_tool._find_all_skills", return_value=skills):
        from hermes_cli.banner import get_available_skills
        result = get_available_skills()

    assert "general" in result
    assert result["general"] == ["orphan-skill"]
