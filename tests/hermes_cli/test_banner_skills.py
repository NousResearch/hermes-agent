"""Tests for banner get_available_skills() — disabled and platform filtering."""

from unittest.mock import patch


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


def test_get_available_skills_null_category_becomes_general():
    """Skills with None category should be grouped under 'general'."""
    skills = [{"name": "orphan-skill", "description": "No cat", "category": None}]
    with patch("tools.skills_tool._find_all_skills", return_value=skills):
        from hermes_cli.banner import get_available_skills
        result = get_available_skills()

    assert "general" in result
    assert result["general"] == ["orphan-skill"]
