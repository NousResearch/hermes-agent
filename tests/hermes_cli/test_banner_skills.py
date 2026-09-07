"""Tests for banner get_available_skills() — disabled and platform filtering."""

import pytest
from unittest.mock import patch


_MOCK_SKILLS = [
    {"name": "skill-a", "description": "A skill", "category": "tools"},
    {"name": "skill-b", "description": "B skill", "category": "tools"},
    {"name": "skill-c", "description": "C skill", "category": "creative"},
]


@pytest.fixture(autouse=True)
def _reset_skills_cache():
    """get_available_skills is memoized per-process (startup perf) — reset
    the cache around each test so patched _find_all_skills results are
    actually observed."""
    import hermes_cli.banner as banner
    banner._available_skills_cache = None
    yield
    banner._available_skills_cache = None


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


def test_get_available_skills_is_memoized():
    """Second call must not re-walk the skills tree (startup perf contract)."""
    import hermes_cli.banner as banner
    calls = []

    def fake_find(**kwargs):
        calls.append(1)
        return list(_MOCK_SKILLS)

    with patch("tools.skills_tool._find_all_skills", side_effect=fake_find):
        first = banner.get_available_skills()
        second = banner.get_available_skills()

    assert first == second
    assert len(calls) == 1
