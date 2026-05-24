"""Tests for agent.knowledge_preferences.KnowledgePreferenceManager."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.knowledge_preferences import KnowledgePreferenceManager


@pytest.fixture
def mock_hermes_home(tmp_path: Path) -> Path:
    """Create a temporary HERMES_HOME directory."""
    home = tmp_path / ".hermes"
    home.mkdir()
    return home


def test_save_preference(mock_hermes_home: Path) -> None:
    mgr = KnowledgePreferenceManager(hermes_home=mock_hermes_home)
    pref_id = mgr.save_preference("frontend", "proj-a", "react", True, "useful pattern")
    assert len(pref_id) == 8
    prefs = mgr.list_preferences()
    assert len(prefs) == 1
    assert prefs[0]["domain"] == "frontend"
    assert prefs[0]["allow"] is True


def test_check_preference_exact_match(mock_hermes_home: Path) -> None:
    mgr = KnowledgePreferenceManager(hermes_home=mock_hermes_home)
    mgr.save_preference("frontend", "proj-a", "react", True)
    result = mgr.check_preference("frontend", "proj-a", "This is about react hooks")
    assert result is not None
    assert result["allow"] is True


def test_check_preference_no_match(mock_hermes_home: Path) -> None:
    mgr = KnowledgePreferenceManager(hermes_home=mock_hermes_home)
    mgr.save_preference("frontend", "proj-a", "react", True)
    result = mgr.check_preference("backend", "proj-a", "database migration")
    assert result is None


def test_check_preference_domain_level(mock_hermes_home: Path) -> None:
    """Domain-level preference (project='*') should match any project."""
    mgr = KnowledgePreferenceManager(hermes_home=mock_hermes_home)
    mgr.save_preference("frontend", "*", "react", False, "not relevant")
    result = mgr.check_preference("frontend", "proj-b", "react pattern")
    assert result is not None
    assert result["allow"] is False


def test_list_preferences(mock_hermes_home: Path) -> None:
    mgr = KnowledgePreferenceManager(hermes_home=mock_hermes_home)
    mgr.save_preference("frontend", "proj-a", "react", True)
    mgr.save_preference("backend", "proj-b", "api", False)
    prefs = mgr.list_preferences()
    assert len(prefs) == 2


def test_delete_preference(mock_hermes_home: Path) -> None:
    mgr = KnowledgePreferenceManager(hermes_home=mock_hermes_home)
    pref_id = mgr.save_preference("frontend", "proj-a", "react", True)
    assert mgr.delete_preference(pref_id) is True
    assert len(mgr.list_preferences()) == 0


def test_delete_nonexistent_preference(mock_hermes_home: Path) -> None:
    mgr = KnowledgePreferenceManager(hermes_home=mock_hermes_home)
    assert mgr.delete_preference("nonexistent") is False


def test_persistence(mock_hermes_home: Path) -> None:
    """Preferences should persist across manager instances."""
    mgr1 = KnowledgePreferenceManager(hermes_home=mock_hermes_home)
    mgr1.save_preference("frontend", "proj-a", "react", True)

    mgr2 = KnowledgePreferenceManager(hermes_home=mock_hermes_home)
    prefs = mgr2.list_preferences()
    assert len(prefs) == 1
    assert prefs[0]["domain"] == "frontend"


def test_empty_preferences(mock_hermes_home: Path) -> None:
    mgr = KnowledgePreferenceManager(hermes_home=mock_hermes_home)
    assert mgr.list_preferences() == []
    assert mgr.check_preference("any", "any", "any content") is None


def test_corrupted_prefs_file(mock_hermes_home: Path) -> None:
    """Should handle corrupted JSON gracefully."""
    prefs_file = mock_hermes_home / "knowledge_preferences.json"
    prefs_file.write_text("not valid json", encoding="utf-8")
    mgr = KnowledgePreferenceManager(hermes_home=mock_hermes_home)
    assert mgr.list_preferences() == []
