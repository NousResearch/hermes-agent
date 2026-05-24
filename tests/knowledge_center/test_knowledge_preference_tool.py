"""Tests for tools.knowledge_preference_tool."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.knowledge_preference_tool import manage_knowledge_preference


@pytest.fixture
def mock_hermes_home(tmp_path: Path) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    return home


@pytest.fixture(autouse=True)
def patch_manager(mock_hermes_home: Path) -> None:
    with patch("tools.knowledge_preference_tool.KnowledgePreferenceManager") as mock_cls:
        from agent.knowledge_preferences import KnowledgePreferenceManager
        real_mgr = KnowledgePreferenceManager(hermes_home=mock_hermes_home)
        mock_cls.return_value = real_mgr
        yield


def test_save_preference() -> None:
    result_str = manage_knowledge_preference(
        action="save",
        domain="frontend",
        project="proj-a",
        pattern="react",
        allow=True,
        reason="useful",
    )
    result = json.loads(result_str)
    assert result["success"] is True
    assert result["action"] == "save"
    assert "pref_id" in result


def test_save_missing_fields() -> None:
    result_str = manage_knowledge_preference(action="save", domain="frontend")
    result = json.loads(result_str)
    assert result["success"] is False
    assert "error" in result


def test_list_preferences() -> None:
    # Save one first
    manage_knowledge_preference(
        action="save", domain="frontend", project="proj-a",
        pattern="react", allow=True,
    )
    result_str = manage_knowledge_preference(action="list")
    result = json.loads(result_str)
    assert result["success"] is True
    assert result["count"] >= 1


def test_delete_preference() -> None:
    # Save one
    save_result = json.loads(manage_knowledge_preference(
        action="save", domain="frontend", project="proj-a",
        pattern="react", allow=True,
    ))
    pref_id = save_result["pref_id"]

    # Delete it
    result_str = manage_knowledge_preference(action="delete", pref_id=pref_id)
    result = json.loads(result_str)
    assert result["success"] is True
    assert result["action"] == "delete"


def test_delete_missing_id() -> None:
    result_str = manage_knowledge_preference(action="delete")
    result = json.loads(result_str)
    assert result["success"] is False
    assert "error" in result


def test_delete_nonexistent() -> None:
    result_str = manage_knowledge_preference(action="delete", pref_id="nonexistent")
    result = json.loads(result_str)
    assert result["success"] is False
    assert "not found" in result["error"].lower()


def test_invalid_action() -> None:
    result_str = manage_knowledge_preference(action="invalid")
    result = json.loads(result_str)
    assert result["success"] is False
    assert "Invalid action" in result["error"]
