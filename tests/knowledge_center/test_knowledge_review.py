"""Tests for tools.knowledge_review."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from tools.knowledge_review import (
    review_knowledge,
    check_requirements,
    _get_queue_path,
    _load_queue,
    _save_queue,
)


@pytest.fixture
def mock_vault(tmp_path: Path) -> Path:
    """Create a temporary vault with domains directory."""
    vault = tmp_path / "vault"
    vault.mkdir()
    domains_dir = vault / "domains"
    domains_dir.mkdir()
    return vault


@pytest.fixture(autouse=True)
def patch_vault(mock_vault: Path) -> None:
    """Patch vault path for all tests."""
    with patch("tools.knowledge_review._get_queue_path", return_value=mock_vault / "domains" / ".review_queue.json"):
        with patch("tools.knowledge_promote._resolve_vault_path", return_value=mock_vault):
            yield


def test_list_empty_queue(mock_vault: Path) -> None:
    result_str = review_knowledge(action="list")
    result = json.loads(result_str)
    assert result["success"] is True
    assert result["count"] == 0
    assert result["items"] == []


def test_add_to_queue(mock_vault: Path) -> None:
    result_str = review_knowledge(
        action="add",
        title="Test Knowledge",
        content="Some content here",
        source_project="proj-a",
        target_domain="frontend",
    )
    result = json.loads(result_str)
    assert result["success"] is True
    assert result["action"] == "add"
    assert "knowledge_id" in result
    assert result["status"] == "pending"

    # Verify it's in the queue
    result_str = review_knowledge(action="list")
    result = json.loads(result_str)
    assert result["count"] == 1
    assert result["items"][0]["title"] == "Test Knowledge"


def test_add_missing_fields(mock_vault: Path) -> None:
    result_str = review_knowledge(action="add", title="Test")
    result = json.loads(result_str)
    assert result["success"] is False
    assert "error" in result


def test_approve_knowledge(mock_vault: Path) -> None:
    # Add to queue
    result_str = review_knowledge(
        action="add",
        title="React Pattern",
        content="Use useEffect for side effects",
        source_project="proj-a",
        target_domain="frontend",
        summary="Test summary",
    )
    add_result = json.loads(result_str)
    knowledge_id = add_result["knowledge_id"]

    # Approve
    result_str = review_knowledge(action="approve", knowledge_id=knowledge_id)
    result = json.loads(result_str)
    assert result["success"] is True
    assert result["action"] == "approve"
    assert "promote_result" in result

    # Verify status changed
    result_str = review_knowledge(action="list")
    result = json.loads(result_str)
    assert result["items"][0]["status"] == "approved"


def test_reject_knowledge(mock_vault: Path) -> None:
    # Add to queue
    result_str = review_knowledge(
        action="add",
        title="Bad Pattern",
        content="Some bad content",
        source_project="proj-a",
        target_domain="frontend",
    )
    add_result = json.loads(result_str)
    knowledge_id = add_result["knowledge_id"]

    # Reject with reason
    result_str = review_knowledge(
        action="reject",
        knowledge_id=knowledge_id,
        reason="Not relevant",
    )
    result = json.loads(result_str)
    assert result["success"] is True
    assert result["action"] == "reject"


def test_defer_knowledge(mock_vault: Path) -> None:
    # Add to queue
    result_str = review_knowledge(
        action="add",
        title="Maybe Later",
        content="Some content",
        source_project="proj-a",
        target_domain="frontend",
    )
    add_result = json.loads(result_str)
    knowledge_id = add_result["knowledge_id"]

    # Defer
    result_str = review_knowledge(action="defer", knowledge_id=knowledge_id)
    result = json.loads(result_str)
    assert result["success"] is True
    assert result["action"] == "defer"


def test_delete_knowledge(mock_vault: Path) -> None:
    # Add to queue
    result_str = review_knowledge(
        action="add",
        title="Delete Me",
        content="Content",
        source_project="proj-a",
        target_domain="frontend",
    )
    add_result = json.loads(result_str)
    knowledge_id = add_result["knowledge_id"]

    # Delete
    result_str = review_knowledge(action="delete", knowledge_id=knowledge_id)
    result = json.loads(result_str)
    assert result["success"] is True

    # Verify removed
    result_str = review_knowledge(action="list")
    result = json.loads(result_str)
    assert result["count"] == 0


def test_approve_missing_id(mock_vault: Path) -> None:
    result_str = review_knowledge(action="approve")
    result = json.loads(result_str)
    assert result["success"] is False
    assert "error" in result


def test_approve_nonexistent_id(mock_vault: Path) -> None:
    result_str = review_knowledge(action="approve", knowledge_id="nonexistent")
    result = json.loads(result_str)
    assert result["success"] is False
    assert "not found" in result["error"].lower()


def test_invalid_action(mock_vault: Path) -> None:
    result_str = review_knowledge(action="invalid")
    result = json.loads(result_str)
    assert result["success"] is False
    assert "Invalid action" in result["error"]


def test_check_requirements_missing() -> None:
    with patch.object(Path, "home", return_value=Path("/nonexistent")):
        with patch.dict(os.environ, {}, clear=True):
            assert check_requirements() is False
