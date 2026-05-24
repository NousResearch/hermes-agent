"""Tests for tools.knowledge_review — Card Store / ReviewInbox backend."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from tools.knowledge_review import (
    review_knowledge,
    check_requirements,
    _get_vault_path,
    _migrate_old_queue,
)


@pytest.fixture
def mock_vault(tmp_path: Path) -> Path:
    """Create a temporary vault with required directories."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "domains").mkdir()
    (vault / "review-queue").mkdir()
    (vault / "sources").mkdir()
    (vault / "knowledge").mkdir()
    (vault / "lessons").mkdir()
    (vault / "patterns").mkdir()
    (vault / "playbooks").mkdir()
    (vault / "skills").mkdir()
    return vault


@pytest.fixture(autouse=True)
def patch_vault(mock_vault: Path) -> None:
    """Patch vault path for all tests."""
    with patch("tools.knowledge_review._get_vault_path", return_value=mock_vault):
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


def test_invalid_action(mock_vault: Path) -> None:
    result_str = review_knowledge(action="invalid")
    result = json.loads(result_str)
    assert result["success"] is False
    assert "Invalid action" in result["error"]


def test_check_requirements_missing() -> None:
    # Must bypass autouse fixture that patches _get_vault_path
    with patch("tools.knowledge_review._get_vault_path", return_value=Path("/nonexistent/path/nowhere")):
        assert check_requirements() is False
