"""Tests for tools.knowledge_promote."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.knowledge_promote import (
    promote_knowledge,
    check_requirements,
    _slugify,
    _resolve_vault_path,
)


@pytest.fixture
def mock_vault(tmp_path: Path) -> Path:
    """Create a temporary vault with domain directories."""
    vault = tmp_path / "vault"
    vault.mkdir()
    domains_dir = vault / "domains"
    domains_dir.mkdir()
    (domains_dir / "frontend").mkdir()
    projects_dir = vault / "projects"
    projects_dir.mkdir()
    (projects_dir / "test-proj.md").write_text("---\ntitle: Test\nproject_slug: test-proj\n---\n# Test\n## Notes\n")
    return vault


def test_slugify() -> None:
    assert _slugify("React Hooks Pattern") == "react-hooks-pattern"
    assert _slugify("  My Note  ") == "my-note"
    assert _slugify("Special!@#$%Chars") == "special-chars"


def test_promote_knowledge_success(mock_vault: Path) -> None:
    with patch("tools.knowledge_promote._resolve_vault_path", return_value=mock_vault):
        result_str = promote_knowledge(
            title="React Hooks Pattern",
            content="Use useEffect for side effects...",
            source_project="test-proj",
            target_domain="frontend",
            summary="A useful pattern",
        )
        result = json.loads(result_str)
        assert result["success"] is True
        assert "note_path" in result
        assert result["domain"] == "frontend"
        assert result["source_project"] == "test-proj"

        # Verify file was created
        note_path = Path(result["note_path"])
        assert note_path.exists()
        content = note_path.read_text(encoding="utf-8")
        assert "React Hooks Pattern" in content
        assert "origin_project: test-proj" in content
        assert "A useful pattern" in content


def test_promote_knowledge_invalid_domain(mock_vault: Path) -> None:
    with patch("tools.knowledge_promote._resolve_vault_path", return_value=mock_vault):
        result_str = promote_knowledge(
            title="Test",
            content="Content",
            source_project="test-proj",
            target_domain="invalid-domain",
        )
        result = json.loads(result_str)
        assert result["success"] is False
        assert "error" in result


def test_promote_knowledge_duplicate_title(mock_vault: Path) -> None:
    with patch("tools.knowledge_promote._resolve_vault_path", return_value=mock_vault):
        # Create first note
        promote_knowledge("React Pattern", "Content 1", "test-proj", "frontend")
        # Create duplicate
        result_str = promote_knowledge("React Pattern", "Content 2", "test-proj", "frontend")
        result = json.loads(result_str)
        assert result["success"] is True
        # Should have a different path (with -1 suffix)
        note_path = Path(result["note_path"])
        assert note_path.exists()
        assert "-1" in note_path.name or "-2" in note_path.name


def test_promote_knowledge_creates_domain_dir(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    with patch("tools.knowledge_promote._resolve_vault_path", return_value=vault):
        result_str = promote_knowledge(
            title="New Domain Note",
            content="Content",
            source_project="test-proj",
            target_domain="backend",
        )
        result = json.loads(result_str)
        assert result["success"] is True
        assert (vault / "domains" / "backend").exists()


def test_check_requirements_missing() -> None:
    with patch.object(Path, "home", return_value=Path("/nonexistent")):
        with patch.dict(os.environ, {}, clear=True):
            assert check_requirements() is False
