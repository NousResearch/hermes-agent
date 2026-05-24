"""Tests for agent.knowledge_relevance.KnowledgeRelevanceEngine."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.knowledge_relevance import KnowledgeRelevanceEngine


@pytest.fixture
def mock_vault(tmp_path: Path) -> Path:
    """Create a temporary vault with project notes."""
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()

    # Create project notes with different stacks and domains
    for slug, stack, domains in [
        ("proj-a", "node/next", ["frontend", "backend"]),
        ("proj-b", "node/vite", ["frontend", "backend"]),
        ("proj-c", "python", ["backend", "data"]),
        ("proj-d", "docker/mixed", ["devops", "backend"]),
    ]:
        note = projects_dir / f"{slug}.md"
        note.write_text(
            f"---\ntitle: {slug}\nproject_slug: {slug}\n---\n"
            f"| Stack | `{stack}` |\n"
            f"\ndomain: [{', '.join(domains)}]\n"
        )

    return tmp_path


def test_is_cross_project_relevant(mock_vault: Path) -> None:
    engine = KnowledgeRelevanceEngine(vault_path=mock_vault)
    content = "React component pattern with hooks for state management"
    assert engine.is_cross_project_relevant(content, "proj-a") is True


def test_is_cross_project_not_relevant(mock_vault: Path) -> None:
    """Random content should have low relevance score."""
    engine = KnowledgeRelevanceEngine(vault_path=mock_vault)
    content = "zqjx zqjx zqjx"
    score = engine.get_relevance_score(content, "proj-c")
    assert score < 0.3


def test_find_matching_projects(mock_vault: Path) -> None:
    engine = KnowledgeRelevanceEngine(vault_path=mock_vault)
    content = "React component pattern with hooks for state management"
    matches = engine.find_matching_projects(content, "proj-a")
    assert "proj-b" in matches


def test_find_matching_projects_random_content(mock_vault: Path) -> None:
    """Random content still matches same-domain projects (domain overlap drives matching)."""
    engine = KnowledgeRelevanceEngine(vault_path=mock_vault)
    content = "zqjx zqjx zqjx"
    matches = engine.find_matching_projects(content, "proj-a")
    assert "proj-b" in matches
    assert isinstance(matches, list)


def test_get_relevance_score_high(mock_vault: Path) -> None:
    engine = KnowledgeRelevanceEngine(vault_path=mock_vault)
    content = "React component pattern with hooks"
    score = engine.get_relevance_score(content, "proj-b")
    assert score > 0.0


def test_get_relevance_score_zero(mock_vault: Path) -> None:
    engine = KnowledgeRelevanceEngine(vault_path=mock_vault)
    content = "zqjx zqjx zqjx"
    score = engine.get_relevance_score(content, "proj-a")
    assert score == 0.0


def test_empty_vault() -> None:
    engine = KnowledgeRelevanceEngine(vault_path=Path("/nonexistent"))
    assert engine.is_cross_project_relevant("test content", "any-project") is False
    assert engine.find_matching_projects("test content", "any-project") == []
    assert engine.get_relevance_score("test content", "any-project") == 0.0
