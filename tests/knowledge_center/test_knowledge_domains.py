"""Tests for agent.knowledge_domains.DomainRelevanceMatcher."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.knowledge_domains import DomainRelevanceMatcher


@pytest.fixture
def mock_vault(tmp_path: Path) -> Path:
    """Create a temporary vault with project notes and domain dirs."""
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    domains_dir = tmp_path / "domains"
    domains_dir.mkdir()
    for d in ["frontend", "backend", "devops", "security", "testing", "data", "mobile", "infrastructure"]:
        (domains_dir / d).mkdir(parents=True, exist_ok=True)

    # Create a project note with domain frontmatter
    note = projects_dir / "test-project.md"
    note.write_text(
        "---\ntitle: Test Project\nproject_slug: test-project\nrole: devex\nrisk: high\ndomain: [frontend, backend]\n---\n# Test\n"
    )

    # Create a domain note
    domain_note = domains_dir / "frontend" / "react-pattern.md"
    domain_note.write_text(
        "---\ntitle: React Pattern\norigin_project: proj-a\n---\nReact hooks pattern\n"
    )

    return tmp_path


def test_classify_known_project(mock_vault: Path) -> None:
    m = DomainRelevanceMatcher(vault_path=mock_vault)
    assert m.classify("test-project") == ["frontend", "backend"]


def test_classify_unknown_project(mock_vault: Path) -> None:
    m = DomainRelevanceMatcher(vault_path=mock_vault)
    assert m.classify("nonexistent-project") == []


def test_match_knowledge_frontend(mock_vault: Path) -> None:
    m = DomainRelevanceMatcher(vault_path=mock_vault)
    content = "Use React hooks with useState and useEffect for component state management"
    domains = m.match_knowledge(content)
    assert "frontend" in domains


def test_match_knowledge_backend(mock_vault: Path) -> None:
    m = DomainRelevanceMatcher(vault_path=mock_vault)
    content = "API endpoint with postgres database migration and authentication middleware"
    domains = m.match_knowledge(content)
    assert "backend" in domains


def test_match_knowledge_devops(mock_vault: Path) -> None:
    m = DomainRelevanceMatcher(vault_path=mock_vault)
    content = "Docker compose deployment with CI/CD pipeline and monitoring"
    domains = m.match_knowledge(content)
    assert "devops" in domains


def test_match_knowledge_empty(mock_vault: Path) -> None:
    m = DomainRelevanceMatcher(vault_path=mock_vault)
    content = "zqjx zqjx zqjx zqjx zqjx"
    domains = m.match_knowledge(content)
    assert domains == []


def test_relevance_score_high(mock_vault: Path) -> None:
    m = DomainRelevanceMatcher(vault_path=mock_vault)
    content = "React component with hooks and state management"
    score = m.get_relevance_score(content, "test-project")
    assert score > 0.0


def test_relevance_score_zero(mock_vault: Path) -> None:
    m = DomainRelevanceMatcher(vault_path=mock_vault)
    content = "zqjx zqjx zqjx zqjx zqjx"
    score = m.get_relevance_score(content, "test-project")
    assert score == 0.0


def test_relevance_score_unknown_project(mock_vault: Path) -> None:
    m = DomainRelevanceMatcher(vault_path=mock_vault)
    content = "React component"
    score = m.get_relevance_score(content, "unknown-project")
    assert score == 0.0


def test_get_domain_notes(mock_vault: Path) -> None:
    m = DomainRelevanceMatcher(vault_path=mock_vault)
    notes = m.get_domain_notes(["frontend"])
    assert len(notes) == 1
    assert notes[0].name == "react-pattern.md"


def test_get_domain_notes_empty(mock_vault: Path) -> None:
    m = DomainRelevanceMatcher(vault_path=mock_vault)
    notes = m.get_domain_notes(["nonexistent-domain"])
    assert notes == []


def test_get_domain_notes_missing_vault() -> None:
    m = DomainRelevanceMatcher(vault_path=Path("/nonexistent/vault"))
    notes = m.get_domain_notes(["frontend"])
    assert notes == []


def test_classify_missing_vault() -> None:
    m = DomainRelevanceMatcher(vault_path=Path("/nonexistent/vault"))
    assert m.classify("any-project") == []
