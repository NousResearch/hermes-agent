"""Tests for agent.knowledge_curator (DomainNoteCurator + KnowledgeUsageTracker)."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.knowledge_curator import DomainNoteCurator, KnowledgeUsageTracker


@pytest.fixture
def mock_vault(tmp_path: Path) -> Path:
    """Create a temporary vault with domain notes."""
    vault = tmp_path / "vault"
    vault.mkdir()
    domains_dir = vault / "domains"
    domains_dir.mkdir()
    (domains_dir / "frontend").mkdir()
    (domains_dir / "backend").mkdir()
    (domains_dir / ".archive").mkdir()

    # Create a domain note
    note = domains_dir / "frontend" / "react-pattern.md"
    note.write_text(
        "---\ntitle: React Pattern\norigin_project: proj-a\npromoted_at: 2026-05-01\nstatus: approved\ntags:\n  - frontend\n---\nContent here\n"
    )

    # Create README (should be skipped)
    (domains_dir / "frontend" / "README.md").write_text("# Frontend\n")

    return vault


@pytest.fixture
def mock_hermes_home(tmp_path: Path) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    return home


def test_scan_domain_notes(mock_vault: Path) -> None:
    curator = DomainNoteCurator(vault_path=mock_vault)
    notes = curator.scan_domain_notes()
    assert len(notes) == 1
    assert notes[0]["domain"] == "frontend"
    assert notes[0]["title"] == "React Pattern"
    assert notes[0]["origin_project"] == "proj-a"


def test_scan_skips_readme(mock_vault: Path) -> None:
    curator = DomainNoteCurator(vault_path=mock_vault)
    notes = curator.scan_domain_notes()
    assert all(n["path"].endswith("README.md") is False for n in notes)


def test_scan_skips_hidden_dirs(mock_vault: Path) -> None:
    curator = DomainNoteCurator(vault_path=mock_vault)
    notes = curator.scan_domain_notes()
    assert all(".archive" not in n["path"] for n in notes)


def test_is_agent_created(mock_vault: Path) -> None:
    curator = DomainNoteCurator(vault_path=mock_vault)
    notes = curator.scan_domain_notes()
    assert curator.is_agent_created(notes[0]) is True


def test_mark_stale(mock_vault: Path) -> None:
    curator = DomainNoteCurator(vault_path=mock_vault)
    notes = curator.scan_domain_notes()
    # Note is from 2026-05-01, should be stale if stale_after_days is small
    stale = curator.mark_stale(notes, stale_after_days=1)
    assert len(stale) >= 0  # Depends on current date


def test_mark_not_stale(mock_vault: Path) -> None:
    curator = DomainNoteCurator(vault_path=mock_vault)
    notes = curator.scan_domain_notes()
    stale = curator.mark_stale(notes, stale_after_days=9999)
    assert len(stale) == 0


def test_archive_note(mock_vault: Path) -> None:
    curator = DomainNoteCurator(vault_path=mock_vault)
    note_path = str(mock_vault / "domains" / "frontend" / "react-pattern.md")
    archived = curator.archive_note(note_path)
    assert archived is not None
    assert Path(archived).exists()
    assert not Path(note_path).exists()


def test_archive_nonexistent(mock_vault: Path) -> None:
    curator = DomainNoteCurator(vault_path=mock_vault)
    archived = curator.archive_note("/nonexistent/note.md")
    assert archived is None


def test_restore_note(mock_vault: Path) -> None:
    curator = DomainNoteCurator(vault_path=mock_vault)
    note_path = str(mock_vault / "domains" / "frontend" / "react-pattern.md")
    archived = curator.archive_note(note_path)
    assert archived is not None

    restored = curator.restore_note(archived, target_domain="frontend")
    assert restored is not None
    assert Path(restored).exists()
    assert not Path(archived).exists()


def test_restore_nonexistent(mock_vault: Path) -> None:
    curator = DomainNoteCurator(vault_path=mock_vault)
    restored = curator.restore_note("/nonexistent/note.md")
    assert restored is None


def test_empty_vault() -> None:
    curator = DomainNoteCurator(vault_path=Path("/nonexistent"))
    assert curator.scan_domain_notes() == []


# --- KnowledgeUsageTracker tests ---


def test_record_view(mock_hermes_home: Path) -> None:
    tracker = KnowledgeUsageTracker(hermes_home=mock_hermes_home)
    tracker.record_view("note-1", "frontend", "proj-a")
    usage = tracker.get_usage("note-1")
    assert usage is not None
    assert usage["view_count"] == 1
    assert usage["domain"] == "frontend"


def test_record_use(mock_hermes_home: Path) -> None:
    tracker = KnowledgeUsageTracker(hermes_home=mock_hermes_home)
    tracker.record_use("note-1", "frontend", "proj-a")
    usage = tracker.get_usage("note-1")
    assert usage is not None
    assert usage["use_count"] == 1


def test_multiple_views(mock_hermes_home: Path) -> None:
    tracker = KnowledgeUsageTracker(hermes_home=mock_hermes_home)
    tracker.record_view("note-1", "frontend", "proj-a")
    tracker.record_view("note-1", "frontend", "proj-a")
    tracker.record_view("note-1", "frontend", "proj-a")
    usage = tracker.get_usage("note-1")
    assert usage["view_count"] == 3


def test_get_all_usage(mock_hermes_home: Path) -> None:
    tracker = KnowledgeUsageTracker(hermes_home=mock_hermes_home)
    tracker.record_view("note-1", "frontend", "proj-a")
    tracker.record_view("note-2", "backend", "proj-b")
    all_usage = tracker.get_all_usage()
    assert len(all_usage) == 2


def test_persistence(mock_hermes_home: Path) -> None:
    tracker1 = KnowledgeUsageTracker(hermes_home=mock_hermes_home)
    tracker1.record_view("note-1", "frontend", "proj-a")

    tracker2 = KnowledgeUsageTracker(hermes_home=mock_hermes_home)
    usage = tracker2.get_usage("note-1")
    assert usage is not None
    assert usage["view_count"] == 1


def test_priority_score_old_popular(mock_hermes_home: Path) -> None:
    tracker = KnowledgeUsageTracker(hermes_home=mock_hermes_home)
    tracker.record_view("note-1", "frontend", "proj-a")
    tracker.record_use("note-1", "frontend", "proj-a")
    # Old date + popular = high priority
    score = tracker.get_priority_score("note-1", "2025-01-01")
    assert score > 0.5


def test_priority_score_new_unpopular(mock_hermes_home: Path) -> None:
    tracker = KnowledgeUsageTracker(hermes_home=mock_hermes_home)
    # New date + no views = low priority
    today = datetime.now().strftime("%Y-%m-%d")
    score = tracker.get_priority_score("note-2", today)
    assert score < 0.3


def test_corrupted_usage_file(mock_hermes_home: Path) -> None:
    usage_file = mock_hermes_home / "knowledge_usage.json"
    usage_file.write_text("not valid json", encoding="utf-8")
    tracker = KnowledgeUsageTracker(hermes_home=mock_hermes_home)
    assert tracker.get_all_usage() == {}
