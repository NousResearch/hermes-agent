"""E2E test: Curator lifecycle — create → stale → archive → restore."""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.knowledge_curator import DomainNoteCurator


@pytest.fixture
def curator_env(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    vault.mkdir()
    domains = vault / "domains"
    domains.mkdir()
    (domains / "frontend").mkdir()
    (domains / ".archive").mkdir()
    return vault


def test_e2e_curator_lifecycle(curator_env: Path) -> None:
    """Test the full curator lifecycle."""
    vault = curator_env

    # Step 1: Create domain note
    note_path = vault / "domains" / "frontend" / "react-pattern.md"
    note_path.write_text(
        "---\ntitle: React Pattern\norigin_project: proj-a\npromoted_at: 2026-01-01\nstatus: approved\ntags:\n  - frontend\n---\nContent\n"
    )
    assert note_path.exists()

    curator = DomainNoteCurator(vault_path=vault)

    # Step 2: Fake timestamp to be old (set mtime to 60 days ago)
    old_time = (datetime.now() - timedelta(days=60)).timestamp()
    import os
    os.utime(note_path, (old_time, old_time))

    # Step 3: Run curator → marks stale
    notes = curator.scan_domain_notes()
    assert len(notes) == 1
    stale = curator.mark_stale(notes, stale_after_days=30)
    assert len(stale) == 1
    assert stale[0]["stale"] is True

    # Step 4: Run curator again → archives
    archived_path = curator.archive_note(str(note_path))
    assert archived_path is not None
    assert Path(archived_path).exists()
    assert not note_path.exists()

    # Step 5: Restore → note back in domain KB
    restored_path = curator.restore_note(archived_path, target_domain="frontend")
    assert restored_path is not None
    assert Path(restored_path).exists()
    assert not Path(archived_path).exists()

    # Verify content is intact
    content = Path(restored_path).read_text(encoding="utf-8")
    assert "React Pattern" in content
    assert "origin_project: proj-a" in content
