"""#91229: emergency state.db backups must be reclaimed on failure paths too.

``preflight_state_db`` used to run its retention sweep only after a successful
snapshot, so every reason the snapshot never happened — a locked source database
(the ordinary state during a failed Windows self-update), a full disk, a missing
``state.db`` — also skipped the cleanup, and full-size backups accumulated
without bound precisely when reclaiming mattered most. These tests pin the
sweep to run unconditionally, before the snapshot attempt.
"""

import sqlite3
from pathlib import Path

import pytest

from hermes_cli.backup_sqlite import (
    EMERGENCY_BACKUP_PREFIX,
    EMERGENCY_BACKUP_RETENTION,
    preflight_state_db,
)

# Timestamps are fixed-width, so lexicographic order == chronological order.
STAMPS = ["2026-08-15T00-00-00-000000Z", "2026-08-17T00-00-00-000000Z",
          "2026-08-19T00-00-00-000000Z", "2026-08-21T00-00-00-000000Z",
          "2026-08-23T00-00-00-000000Z"]


def _write_old_backups(home: Path, count: int) -> list[Path]:
    made = []
    for stamp in STAMPS[:count]:
        bak = home / f"{EMERGENCY_BACKUP_PREFIX}{stamp}-1234.bak"
        bak.write_bytes(b"old snapshot")
        made.append(bak)
    return made


def _write_state_db(home: Path) -> None:
    conn = sqlite3.connect(home / "state.db")
    conn.execute("CREATE TABLE t (x)")
    conn.commit()
    conn.close()


def _backups(home: Path) -> list[str]:
    return sorted(p.name for p in home.glob(f"{EMERGENCY_BACKUP_PREFIX}*.bak"))


def test_locked_source_still_prunes(tmp_path, monkeypatch):
    """The #91229 shape: the snapshot fails (locked db / full disk), the sweep runs anyway."""
    _write_state_db(tmp_path)
    _write_old_backups(tmp_path, len(STAMPS))
    monkeypatch.setattr("hermes_cli.backup_sqlite._safe_copy_db", lambda *a, **k: False)
    with pytest.raises(RuntimeError):
        preflight_state_db(tmp_path)
    # The oldest backups past retention are gone even though no new one was written.
    remaining = _backups(tmp_path)
    assert remaining == [f"{EMERGENCY_BACKUP_PREFIX}{s}-1234.bak" for s in STAMPS[-EMERGENCY_BACKUP_RETENTION:]]


def test_missing_state_db_still_prunes(tmp_path):
    """A fresh install (no state.db) must not keep stale backups from a previous one."""
    _write_old_backups(tmp_path, len(STAMPS))
    result = preflight_state_db(tmp_path)
    assert result["path"] is None
    assert len(_backups(tmp_path)) == EMERGENCY_BACKUP_RETENTION


def test_successful_snapshot_keeps_retention(tmp_path):
    """A healthy path stays capped: two fresh snapshots on top of old backups keep only the newest."""
    _write_state_db(tmp_path)
    _write_old_backups(tmp_path, len(STAMPS))
    preflight_state_db(tmp_path)
    preflight_state_db(tmp_path)
    assert len(_backups(tmp_path)) == EMERGENCY_BACKUP_RETENTION


def test_win_unpacked_bak_is_never_touched(tmp_path):
    """`win-unpacked.bak` is a rollback copy maintained by the desktop pack step, not residue."""
    _write_state_db(tmp_path)
    _write_old_backups(tmp_path, len(STAMPS))
    rollback = tmp_path / "win-unpacked.bak"
    rollback.write_bytes(b"rollback copy")
    preflight_state_db(tmp_path)
    assert rollback.exists()


def test_retention_covers_partial_and_foreign_files(tmp_path):
    """Only this module's exact prefix+suffix shape is pruned; neighbours survive."""
    _write_state_db(tmp_path)
    _write_old_backups(tmp_path, len(STAMPS))
    partial = tmp_path / f"{EMERGENCY_BACKUP_PREFIX}2026-01-01T00-00-00-000000Z.partial"
    partial.write_bytes(b"interrupted staging file")
    user_file = tmp_path / "notes.bak"
    user_file.write_bytes(b"user file")
    preflight_state_db(tmp_path)
    assert partial.exists()
    assert user_file.exists()
