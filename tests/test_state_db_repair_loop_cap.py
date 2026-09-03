"""#86747 regression: the state.db repair loop must be bounded across restarts
and must not accumulate identical multi-hundred-MB forensic backups.

Reported incident: b-tree page corruption (a class none of the repair
strategies can heal) failed `repair_state_db_schema` on every process start
for 11 days — 105 attempts, each taking a fresh ~900MB
`state.db.malformed-backup-*` copy of the SAME damaged bytes, 89GB total —
while `_claim_repair_attempt`'s in-memory set only bounded a single process.

Fix under test:
- persistent sidecar attempt ledger (`<db>.repair-attempts.json`): after
  `_MAX_PERSISTENT_REPAIR_ATTEMPTS` failed passes on the same file
  fingerprint, `repair_state_db_schema` refuses with a terminal, actionable
  error instead of re-running surgery;
- `_backup_db_file` dedupes against the newest existing backup (same
  size+mtime → reuse) and prunes to `_MAX_MALFORMED_BACKUPS` copies.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from unittest.mock import patch

import hermes_state
from hermes_state import (
    _MAX_MALFORMED_BACKUPS,
    _MAX_PERSISTENT_REPAIR_ATTEMPTS,
    _backup_db_file,
    _existing_malformed_backups,
    _persistent_repair_attempts_exhausted,
    _prune_malformed_backups,
    _record_repair_outcome,
    _repair_ledger_path,
    repair_state_db_schema,
)


def _make_unrepairable_db(tmp_path: Path) -> Path:
    """A file sqlite3 opens as garbage — no strategy can heal it."""
    db = tmp_path / "state.db"
    db.write_bytes(b"SQLite format 3\x00" + os.urandom(4096))
    return db


def _make_healthy_db(tmp_path: Path) -> Path:
    db = tmp_path / "state.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE t (x)")
    conn.execute("INSERT INTO t VALUES (1)")
    conn.commit()
    conn.close()
    return db


# ---------------------------------------------------------------------------
# Persistent attempt ledger
# ---------------------------------------------------------------------------


class TestPersistentAttemptCap:
    def test_failed_repairs_accumulate_in_ledger(self, tmp_path):
        db = _make_unrepairable_db(tmp_path)
        report = repair_state_db_schema(db)
        assert report["repaired"] is False
        ledger = json.loads(_repair_ledger_path(db).read_text())
        assert ledger["failed_attempts"] == 1

    def test_repair_refuses_after_cap_with_terminal_error(self, tmp_path):
        db = _make_unrepairable_db(tmp_path)
        for _ in range(_MAX_PERSISTENT_REPAIR_ATTEMPTS):
            report = repair_state_db_schema(db)
            assert report["repaired"] is False
        # Budget burned: the next call must refuse WITHOUT running surgery
        # (and without taking another backup).
        backups_before = len(_existing_malformed_backups(db))
        with patch.object(hermes_state, "_repair_state_db_schema_locked") as surgery:
            report = repair_state_db_schema(db)
        surgery.assert_not_called()
        assert report["repaired"] is False
        assert "Manual recovery required" in report["error"]
        assert ".recover" in report["error"]
        assert len(_existing_malformed_backups(db)) == backups_before

    def test_changed_file_resets_the_budget(self, tmp_path):
        db = _make_unrepairable_db(tmp_path)
        for _ in range(_MAX_PERSISTENT_REPAIR_ATTEMPTS):
            repair_state_db_schema(db)
        assert _persistent_repair_attempts_exhausted(db)
        # A restored/replaced file (different size+mtime) gets fresh attempts.
        db.write_bytes(b"SQLite format 3\x00" + os.urandom(8192))
        assert not _persistent_repair_attempts_exhausted(db)

    def test_successful_repair_clears_the_ledger(self, tmp_path):
        db = _make_healthy_db(tmp_path)
        _record_repair_outcome(db, repaired=False)
        assert _repair_ledger_path(db).exists()
        _record_repair_outcome(db, repaired=True)
        assert not _repair_ledger_path(db).exists()

    def test_corrupt_ledger_is_ignored_not_fatal(self, tmp_path):
        db = _make_unrepairable_db(tmp_path)
        _repair_ledger_path(db).write_text("{not json")
        assert not _persistent_repair_attempts_exhausted(db)
        # And a repair pass overwrites it cleanly.
        repair_state_db_schema(db)
        assert json.loads(_repair_ledger_path(db).read_text())["failed_attempts"] == 1


# ---------------------------------------------------------------------------
# Backup dedupe + retention cap
# ---------------------------------------------------------------------------


class TestBackupDedupeAndCap:
    def test_identical_file_is_not_backed_up_twice(self, tmp_path):
        db = _make_unrepairable_db(tmp_path)
        first, err = _backup_db_file(db)
        assert err is None and first is not None
        second, err = _backup_db_file(db)
        assert err is None
        # Same damaged bytes (size+mtime unchanged) → the existing backup is
        # reused instead of copying another ~900MB.
        assert second == first
        assert len(_existing_malformed_backups(db)) == 1

    def test_changed_file_gets_a_new_backup(self, tmp_path):
        db = _make_unrepairable_db(tmp_path)
        first, _ = _backup_db_file(db)
        db.write_bytes(b"SQLite format 3\x00" + os.urandom(2048))
        second, err = _backup_db_file(db)
        assert err is None
        assert second != first

    def test_retention_cap_prunes_oldest(self, tmp_path):
        db = _make_unrepairable_db(tmp_path)
        # Seed more than the cap with distinct fake timestamped backups.
        for i in range(_MAX_MALFORMED_BACKUPS + 3):
            fake = db.with_name(f"{db.name}.malformed-backup-20260801_00000{i}")
            fake.write_bytes(b"x")
            fake.with_name(fake.name + "-wal").write_bytes(b"w")
        _prune_malformed_backups(db)
        remaining = _existing_malformed_backups(db)
        assert len(remaining) == _MAX_MALFORMED_BACKUPS
        # Newest kept (sorted by timestamp suffix, descending).
        names = [p.name for p in remaining]
        assert names == sorted(names, reverse=True)
        # Sidecars of pruned backups are gone too.
        leftover_sidecars = [
            p for p in tmp_path.iterdir()
            if p.name.endswith("-wal")
            and p.with_name(p.name[:-4]) not in remaining
        ]
        assert leftover_sidecars == []

    def test_backup_via_repair_path_does_not_accumulate(self, tmp_path):
        """End-to-end: repeated failed repairs on the same file keep ONE backup."""
        db = _make_unrepairable_db(tmp_path)
        for _ in range(_MAX_PERSISTENT_REPAIR_ATTEMPTS):
            repair_state_db_schema(db)
        assert len(_existing_malformed_backups(db)) == 1
