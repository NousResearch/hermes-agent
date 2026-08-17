"""Regression: the state.db repair-loop guards must survive an mtime change.

Incident (2026-08-17): a malformed-SCHEMA state.db sent Hermes into an
unbounded repair loop that wrote a fresh 98MB forensic copy every ~10s —
2.3GB in 20 minutes, disk heading to zero, whole agent fleet at risk.

The #86747 guards were already present and did NOT hold, because both keyed
on ``size:mtime_ns``:

* ``_db_fingerprint`` -> the ledger's attempt counter reset to 1 on every
  pass, so ``_MAX_PERSISTENT_REPAIR_ATTEMPTS`` was never reached;
* ``_backup_db_file``'s dedupe compared mtime, so it never matched and each
  pass wrote another full-size copy.

Unlike the b-tree damage of #86747, the malformed-SCHEMA class still opens
and accepts writes (only ``sqlite_master`` is unreadable), so live writers,
WAL checkpoints and the in-place repair strategies all move mtime between
passes. These tests pin the guards to content, not mtime, and add the
missing free-space refusal.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from unittest.mock import patch

import hermes_state
from hermes_state import (
    _MAX_MALFORMED_BACKUPS,
    _MAX_PERSISTENT_REPAIR_ATTEMPTS,
    _REPAIR_BACKUP_MIN_FREE_BYTES,
    _backup_db_file,
    _db_fingerprint,
    _existing_malformed_backups,
    _persistent_repair_attempts_exhausted,
    _record_repair_outcome,
    _repair_backup_headroom_bytes,
)


def _damaged_db(tmp_path: Path, size: int = 200_000) -> Path:
    db = tmp_path / "state.db"
    db.write_bytes(b"SQLite format 3\x00" + os.urandom(size))
    return db


# ---------------------------------------------------------------------------
# Fingerprint stability
# ---------------------------------------------------------------------------


def test_fingerprint_survives_mtime_change(tmp_path):
    """A touched-but-unchanged file keeps its identity (the incident's core)."""
    db = _damaged_db(tmp_path)
    before = _db_fingerprint(db)
    time.sleep(0.01)
    os.utime(db, None)  # live writer / WAL checkpoint / in-place repair pass
    assert _db_fingerprint(db) == before


def test_fingerprint_changes_when_contents_change(tmp_path):
    """Genuine recovery must still reset the attempt budget."""
    db = _damaged_db(tmp_path)
    before = _db_fingerprint(db)
    db.write_bytes(b"SQLite format 3\x00" + os.urandom(200_000))
    assert _db_fingerprint(db) != before


def test_fingerprint_changes_on_truncation(tmp_path):
    db = _damaged_db(tmp_path)
    before = _db_fingerprint(db)
    with open(db, "r+b") as fh:
        fh.truncate(1024)
    assert _db_fingerprint(db) != before


# ---------------------------------------------------------------------------
# Attempt ledger
# ---------------------------------------------------------------------------


def test_attempt_budget_exhausts_despite_mtime_churn(tmp_path):
    """The loop must terminate even when every pass touches the file."""
    db = _damaged_db(tmp_path)
    for _ in range(_MAX_PERSISTENT_REPAIR_ATTEMPTS):
        assert not _persistent_repair_attempts_exhausted(db)
        _record_repair_outcome(db, repaired=False)
        time.sleep(0.01)
        os.utime(db, None)
    assert _persistent_repair_attempts_exhausted(db)


def test_successful_repair_clears_budget(tmp_path):
    db = _damaged_db(tmp_path)
    for _ in range(_MAX_PERSISTENT_REPAIR_ATTEMPTS):
        _record_repair_outcome(db, repaired=False)
    assert _persistent_repair_attempts_exhausted(db)
    _record_repair_outcome(db, repaired=True)
    assert not _persistent_repair_attempts_exhausted(db)


# ---------------------------------------------------------------------------
# Backup dedupe
# ---------------------------------------------------------------------------


def test_backup_dedupes_across_mtime_change(tmp_path):
    """Repeated passes over identical bytes must not each write a new copy."""
    db = _damaged_db(tmp_path)
    first, err = _backup_db_file(db)
    assert err is None and first is not None
    for _ in range(5):
        time.sleep(0.01)
        os.utime(db, None)
        again, err = _backup_db_file(db)
        assert err is None
        assert again == first, "a touched-but-identical DB was copied again"
    assert len(_existing_malformed_backups(db)) == 1


def test_backup_retention_cap_still_holds(tmp_path):
    """Genuinely different damaged states are kept, but bounded."""
    db = _damaged_db(tmp_path)
    for _ in range(_MAX_MALFORMED_BACKUPS + 3):
        db.write_bytes(b"SQLite format 3\x00" + os.urandom(200_000))
        _backup_db_file(db)
    assert len(_existing_malformed_backups(db)) <= _MAX_MALFORMED_BACKUPS


# ---------------------------------------------------------------------------
# Free-space guard
# ---------------------------------------------------------------------------


def test_backup_refused_when_disk_would_be_exhausted(tmp_path):
    """A nearly-full volume must not be finished off by the forensic copy."""
    db = _damaged_db(tmp_path)
    tight = type(
        "Usage",
        (),
        {"total": 10_000_000_000, "used": 0, "free": _REPAIR_BACKUP_MIN_FREE_BYTES // 2},
    )()
    with patch("shutil.disk_usage", return_value=tight):
        path, reason = _backup_db_file(db)
    assert path is None
    assert reason is not None and "free" in reason.lower()
    assert not _existing_malformed_backups(db)


def test_backup_allowed_on_small_volume_with_room(tmp_path):
    """A flat multi-GB floor would disable repair on small VMs/containers.

    50MB DB on a 10GB volume with 1.5GB free fits with ~30x headroom; the
    guard must allow it rather than hard-stopping repair forever.
    """
    db = _damaged_db(tmp_path, size=50_000_000)
    small_vm = type(
        "Usage", (), {"total": 10_000_000_000, "used": 8_500_000_000, "free": 1_500_000_000}
    )()
    with patch("shutil.disk_usage", return_value=small_vm):
        path, reason = _backup_db_file(db)
    assert reason is None and path is not None


def test_headroom_scales_with_volume_size():
    """Big volumes reserve proportionally; small ones keep a modest floor."""
    assert _repair_backup_headroom_bytes(1_000_000_000) == _REPAIR_BACKUP_MIN_FREE_BYTES
    assert _repair_backup_headroom_bytes(1_000_000_000_000) > _REPAIR_BACKUP_MIN_FREE_BYTES


def test_disk_guard_accounts_for_sidecars(tmp_path):
    """The copy includes -wal/-shm, so the space check must count them."""
    db = _damaged_db(tmp_path, size=1_000_000)
    db.with_name(db.name + "-wal").write_bytes(os.urandom(400_000_000))
    usage = type(
        "Usage",
        (),
        {"total": 10_000_000_000, "used": 0, "free": _REPAIR_BACKUP_MIN_FREE_BYTES + 300_000_000},
    )()
    with patch("shutil.disk_usage", return_value=usage):
        path, reason = _backup_db_file(db)
    assert path is None, "sidecar bytes were ignored by the free-space check"
    assert reason is not None


def test_failed_copy_leaves_no_countable_debris(tmp_path):
    """Prune only runs on success, so a failed copy must self-clean.

    Otherwise partials matching the backup prefix accumulate unbounded and,
    on a later successful pass, are KEPT (newest by name) while intact
    forensic copies get pruned away.
    """
    db = _damaged_db(tmp_path, size=1_000_000)
    db.with_name(db.name + "-wal").write_bytes(os.urandom(1_000_000))
    roomy = type(
        "Usage", (), {"total": 500_000_000_000, "used": 0, "free": 400_000_000_000}
    )()
    real_copy2 = shutil.copy2

    def sidecar_fails(src, dst, *a, **kw):
        if str(src).endswith("-wal"):
            Path(dst).write_bytes(b"PARTIAL" * 100)
            raise OSError(28, "No space left on device")
        return real_copy2(src, dst, *a, **kw)

    with patch("shutil.disk_usage", return_value=roomy), \
            patch("shutil.copy2", sidecar_fails):
        for _ in range(6):
            _backup_db_file(db)
            time.sleep(0.01)
            os.utime(db, None)

    assert len(_existing_malformed_backups(db)) <= _MAX_MALFORMED_BACKUPS

    # a later successful pass must sweep any staging debris
    with patch("shutil.disk_usage", return_value=roomy):
        path, reason = _backup_db_file(db)
    assert reason is None and path is not None
    strays = list(tmp_path.glob("*.incomplete*"))
    assert not strays, f"staging debris survived: {strays}"


def test_backup_allowed_with_ample_disk(tmp_path):
    db = _damaged_db(tmp_path)
    roomy = type(
        "Usage", (), {"total": 0, "used": 0, "free": _REPAIR_BACKUP_MIN_FREE_BYTES * 10}
    )()
    with patch("shutil.disk_usage", return_value=roomy):
        path, reason = _backup_db_file(db)
    assert reason is None and path is not None


def test_repair_aborts_when_backup_refused_for_disk(tmp_path):
    """Refused backup is a HARD STOP — never mutate the only damaged copy."""
    db = _damaged_db(tmp_path)
    tight = type(
        "Usage", (), {"total": 0, "used": 0, "free": _REPAIR_BACKUP_MIN_FREE_BYTES // 2}
    )()
    with patch("shutil.disk_usage", return_value=tight):
        report = hermes_state.repair_state_db_schema(db)
    assert not report.get("repaired")
    assert "free" in (report.get("error") or "").lower()
