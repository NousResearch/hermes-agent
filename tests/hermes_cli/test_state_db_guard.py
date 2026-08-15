"""Tests for the state.db integrity guard used by the update flow (#68474).

Exercises ``verify_sqlite_integrity`` and ``copy_db_and_verify`` against REAL
SQLite files (valid, zeroed, truncated) — the exact corruption signature from
issue #68474 (file kept at original size, 100% null bytes, header gone).
"""

import sqlite3

import pytest

from hermes_cli.backup import copy_db_and_verify, verify_sqlite_integrity


@pytest.fixture()
def valid_db(tmp_path):
    path = tmp_path / "state.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY, name TEXT)")
    conn.executemany(
        "INSERT INTO sessions (name) VALUES (?)", [(f"s{i}",) for i in range(50)]
    )
    conn.commit()
    conn.close()
    return path










def test_oversized_db_probe_catches_malformed_schema(tmp_path):
    """Skipping the pragma must not mean skipping corruption detection.

    A file with a valid header whose schema cannot be parsed has to fail
    the oversized path via the cheap structural probe.
    """
    path = tmp_path / "state.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (a INTEGER)")
    conn.commit()
    conn.close()

    raw = bytearray(path.read_bytes())
    # Corrupt the schema b-tree page (page 2 onward) while leaving the
    # 16-byte header magic intact, so only the probe can catch it.
    for i in range(100, min(len(raw), 4096)):
        raw[i] = 0xFF
    path.write_bytes(bytes(raw))

    res = verify_sqlite_integrity(path, max_bytes=1)
    assert res["valid"] is False
    assert "probe" in res["message"]






def test_restore_flow_end_to_end(valid_db, tmp_path):
    """Simulate the #68474 recovery path: live db zeroed, snapshot valid →
    restore snapshot over live file → verify restored copy."""
    import shutil

    snap = tmp_path / "state-snapshots" / "20260721-pre-update" / "state.db"
    snap.parent.mkdir(parents=True)
    shutil.copy2(valid_db, snap)

    # Zero the live db (the bug).
    size = valid_db.stat().st_size
    valid_db.write_bytes(b"\x00" * size)
    assert verify_sqlite_integrity(valid_db)["valid"] is False
    assert verify_sqlite_integrity(snap)["valid"] is True

    # The guard's restore step.
    shutil.copy2(snap, valid_db)
    restored = verify_sqlite_integrity(valid_db)
    assert restored["valid"] is True
    conn = sqlite3.connect(valid_db)
    assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 50
    conn.close()


class TestPreUpdateBackupIntegrityGuard:
    """E2E: run the real ``_run_pre_update_backup`` against a temp
    HERMES_HOME whose state.db is corrupted mid-flight (#68474)."""

    @pytest.fixture()
    def hermes_home(self, tmp_path, monkeypatch):
        from pathlib import Path
        import sys

        root = tmp_path / ".hermes"
        root.mkdir()
        (root / "config.yaml").write_text("model:\n  provider: openrouter\n")
        db = root / "state.db"
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()
        monkeypatch.setenv("HERMES_HOME", str(root))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        for mod in list(sys.modules.keys()):
            if mod.startswith("hermes_cli.config") or mod == "hermes_constants":
                del sys.modules[mod]
        return root

    def test_healthy_db_stays_quiet(self, hermes_home, capsys):
        from argparse import Namespace

        from hermes_cli.main import _run_pre_update_backup

        snap_id = _run_pre_update_backup(Namespace(no_backup=False, backup=False))
        out = capsys.readouterr().out
        assert snap_id is not None
        assert "Pre-update snapshot" in out
        assert "integrity check FAILED" not in out

    def test_zeroed_db_after_snapshot_is_loud(self, hermes_home, capsys, monkeypatch):
        """If state.db is zeroed right after the snapshot completes, the
        guard must warn loudly instead of proceeding silently (exit-0 mask)."""
        from argparse import Namespace

        import hermes_cli.backup as backup_mod
        from hermes_cli.main import _run_pre_update_backup

        real_create = backup_mod.create_quick_snapshot

        def create_then_zero(**kwargs):
            snap_id = real_create(**kwargs)
            live = hermes_home / "state.db"
            live.write_bytes(b"\x00" * live.stat().st_size)
            return snap_id

        monkeypatch.setattr(backup_mod, "create_quick_snapshot", create_then_zero)
        snap_id = _run_pre_update_backup(Namespace(no_backup=False, backup=False))
        out = capsys.readouterr().out
        assert snap_id is not None
        assert "integrity check FAILED" in out
        assert "Snapshot copy is valid" in out
