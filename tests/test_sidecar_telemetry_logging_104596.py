"""#104596 telemetry: WAL-generation teardown and journal-mode probe failures must be visible.

The state.db split-brain could never be attributed because every sidecar removal and every
failed journal-mode probe was silent. These pin the warning contract: removal logs carry the
path, the probe warning fires once per (process, db_label), and the ephemeral open/close sites
against state.db announce themselves once per process.
"""
import logging
import sqlite3

import pytest

import hermes_state_repair
import gateway.lifecycle_ledger as lifecycle_mod
import gateway.readiness as readiness_mod
from hermes_state_wal import _on_disk_journal_mode, _probe_unknown_warned_lock, _probe_unknown_warned_paths


@pytest.fixture(autouse=True)
def _reset_telemetry_state():
    def _clear():
        with _probe_unknown_warned_lock:
            _probe_unknown_warned_paths.clear()
        readiness_mod._ephemeral_probe_hazard_warned = False
        lifecycle_mod._integrity_ephemeral_warned = False

    _clear()
    yield
    _clear()


def test_unlink_db_triple_warns_with_path(tmp_path, caplog):
    db = tmp_path / "scratch.db"
    db.write_bytes(b"x")
    with caplog.at_level(logging.WARNING):
        assert hermes_state_repair._unlink_db_triple(db) is None
    assert any(str(db) in rec.getMessage() for rec in caplog.records)


def test_clear_stale_sqlite_sidecars_warns_with_removed(tmp_path, caplog):
    from hermes_cli.update_cmd_maint import _clear_stale_sqlite_sidecars

    db = tmp_path / "state.db"
    db.write_bytes(b"SQLite format 3")
    for suffix in ("-wal", "-shm"):
        (tmp_path / ("state.db" + suffix)).write_bytes(b"x")
    with caplog.at_level(logging.WARNING):
        _clear_stale_sqlite_sidecars(db)
    assert not (tmp_path / "state.db-wal").exists()
    assert any("state.db-wal" in rec.getMessage() and "state.db-shm" in rec.getMessage()
               for rec in caplog.records)


def test_backup_restore_fallback_warns_removed_sidecars(tmp_path, caplog, monkeypatch):
    import hermes_cli.backup as backup_mod

    src = tmp_path / "snap.db"
    src.write_bytes(b"")
    dst = tmp_path / "state.db"
    dst.write_bytes(b"x")
    (tmp_path / "state.db-wal").write_bytes(b"w")
    monkeypatch.setattr(backup_mod, "_foreign_db_holder_pids", lambda p: [])
    with caplog.at_level(logging.WARNING):
        assert backup_mod._unlink_move_restore_db(src, dst) is True
    assert any("state.db-wal" in rec.getMessage() for rec in caplog.records)


class _LockedConn:
    def execute(self, _sql):
        raise sqlite3.OperationalError("database is locked")


def test_probe_unknown_warns_once_per_db(caplog):
    with caplog.at_level(logging.WARNING):
        assert _on_disk_journal_mode(_LockedConn(), db_label="locked.db") is None
        assert _on_disk_journal_mode(_LockedConn(), db_label="locked.db") is None
        assert _on_disk_journal_mode(_LockedConn(), db_label="other.db") is None
    locked_msgs = [rec.getMessage() for rec in caplog.records if "locked.db" in rec.getMessage()]
    assert len(locked_msgs) == 1
    assert any("other.db" in rec.getMessage() for rec in caplog.records)


def test_readiness_probe_warns_once(tmp_path, caplog):
    conn = sqlite3.connect(tmp_path / "state.db")
    conn.execute("CREATE TABLE t(x)")
    conn.commit()
    conn.close()
    with caplog.at_level(logging.WARNING):
        assert readiness_mod._probe_state_db(tmp_path)["status"] == "ok"
        assert readiness_mod._probe_state_db(tmp_path)["status"] == "ok"
    hazard = [rec.getMessage() for rec in caplog.records if "#104596" in rec.getMessage()]
    assert len(hazard) == 1


def test_lifecycle_integrity_warns_once(tmp_path, caplog):
    (tmp_path / "state.db").write_bytes(b"")
    with caplog.at_level(logging.WARNING):
        lifecycle_mod.check_state_db_integrity(home=tmp_path)
        lifecycle_mod.check_state_db_integrity(home=tmp_path)
    hazard = [rec.getMessage() for rec in caplog.records if "#104596" in rec.getMessage()]
    assert len(hazard) == 1
