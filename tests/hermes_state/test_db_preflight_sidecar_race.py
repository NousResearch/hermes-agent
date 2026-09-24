"""SQLite may remove WAL sidecars between preflight discovery and access checks."""

import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from hermes_state_repair import preflight_db_writability


@pytest.mark.parametrize("suffix", ["-wal", "-shm"])
@pytest.mark.parametrize("inside_home", [True, False])
def test_checkpoint_removes_discovered_sidecar_without_readonly_failure(
    tmp_path, monkeypatch, suffix, inside_home
):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    db = (home if inside_home else tmp_path) / "kanban.db"
    conn = sqlite3.connect(db)
    assert conn.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    conn.execute("CREATE TABLE evidence (value TEXT)")
    conn.execute("INSERT INTO evidence VALUES ('committed before checkpoint')")
    conn.commit()
    sidecar = db.with_name(db.name + suffix)
    assert sidecar.is_file()
    real_access = os.access
    checkpointed = False

    def access_after_checkpoint(path, mode, *args, **kwargs):
        nonlocal checkpointed
        if Path(path) == sidecar and not checkpointed:
            # The preflight has already collected the existing sidecar. Closing
            # SQLite's last writer checkpoints its data and removes WAL + SHM.
            conn.close()
            checkpointed = True
            assert not sidecar.exists()
        return real_access(path, mode, *args, **kwargs)

    monkeypatch.setattr(os, "access", access_after_checkpoint)
    try:
        preflight_db_writability(db, db_label="kanban.db")
        assert checkpointed
        with closing(sqlite3.connect(db)) as readback:
            assert readback.execute("SELECT value FROM evidence").fetchall() == [
                ("committed before checkpoint",)
            ]
    finally:
        conn.close()


def test_existing_inaccessible_sidecar_still_refuses_open(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Outside the repair scope: a real permission denial must still fail closed.
    db = tmp_path / "kanban.db"
    with closing(sqlite3.connect(db)) as conn:
        conn.execute("CREATE TABLE evidence (value TEXT)")
    wal = db.with_name(db.name + "-wal")
    wal.write_bytes(b"preserve inaccessible sidecar")
    real_access = os.access
    monkeypatch.setattr(os, "access", lambda p, mode: False if Path(p) == wal else real_access(p, mode))

    with pytest.raises(sqlite3.OperationalError, match="read-only for this user"):
        preflight_db_writability(db, db_label="kanban.db")
    assert wal.read_bytes() == b"preserve inaccessible sidecar"
