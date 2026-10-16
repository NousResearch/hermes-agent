"""A WAL/SHM sidecar that SQLite removes mid-preflight is not "read-only".

Another connection's checkpoint or close can delete ``-wal``/``-shm`` between the
preflight's ``is_file()`` listing and its ``os.access`` check. ``os.access`` then reports
False for the missing file and the preflight used to raise "is read-only for this user",
failing an unrelated Kanban/cron open. Runs as any user, root included.
"""

import os
from pathlib import Path

import pytest

import hermes_state_repair
from hermes_state import preflight_db_writability


@pytest.mark.parametrize("suffix", ["-wal", "-shm"])
def test_sidecar_removed_during_preflight_is_ignored(tmp_path, monkeypatch, suffix):
    db = tmp_path / "kanban.db"
    db.write_bytes(b"")
    sidecar = Path(str(db) + suffix)
    sidecar.write_bytes(b"")
    real_access = os.access

    def racing_access(path, mode, *args, **kwargs):
        if Path(path) == sidecar:
            sidecar.unlink(missing_ok=True)  # the checkpoint wins the race
            return False
        return real_access(path, mode, *args, **kwargs)

    monkeypatch.setattr(hermes_state_repair.os, "access", racing_access)
    preflight_db_writability(db, db_label="kanban.db")  # must not raise


def test_existing_unwritable_sidecar_still_fails(tmp_path, monkeypatch):
    db = tmp_path / "kanban.db"
    db.write_bytes(b"")
    sidecar = Path(str(db) + "-wal")
    sidecar.write_bytes(b"")
    real_access = os.access
    monkeypatch.setattr(hermes_state_repair.os, "access",
                        lambda path, mode, *a, **k: False if Path(path) == sidecar else real_access(path, mode, *a, **k))
    monkeypatch.setattr(hermes_state_repair.os, "chmod", lambda *a, **k: None)
    with pytest.raises(Exception, match="read-only for this user"):
        preflight_db_writability(db, db_label="kanban.db")
