"""`hermes sessions repair --check-only` must fail (non-zero) when the store is unhealthy.

Automation gates on the exit status; printing the reason and exiting 0 read as "healthy"
(#63386, PR #103321).
"""
import argparse
from pathlib import Path

from hermes_cli import sessions_cmd
from persistence_assembly import open_persistence, profile_scope


def test_check_only_exit_status_tracks_probe_verdict(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    with open_persistence(profile_scope(home=tmp_path)) as store:
        store.ready()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    args = argparse.Namespace(check_only=True, no_backup=False)

    assert not sessions_cmd._cmd_repair(args)
    # Destroy the schema header so the probe fails on its first statement.
    with open(db_path, "r+b") as f:
        f.seek(100)
        f.write(b"\xff" * (4096 - 100))
    for side in ("-wal", "-shm"):
        Path(str(db_path) + side).unlink(missing_ok=True)
    assert sessions_cmd._cmd_repair(args) == 1
    assert db_path.stat().st_size > 0, "check-only must not touch the file"
