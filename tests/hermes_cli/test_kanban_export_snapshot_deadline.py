"""Board export refuses a persistently busy snapshot and preserves WAL data."""

import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tarfile

import pytest


def _new_board(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb

    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "home"))
    for key in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD"):
        monkeypatch.delenv(key, raising=False)
    kb.create_board("alpha")
    return kb.kanban_db_path("alpha")


@pytest.mark.platforms("posix")
def test_export_refuses_locked_snapshot_without_replacing_archive(tmp_path, monkeypatch):
    source = _new_board(tmp_path, monkeypatch)
    archive = tmp_path / "export.tar.gz"
    archive.write_bytes(b"previous export")
    worktree = Path(__file__).resolve().parents[2]
    command = """
import sys
from hermes_cli.kanban_transfer import export_board
try:
    export_board('alpha', sys.argv[1])
except RuntimeError as exc:
    print(str(exc))
    sys.exit(2)
"""
    with sqlite3.connect(source) as blocker:
        assert blocker.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete"
        source_bytes = source.read_bytes()
        blocker.execute("BEGIN EXCLUSIVE")
        child = subprocess.Popen(
            [sys.executable, "-c", command, str(archive)],
            cwd=worktree,
            env={**os.environ, "PYTHONPATH": str(worktree)},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            # The canonical backup helper allows ten seconds of BUSY replies;
            # give interpreter startup and filesystem work another ten seconds.
            stdout, stderr = child.communicate(timeout=20)
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate()
            blocker.rollback()

    assert child.returncode == 2, stderr
    assert "consistent SQLite snapshot" in stdout
    assert archive.read_bytes() == b"previous export"
    assert source.read_bytes() == source_bytes


def test_export_keeps_committed_uncheckpointed_wal_rows(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli.kanban_transfer import export_board

    source = _new_board(tmp_path, monkeypatch)
    # Keep the writer alive, so closing the last connection cannot checkpoint
    # the committed row before the actual board-export snapshot is made.
    with kbc.connect_closing(board="alpha") as writer:
        writer.execute("PRAGMA wal_autocheckpoint=0")
        task_id = kb.create_task(writer, title="latest WAL task", body="committed content")
        assert Path(str(source) + "-wal").stat().st_size > 0
        # Verify the main file alone does not already contain the new row.
        with sqlite3.connect(source.resolve().as_uri() + "?immutable=1", uri=True) as main_only:
            assert main_only.execute("SELECT COUNT(*) FROM tasks WHERE id=?", (task_id,)).fetchone()[0] == 0

        result = export_board("alpha", str(tmp_path / "export"))
        with tarfile.open(result["archive"], "r:gz") as archive:
            db_file = archive.extractfile("alpha/kanban.db")
            assert db_file is not None
            snapshot = tmp_path / "snapshot.db"
            snapshot.write_bytes(db_file.read())
        with sqlite3.connect(snapshot) as exported:
            assert exported.execute("PRAGMA quick_check").fetchall() == [("ok",)]
            assert exported.execute("SELECT title, body FROM tasks WHERE id=?", (task_id,)).fetchone() == (
                "latest WAL task", "committed content",
            )
        assert writer.execute("SELECT body FROM tasks WHERE id=?", (task_id,)).fetchone()[0] == "committed content"
    assert result["counts"]["tasks"] == 1
