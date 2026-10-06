"""File previews may not cancel SQLite's live POSIX locks (including WAL sidecars)."""
import asyncio
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest
from fastapi import HTTPException

from agent.context_references import _expand_path_reference, parse_context_references
from hermes_state import SessionDB
from hermes_cli.web_routers.files import fs_download, fs_read_text
from tests.posix_lock_probe import own_posix_locks


@pytest.mark.platforms("linux")
@pytest.mark.requires_wal
@pytest.mark.parametrize("route,target_kind", [
    ("file", "main"), ("folder", "directory"), ("desktop", "main"), ("desktop", "shm"),
])
def test_preview_preserves_live_database_locks(tmp_path, route, target_kind):
    path = tmp_path / "state.db"
    text = tmp_path / "normal.txt"
    text.write_text("ordinary readable text", encoding="utf-8")
    db = SessionDB(path)
    lock_fd = None
    try:
        db.create_session("preview-test", "cli")
        db.append_message("preview-test", "user", "before preview")
        shm = Path(str(path) + "-shm")
        target = {"main": path, "shm": shm, "directory": tmp_path}[target_kind]
        conn = db._conn
        assert isinstance(conn, sqlite3.Connection)
        conn.execute("CREATE TABLE preview_markers (value TEXT)")
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT INTO preview_markers VALUES ('first process')")
        # WAL writers reliably hold -shm locks, but not always a main-file lock.
        # Hold our own POSIX main-file lock to prove that preview close() does
        # not cancel any locks on that inode, regardless of SQLite's WAL timing.
        import fcntl
        lock_fd = os.open(path, os.O_RDONLY)
        fcntl.lockf(lock_fd, fcntl.LOCK_SH, 1, 4096)

        def rival_locked():
            code = ("import sqlite3,sys; c=sqlite3.connect(sys.argv[1], timeout=0); "
                    "c.execute('BEGIN IMMEDIATE'); c.rollback(); c.close()")
            result = subprocess.run([sys.executable, "-c", code, str(path)],
                                    capture_output=True, text=True, timeout=10)
            return result.returncode != 0 and "database is locked" in result.stderr

        before = (own_posix_locks(path), own_posix_locks(shm))
        assert all(before), "fixture must hold POSIX main and WAL-sidecar locks"
        assert rival_locked(), "second process must be excluded before the preview"

        if route == "desktop":
            # FileResponse opens/closes in-process too, so a download must be refused as well,
            # with the same (sidecar-aware) refusal text as the read.
            for route_fn in (fs_read_text, lambda p: asyncio.run(fs_download(p))):
                with pytest.raises(HTTPException) as refused:
                    route_fn(str(target))
                assert refused.value.status_code == 409
                if target_kind == "shm":
                    assert "main database" in refused.value.detail
            assert fs_read_text(str(text))["text"] == "ordinary readable text"
        else:
            ref = parse_context_references(f"@{route}:{target}")[0]
            warning, block = _expand_path_reference(ref, tmp_path.parent)
            assert warning is None
            assert block is not None
            if route == "file":
                assert "not previewed" in block
            ordinary = parse_context_references(f"@file:{text}")[0]
            warning, block = _expand_path_reference(ordinary, tmp_path.parent)
            assert warning is None and block is not None and "ordinary readable text" in block

        assert (own_posix_locks(path), own_posix_locks(shm)) == before
        assert rival_locked(), "second process entered a still-open write transaction"
        conn.commit()
        code = (
            "import sqlite3,sys; c=sqlite3.connect(sys.argv[1], timeout=2); "
            "assert c.execute('SELECT value FROM preview_markers').fetchall() == [('first process',)]; "
            "c.execute(\"INSERT INTO preview_markers VALUES ('second process')\"); "
            "c.commit(); c.close()"
        )
        rival = subprocess.run([sys.executable, "-c", code, str(path)],
                               capture_output=True, text=True, timeout=10)
        assert rival.returncode == 0, rival.stderr
        assert [row[0] for row in conn.execute(
            "SELECT value FROM preview_markers ORDER BY rowid"
        )] == ["first process", "second process"]
        db.append_message("preview-test", "assistant", "after preview")
        assert len(db.get_messages("preview-test")) == 2
    finally:
        db.close()
        if lock_fd is not None:
            os.close(lock_fd)


def test_closed_database_can_still_be_previewed(tmp_path):
    path = tmp_path / "offline.db"
    db = SessionDB(path)
    db.create_session("offline", "cli")
    db.close()
    ref = parse_context_references(f"@file:{path}")[0]
    warning, block = _expand_path_reference(ref, tmp_path.parent)
    assert warning is None and block is not None and "binary file" in block
    preview = fs_read_text(str(path))
    assert preview["binary"] is True and preview["byteSize"] == path.stat().st_size
    assert asyncio.run(fs_download(str(path))).path == str(path)
