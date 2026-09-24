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
from hermes_cli.web_routers.files import fs_read_text


@pytest.mark.linux_only
@pytest.mark.parametrize("route,target_kind", [
    ("file", "main"), ("file", "shm"), ("file", "shm_alias"), ("file", "wal"),
    ("folder", "directory"), ("desktop", "main"),
    ("desktop", "shm"), ("desktop", "shm_alias"), ("desktop", "wal"),
])
def test_preview_preserves_live_database_locks(tmp_path, route, target_kind):
    path = tmp_path / "state.db"
    text = tmp_path / "normal.txt"
    text.write_text("ordinary readable text", encoding="utf-8")
    db = SessionDB(path)
    try:
        db.create_session("preview-test", "cli")
        db.append_message("preview-test", "user", "before preview")
        shm = Path(str(path) + "-shm")
        wal = Path(str(path) + "-wal")
        alias = tmp_path / "linked-shm"
        alias.symlink_to(shm)
        target = {"main": path, "shm": shm, "shm_alias": alias,
                  "wal": wal, "directory": tmp_path}[target_kind]
        conn = db._conn
        assert isinstance(conn, sqlite3.Connection)
        conn.execute("BEGIN IMMEDIATE")

        def posix_locks(file):
            inode = file.stat().st_ino
            return sorted(line.split(": ", 1)[1] for line in Path("/proc/locks").read_text(encoding="utf-8").splitlines()
                          if f":{inode} " in line and f"POSIX  ADVISORY" in line
                          and f" {os.getpid()} " in line)

        def rival_locked():
            code = ("import sqlite3,sys; c=sqlite3.connect(sys.argv[1], timeout=0); "
                    "c.execute('BEGIN IMMEDIATE'); c.rollback(); c.close()")
            result = subprocess.run([sys.executable, "-c", code, str(path)],
                                    capture_output=True, text=True, timeout=10)
            return result.returncode != 0 and "database is locked" in result.stderr

        before = (posix_locks(path), posix_locks(shm))
        assert all(before), "fixture must hold POSIX main and WAL-sidecar locks"
        assert rival_locked(), "second process must be excluded before the preview"

        if route == "desktop":
            with pytest.raises(HTTPException) as refused:
                asyncio.run(fs_read_text(str(target)))
            assert refused.value.status_code == 409
            assert asyncio.run(fs_read_text(str(text)))["text"] == "ordinary readable text"
        else:
            ref = parse_context_references(f"@{route}:{target}")[0]
            warning, block = _expand_path_reference(ref, tmp_path.parent)
            assert warning is None
            assert block is not None
            assert "not previewed" in block if route == "file" else "state.db" in block
            ordinary = parse_context_references(f"@file:{text}")[0]
            warning, block = _expand_path_reference(ordinary, tmp_path.parent)
            assert warning is None and block is not None and "ordinary readable text" in block

        assert (posix_locks(path), posix_locks(shm)) == before
        assert rival_locked(), "second process entered a still-open write transaction"
        conn.rollback()
        db.append_message("preview-test", "assistant", "after preview")
        assert len(db.get_messages("preview-test")) == 2
    finally:
        db.close()


def test_closed_database_can_still_be_previewed(tmp_path):
    path = tmp_path / "offline.db"
    db = SessionDB(path)
    db.create_session("offline", "cli")
    db.close()
    ref = parse_context_references(f"@file:{path}")[0]
    warning, block = _expand_path_reference(ref, tmp_path.parent)
    assert warning is None and block is not None and "binary file" in block
    preview = asyncio.run(fs_read_text(str(path)))
    assert preview["binary"] is True and preview["byteSize"] == path.stat().st_size
