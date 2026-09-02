"""Refuse SessionDB open/write when a deleted WAL generation is still held.

A live writer that keeps the unlinked ``state.db-wal`` inode while a second
opener would mint a fresh WAL is the split-brain that produces intermittent
``database disk image is malformed`` / ``disk I/O error``. The store must
fail closed on both the open and write paths instead of creating the second
generation.
"""

import os
import sqlite3
import sys
from pathlib import Path

import pytest

import hermes_state
from hermes_state import (
    DeletedWalGenerationError,
    SessionDB,
    classify_persistence_error,
    iter_deleted_sqlite_sidecar_holders,
    refuse_deleted_wal_generation,
)


@pytest.fixture
def force_wal(monkeypatch):
    """Pin WAL so this host's vulnerable SQLite still matches production topology."""
    monkeypatch.setattr(
        hermes_state, "is_sqlite_wal_reset_vulnerable", lambda version_info=None: False
    )
    monkeypatch.setattr(hermes_state, "resolve_journal_mode", lambda: "wal")


def _make_db(path: Path, session_id: str, content: str) -> SessionDB:
    db = SessionDB(db_path=path)
    db.create_session(session_id, "cli")
    db.append_message(session_id, role="user", content=content)
    return db


def _require_wal(db: SessionDB) -> Path:
    if not db._wal_active:
        db.close()
        pytest.skip("WAL not active on this filesystem")
    wal = Path(os.fspath(db.db_path) + "-wal")
    if not wal.exists():
        db.close()
        pytest.skip("WAL sidecar missing after first write")
    return wal


def _unlink_sidecars(db_path: Path) -> None:
    for suffix in ("-wal", "-shm"):
        sidecar = Path(os.fspath(db_path) + suffix)
        if sidecar.exists():
            os.unlink(sidecar)


def test_classify_deleted_wal_is_replaced_not_disk():
    err = DeletedWalGenerationError(
        "FATAL: a live process holds a deleted state.db-wal or state.db-shm "
        "inode while the path names a different (or missing) generation."
    )
    assert classify_persistence_error(err) == "replaced"
    assert classify_persistence_error(str(err)) == "replaced"


def test_iter_holders_empty_on_non_linux(monkeypatch, tmp_path):
    monkeypatch.setattr(hermes_state, "_IS_WINDOWS", True)
    assert iter_deleted_sqlite_sidecar_holders(tmp_path / "state.db") == []


def test_clean_open_and_second_open_still_work(tmp_path, force_wal):
    path = tmp_path / "state.db"
    db = _make_db(path, "s1", "hello")
    _require_wal(db)
    db.close()
    reopened = SessionDB(db_path=path)
    try:
        reopened.append_message("s1", role="user", content="second-open")
        rows = reopened.get_messages("s1")
        assert any(m["content"] == "second-open" for m in rows)
    finally:
        reopened.close()


def test_delete_journal_two_writers_still_work(tmp_path, monkeypatch):
    monkeypatch.setattr(hermes_state, "resolve_journal_mode", lambda: "delete")
    monkeypatch.setattr(
        hermes_state, "is_sqlite_wal_reset_vulnerable", lambda version_info=None: False
    )
    path = tmp_path / "state.db"
    a = _make_db(path, "s", "from-a")
    try:
        assert not Path(os.fspath(path) + "-wal").exists()
        b = SessionDB(db_path=path)
        try:
            b.append_message("s", role="user", content="from-b")
            contents = [m["content"] for m in b.get_messages("s")]
            assert "from-a" in contents
            assert "from-b" in contents
        finally:
            b.close()
    finally:
        a.close()


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="deleted-WAL /proc scan is Linux-only",
)
def test_iter_finds_self_after_wal_unlink(tmp_path, force_wal):
    path = tmp_path / "state.db"
    db = _make_db(path, "s", "held")
    wal = _require_wal(db)
    inode_before = wal.stat().st_ino
    _unlink_sidecars(path)
    holders = iter_deleted_sqlite_sidecar_holders(path, include_self=True)
    try:
        assert holders, "expected this process to still hold the deleted WAL inode"
        assert any("(deleted)" in target for _pid, target in holders)
        assert any(target.rstrip(" (deleted)").endswith("-wal") or
                   target.rstrip(" (deleted)").endswith("-shm")
                   for _pid, target in holders)
        assert not wal.exists() or wal.stat().st_ino != inode_before
    finally:
        db.close()


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="deleted-WAL /proc scan is Linux-only",
)
def test_second_sessiondb_open_refuses_and_does_not_mint_wal(tmp_path, force_wal):
    path = tmp_path / "state.db"
    writer = _make_db(path, "s", "before-unlink")
    wal = _require_wal(writer)
    inode_before = wal.stat().st_ino
    _unlink_sidecars(path)
    assert not wal.exists()

    with pytest.raises(DeletedWalGenerationError, match="deleted state.db-wal"):
        SessionDB(db_path=path)

    assert not wal.exists(), "open must refuse before sqlite3.connect mints a WAL"
    # If a WAL somehow reappeared it must not be a new generation.
    if wal.exists():
        assert wal.stat().st_ino == inode_before
    writer.close()


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="deleted-WAL write halt uses Linux unlink semantics",
)
def test_writer_halts_after_own_wal_unlinked(tmp_path, force_wal):
    path = tmp_path / "state.db"
    db = _make_db(path, "s", "before")
    _require_wal(db)
    recorded = db._db_sidecar_identity.get("-wal")
    assert recorded is not None
    _unlink_sidecars(path)

    with pytest.raises(DeletedWalGenerationError, match="deleted state.db-wal"):
        db.append_message("s", role="user", content="after-unlink")
    assert db._db_wal_generation_lost is True

    with pytest.raises(DeletedWalGenerationError):
        db.append_message("s", role="user", content="second-after-halt")
    db.close()


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="deleted-WAL /proc scan is Linux-only",
)
def test_refuse_helper_raises_while_deleted_wal_held(tmp_path, force_wal):
    path = tmp_path / "state.db"
    raw = sqlite3.connect(str(path))
    try:
        raw.execute("PRAGMA journal_mode=WAL")
        raw.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
        raw.execute("INSERT INTO t VALUES (1, 'held')")
        raw.commit()
        wal = Path(str(path) + "-wal")
        assert wal.exists()
        os.unlink(wal)
        shm = Path(str(path) + "-shm")
        if shm.exists():
            os.unlink(shm)
        with pytest.raises(DeletedWalGenerationError):
            refuse_deleted_wal_generation(path)
        assert not wal.exists()
    finally:
        raw.close()
