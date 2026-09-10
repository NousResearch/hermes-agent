"""Refuse SessionDB open/write when a deleted WAL generation is still held.

A live writer that keeps the unlinked ``state.db-wal`` inode while a second
opener would mint a fresh WAL is the split-brain that produces intermittent
``database disk image is malformed`` / ``disk I/O error``. The store must
fail closed on both the open and write paths instead of creating the second
generation.
"""

import gc
import os
import sqlite3
import sys
from pathlib import Path

import pytest

import hermes_state
import hermes_state_dbfile
import hermes_state_readpool
import hermes_state_wal
from hermes_state import (
    DeletedWalGenerationError, SessionDB, _close_time_checkpoint_configurable, classify_persistence_error,
    refuse_deleted_wal_generation,
)
from hermes_state_dbfile import _pread_db_header, iter_deleted_sqlite_sidecar_holders
from tests.hermes_state._wal_generation_harness import (
    gateway_writer, integrity_ok_path, lose_sidecars, make_db, message_count, pin_wal, require_wal,
    write_second_generation,
)


@pytest.fixture
def force_wal(monkeypatch):
    pin_wal(monkeypatch)


def test_classify_deleted_wal_is_replaced_not_disk():
    err = DeletedWalGenerationError(
        "FATAL: a live process holds a deleted state.db-wal or state.db-shm "
        "inode while the path names a different (or missing) generation."
    )
    assert classify_persistence_error(err) == "replaced"
    assert classify_persistence_error(str(err)) == "replaced"


def test_iter_holders_empty_on_non_linux(monkeypatch, tmp_path):
    monkeypatch.setattr(hermes_state.sys, "platform", "win32")
    assert iter_deleted_sqlite_sidecar_holders(tmp_path / "state.db") == []


def test_clean_open_and_second_open_still_work(tmp_path, force_wal):
    path = tmp_path / "state.db"
    db = make_db(path, "s1", "hello")
    require_wal(db)
    db.close()
    reopened = SessionDB(db_path=path)
    try:
        reopened.append_message("s1", role="user", content="second-open")
        rows = reopened.get_messages("s1")
        assert any(m["content"] == "second-open" for m in rows)
    finally:
        reopened.close()


def test_delete_journal_two_writers_still_work(tmp_path, monkeypatch):
    monkeypatch.setattr(hermes_state_wal, "resolve_journal_mode", lambda: "delete")
    monkeypatch.setattr(
        hermes_state_wal, "is_sqlite_wal_reset_vulnerable", lambda version_info=None: False
    )
    path = tmp_path / "state.db"
    a = make_db(path, "s", "from-a")
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
    db = make_db(path, "s", "held")
    wal = require_wal(db)
    inode_before = wal.stat().st_ino
    lose_sidecars(path, rename=False)
    holders = iter_deleted_sqlite_sidecar_holders(path)
    try:
        assert holders, "expected this process to still hold the deleted WAL inode"
        assert any("(deleted)" in target for _pid, target in holders)
        assert any(
            target.removesuffix(" (deleted)").endswith(("-wal", "-shm"))
            for _pid, target in holders
        )
        assert not wal.exists() or wal.stat().st_ino != inode_before
    finally:
        db.close()


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="deleted-WAL /proc scan is Linux-only",
)
def test_second_sessiondb_open_refuses_and_does_not_mint_wal(tmp_path, force_wal):
    path = tmp_path / "state.db"
    writer = make_db(path, "s", "before-unlink")
    wal = require_wal(writer)
    inode_before = wal.stat().st_ino
    lose_sidecars(path, rename=False)
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
    reason="deleted-WAL /proc scan is Linux-only",
)
def test_iter_holders_ignores_live_unhashed_dentry(tmp_path, force_wal, monkeypatch):
    """OpenZFS can report a live, still-linked file's /proc fd target with the
    `` (deleted)`` suffix (dentry unhashed, nlink still 1) even though nothing
    was actually unlinked. The scan must not treat that as an orphaned WAL."""
    path = tmp_path / "state.db"
    db = _make_db(path, "s", "held")
    wal = _require_wal(db)
    real_readlink = os.readlink

    def fake_readlink(fd_path, *args, **kwargs):
        target = real_readlink(fd_path, *args, **kwargs)
        if target.endswith(("-wal", "-shm")):
            return target + " (deleted)"
        return target

    monkeypatch.setattr(hermes_state_dbfile.os, "readlink", fake_readlink)
    try:
        assert iter_deleted_sqlite_sidecar_holders(path) == []
        assert wal.exists()
    finally:
        db.close()


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="deleted-WAL write halt uses Linux unlink semantics",
)
def test_write_path_ignores_live_unhashed_dentry(tmp_path, force_wal, monkeypatch):
    """Same OpenZFS artifact as above, but on the sticky in-process write-path
    probe (_wal_generation_was_lost), which the open-path fix alone does not cover."""
    path = tmp_path / "state.db"
    db = _make_db(path, "s", "held")
    _require_wal(db)
    # Mimic the post-close-race state that forces the /proc probe path.
    db._db_sidecar_identity = {}
    real_readlink = os.readlink

    def fake_readlink(fd_path, *args, **kwargs):
        target = real_readlink(fd_path, *args, **kwargs)
        if target.endswith(("-wal", "-shm")):
            return target + " (deleted)"
        return target

    monkeypatch.setattr(hermes_state_readpool.os, "readlink", fake_readlink)
    try:
        assert db._wal_generation_was_lost() is False
        db.append_message("s", role="user", content="after-artifact")
        rows = db.get_messages("s")
        assert any(m["content"] == "after-artifact" for m in rows)
    finally:
        db.close()


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="deleted-WAL write halt uses Linux unlink semantics",
)
def test_writer_halts_after_own_wal_unlinked(tmp_path, force_wal):
    path = tmp_path / "state.db"
    db = make_db(path, "s", "before")
    require_wal(db)
    recorded = db._db_sidecar_identity.get("-wal")
    assert recorded is not None
    lose_sidecars(path, rename=False)

    with pytest.raises(DeletedWalGenerationError, match="deleted state.db-wal"):
        db.append_message("s", role="user", content="after-unlink")
    assert db._db_wal_generation_lost is True

    with pytest.raises(DeletedWalGenerationError):
        db.append_message("s", role="user", content="second-after-halt")
    db.close()


@pytest.mark.skipif(_close_time_checkpoint_configurable(),
                    reason="setconfig can switch the close-time checkpoint off: no retirement capability needed")
def test_missing_retirement_capability_refuses_writer_before_open(tmp_path, monkeypatch):
    import ctypes

    existing = tmp_path / "existing.db"
    make_db(existing, "seed", "read-only access remains available").close()

    class MissingPythonAPI:
        def __getitem__(self, name):
            raise AttributeError(name)

    monkeypatch.setattr(ctypes, "pythonapi", MissingPythonAPI())
    unopened = tmp_path / "unopened" / "state.db"
    with pytest.raises(RuntimeError, match="requires CPython with ctypes"):
        SessionDB(db_path=unopened)
    assert not unopened.parent.exists()
    with SessionDB(db_path=existing, read_only=True) as reader:
        assert reader.get_messages("seed")[0]["content"] == "read-only access remains available"


# ── Integrity across close/shutdown, not just "no explicit PRAGMA" ─────────────────────────────────
#
# Asserting that no ``wal_checkpoint`` string was executed at close would be necessary but NOT sufficient:
# on Python < 3.12 ``sqlite3.Connection.close()`` runs SQLite's own internal last-connection checkpoint
# with no SQL of ours, and it checkpoints the STALE generation's frames over pages the newer generation
# already rewrote — the #105670 corruption. These tests unlink the sidecars for real, write and checkpoint
# a second generation through the path, then assert the FILE is intact (and the second generation's rows
# survive) both after ``SessionDB.close()`` and after the writer PROCESS exits.


def _deleted_wal_descriptor(identity: tuple, fd_directory: str) -> int:
    """Find the test's exact unlinked inode without extending its lifetime."""
    for name in os.listdir(fd_directory):
        try:
            fd = int(name)
            stat = os.fstat(fd)
            if (stat.st_dev, stat.st_ino) == identity:
                assert stat.st_nlink == 0
                return fd
        except OSError:
            continue
    pytest.fail("the owning SQLite connection discarded its unlinked WAL")


def _descriptor_contents(fd: int, identity: tuple) -> bytes:
    stat = os.fstat(fd)
    assert (stat.st_dev, stat.st_ino) == identity, "the retained descriptor was closed or reused"
    return os.pread(fd, stat.st_size, 0)


def _assert_retirement_preserves_recoverable_wal(tmp_path, *, fd_directory, also_corrupt):
    """Keep committed old-WAL-only data recoverable across close().

    A second valid WAL uses the same deleted pathname but belongs to another
    connection; closing the first owner must not modify that inode. Where the
    runtime cannot switch off SQLite's close-time checkpoint the owner is retired
    unclosed and its inode stays readable; elsewhere the capture taken at close()
    is the copy. Neither claims that unlinked files survive process exit.
    """
    path = tmp_path / "state.db"
    db = make_db(path, "gw-0", "seed")
    wal = require_wal(db)
    db._conn.execute("PRAGMA wal_autocheckpoint=0")
    db._try_wal_checkpoint()
    sentinel = "committed only in the retired WAL " + "x" * 3000
    db.append_message("gw-0", role="assistant", content=sentinel)
    original_stat = wal.stat()
    original_identity = (original_stat.st_dev, original_stat.st_ino)
    lose_sidecars(path, rename=False)

    # SQLite is the sole owner of the original inode; dup/open here would mask
    # a close() that incorrectly discards the only remaining copy.
    original_fd = _deleted_wal_descriptor(original_identity, fd_directory)
    original_contents = _descriptor_contents(original_fd, original_identity)
    assert original_contents

    other_path = tmp_path / "other.db"
    other = sqlite3.connect(str(other_path))
    try:
        other.execute("PRAGMA journal_mode=WAL")
        other.execute("CREATE TABLE independent (content TEXT)")
        other.execute("INSERT INTO independent VALUES ('another owner committed this')")
        other.commit()
        other_wal = Path(str(other_path) + "-wal")
        other_wal.rename(wal)
        other_stat = wal.stat()
        other_identity = (other_stat.st_dev, other_stat.st_ino)
        wal.unlink()
        other_fd = _deleted_wal_descriptor(other_identity, fd_directory)
        other_contents = _descriptor_contents(other_fd, other_identity)
        assert original_identity != other_identity
        assert other_contents

        # A previously observed structural error must not bypass lost-generation retirement.
        db._db_corrupt = also_corrupt
        db.close()
        db.close()  # repeated owner release must not take another permanent reference
        assert db._db_wal_generation_lost is True
        assert db._conn is None
        artifact = db._retired_generation_capture
        assert artifact is not None
        del db
        gc.collect()

        if _close_time_checkpoint_configurable():
            # Closed for real: the capture taken at close() is the surviving copy.
            retired_wal_bytes = (artifact / "state.db-wal").read_bytes()
        else:
            # Retired unclosed: idle readers may have closed, but the writer still holds this inode.
            original_fd = _deleted_wal_descriptor(original_identity, fd_directory)
            retired_wal_bytes = _descriptor_contents(original_fd, original_identity)
        assert retired_wal_bytes == original_contents
        assert _descriptor_contents(other_fd, other_identity) == other_contents

        # The writer is now quiescent. Recover into copied files, never reopen
        # its original path in this process while it holds the old SHM inode.
        recovery_path = tmp_path / "recovery.db"
        # Reuse the cached main-file descriptor: opening and closing a new one
        # here would cancel this process's SQLite advisory locks.
        image_size = path.stat().st_size
        main_image = _pread_db_header(path, image_size)
        assert main_image is not None and len(main_image) == image_size
        recovery_path.write_bytes(main_image)
        control = sqlite3.connect(recovery_path.as_uri() + "?immutable=1", uri=True)
        try:
            assert control.execute(
                "SELECT COUNT(*) FROM messages WHERE content = ?", (sentinel,)
            ).fetchone()[0] == 0, "control transaction was already in the main database"
        finally:
            control.close()
        Path(str(recovery_path) + "-wal").write_bytes(retired_wal_bytes)
        recovered = sqlite3.connect(str(recovery_path))
        try:
            assert recovered.execute(
                "SELECT COUNT(*) FROM messages WHERE content = ?", (sentinel,)
            ).fetchone()[0] == 1
            assert recovered.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        finally:
            recovered.close()
    finally:
        other.close()


@pytest.mark.linux_only
@pytest.mark.parametrize("also_corrupt", [False, True])
def test_retirement_preserves_recoverable_wal_and_other_deleted_generation(tmp_path, force_wal, also_corrupt):
    _assert_retirement_preserves_recoverable_wal(
        tmp_path, fd_directory="/proc/self/fd", also_corrupt=also_corrupt,
    )


@pytest.mark.macos_only
@pytest.mark.parametrize("also_corrupt", [False, True])
def test_retirement_preserves_unlinked_wal_on_macos(tmp_path, force_wal, also_corrupt):
    _assert_retirement_preserves_recoverable_wal(
        tmp_path, fd_directory="/dev/fd", also_corrupt=also_corrupt,
    )


def _assert_new_generation_survives_retirement_and_exit(tmp_path, *, rename_sidecars):
    with gateway_writer(tmp_path) as gw:
        path = gw.path
        gw.next_event("ready")
        for suffix in ("-wal", "-shm"):
            assert Path(str(path) + suffix).exists()
        lose_sidecars(path, rename=rename_sidecars)
        expected = write_second_generation(path, n_rows=400)
        assert integrity_ok_path(path)

        gw.send("write")
        assert gw.next_event("write")["refused"] is True

        gw.send("close")
        gw.next_event("closed")
        assert integrity_ok_path(path), "close/GC checkpointed the stale WAL over the main file"
        assert message_count(path) == expected, "close/GC rolled back the newer rows"

        gw.send("quit")
        assert gw.wait_exit(timeout=20) == 0, gw.stderr_text()
        assert integrity_ok_path(path), "normal exit checkpointed the stale WAL over the main file"
        assert message_count(path) == expected, "normal exit rolled back the newer rows"


@pytest.mark.linux_only
def test_deleted_wal_generation_survives_close_and_clean_process_exit(tmp_path):
    _assert_new_generation_survives_retirement_and_exit(tmp_path, rename_sidecars=False)


@pytest.mark.macos_only
def test_renamed_wal_generation_survives_close_and_clean_process_exit(tmp_path):
    _assert_new_generation_survives_retirement_and_exit(tmp_path, rename_sidecars=True)


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
