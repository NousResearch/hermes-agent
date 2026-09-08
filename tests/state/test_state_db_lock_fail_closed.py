"""Unopenable admission lock files must fail CLOSED (#100368).

`state.db` has two cross-process admission authorities that gate destructive
work on a file several Hermes processes share (gateway service, the Desktop
app's `hermes serve` backend, CLI sessions, the TUI slash worker):

* `hermes_state_common.fts_rebuild_admission` — full structural FTS rebuilds
* `hermes_state_repair._cross_process_repair_lock`   — writable_schema surgery / VACUUM

Both document themselves as fail-closed, and both honoured that only for a
*timed-out* acquire. When the lock file could not be `open()`ed at all they
yielded True and proceeded "with in-process serialisation only" — which is no
cross-process authority whatsoever.

That inversion is reachable exactly when it does the most damage. Creating the
lock file needs a directory entry and an inode, so on a full disk `open()`
raises ENOSPC — while a sibling process that opened ITS handle before the disk
filled is still mid-rebuild or mid-surgery. Every process then ran concurrent
destructive work on the same live DB, i.e. the precise interleaving PR #93200
added these locks to prevent. #100368 reports that shape: a disk-full trigger,
then a fresh corruption on every boot with other writers alive, and no
re-corruption on a boot with zero other writers.

These tests drive a real unopenable lock path (a directory where the code
expects a file, so `open()` raises a genuine OSError from the kernel) rather
than monkeypatching the helpers, and assert the deferral is honoured at both
the primitive and the behavior level.
"""

import sqlite3
import sys
from pathlib import Path

import pytest

import hermes_state
import hermes_state_repair
import hermes_state_common
from hermes_state import SessionDB
from hermes_state_repair import repair_state_db_schema


def _make_unopenable(lock_path: Path) -> None:
    """Make ``open(lock_path, "a+b")`` raise a real OSError.

    A directory standing where the code expects a regular file yields
    IsADirectoryError on POSIX and PermissionError on Windows — both OSError,
    both raised by the kernel. This stands in for the ENOSPC/EMFILE the field
    reports hit, without needing to fill a real disk.
    """
    lock_path.unlink(missing_ok=True)
    lock_path.mkdir(parents=True, exist_ok=True)
    with pytest.raises(OSError):
        open(lock_path, "a+b").close()


# ── FTS rebuild authority ───────────────────────────────────────────────────


def test_fts_admission_fails_closed_when_lock_file_is_unopenable(tmp_path):
    """The primitive must refuse admission, not fall back to no authority."""
    db_path = tmp_path / "state.db"
    _make_unopenable(db_path.with_name(db_path.name + ".fts_rebuild.lock"))

    with hermes_state_common.fts_rebuild_admission(db_path) as admitted:
        assert admitted is False


def test_fts_admission_still_admits_a_pathless_db(tmp_path):
    """Guardrail: an in-memory store has no cross-process surface at all.

    The fix must not turn the legitimate no-op case into a permanent deferral.
    """
    with hermes_state_common.fts_rebuild_admission(None) as admitted:
        assert admitted is True


def test_rebuild_fts_defers_when_lock_file_is_unopenable(tmp_path):
    """Behavior: the rebuild entry point reports no progress and rebuilds nothing."""
    db = SessionDB(db_path=tmp_path / "state.db")
    if not db._fts_enabled:
        db.close()
        pytest.skip("FTS5 unavailable in this build")
    try:
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "hello world")

        # Sanity: with an openable lock the rebuild really runs, so a 0 below
        # is the deferral and not an unrelated no-op.
        assert db.rebuild_fts() >= 1

        _make_unopenable(
            db.db_path.with_name(db.db_path.name + ".fts_rebuild.lock")
        )
        assert db.rebuild_fts() == 0
    finally:
        try:
            db.close()
        except Exception:
            pass


# ── Schema-surgery authority ────────────────────────────────────────────────


def _build_healthy_db(db_path: Path) -> None:
    db = SessionDB(db_path=db_path)
    db.create_session("s1", source="test")
    db.append_message("s1", "user", "hello world")
    db.close()


def _corrupt_duplicate_fts(db_path: Path) -> None:
    """Inject a duplicate messages_fts row into sqlite_master.

    Reproduces 'malformed database schema (messages_fts) - table
    messages_fts already exists'.
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA writable_schema=ON")
    conn.execute(
        "INSERT INTO sqlite_master (type, name, tbl_name, rootpage, sql) "
        "SELECT type, name, tbl_name, rootpage, sql FROM sqlite_master "
        "WHERE name='messages_fts'"
    )
    conn.commit()
    conn.close()


def test_repair_lock_fails_closed_when_lock_file_is_unopenable(tmp_path):
    """The primitive must refuse the repair authority."""
    db_path = tmp_path / "state.db"
    _make_unopenable(db_path.with_name(db_path.name + ".repair.lock"))

    with hermes_state_repair._cross_process_repair_lock(db_path) as holding:
        assert holding is False


@pytest.mark.skipif(sys.platform == "win32", reason="writable_schema corruption harness")
def test_repair_skips_surgery_when_lock_file_is_unopenable(tmp_path):
    """Behavior: no writable_schema surgery, no forensic backup, DB untouched.

    A full disk is the worst possible moment to start an unsynchronised
    VACUUM on a live shared DB, and it is exactly when the lock file cannot
    be created.
    """
    db_path = tmp_path / "state.db"
    _build_healthy_db(db_path)
    _corrupt_duplicate_fts(db_path)
    assert hermes_state_repair._db_opens_cleanly(db_path) is not None
    before = db_path.read_bytes()

    _make_unopenable(db_path.with_name(db_path.name + ".repair.lock"))

    report = repair_state_db_schema(db_path)

    assert report["repaired"] is False
    assert "repair lock" in (report["error"] or "")
    assert report["backup_path"] is None
    assert not list(tmp_path.glob("state.db.malformed-backup-*"))
    # The damaged image is left byte-identical for the next (authorised) pass.
    assert db_path.read_bytes() == before
