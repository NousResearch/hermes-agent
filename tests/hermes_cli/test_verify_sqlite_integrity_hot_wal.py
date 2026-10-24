"""#140053: a mismatched main header must not authorize restore while a hot WAL is recoverable.

The post-update guard's corruption verdict comes from ``verify_sqlite_integrity``, whose
header branch used to return before SQLite itself was ever consulted. A crashed writer can
leave every committed page (page 1 included) uncheckpointed in the WAL with a damaged main
header; a ``mode=ro`` open recovers those rows, so the header alone must not authorize the
snapshot restore that would discard them.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

from hermes_cli.backup import _quick_snapshot_root, verify_sqlite_integrity


def _crashed_writer_fixture(dst: Path, row: str) -> None:
    """Copy a WAL-mode db/-wal/-shm trio while its writer is still open, then close it.

    The copied trio keeps every committed row uncheckpointed in the WAL — the on-disk state
    a crashed writer leaves behind. Callers damage the copy's main header afterwards.
    """
    src = dst.with_name("writer-source.db")
    conn = sqlite3.connect(src)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE t(x)")
        conn.execute("INSERT INTO t VALUES (?)", (row,))
        conn.commit()
        for suffix in ("", "-wal", "-shm"):
            shutil.copy2(
                src.with_name(src.name + suffix), dst.with_name(dst.name + suffix)
            )
    finally:
        conn.close()
        for suffix in ("", "-wal", "-shm"):
            src.with_name(src.name + suffix).unlink(missing_ok=True)


def _zero_main_header(db: Path) -> None:
    with open(db, "r+b") as handle:
        handle.write(b"\x00" * 16)


def test_mismatched_header_with_recoverable_hot_wal_is_valid(tmp_path):
    db = tmp_path / "state.db"
    _crashed_writer_fixture(db, "committed-row")
    _zero_main_header(db)

    result = verify_sqlite_integrity(db, check_header=True, run_pragma=True)

    assert result["valid"] is True
    assert "WAL-aware probe" in result["message"]


def test_mismatched_header_without_wal_stays_corrupt(tmp_path):
    db = tmp_path / "state.db"
    _crashed_writer_fixture(db, "committed-row")
    _zero_main_header(db)
    for suffix in ("-wal", "-shm"):
        db.with_name(db.name + suffix).unlink()

    result = verify_sqlite_integrity(db, check_header=True, run_pragma=True)

    assert result["valid"] is False
    assert "missing SQLite header magic" in result["message"]


def test_mismatched_header_with_unrecoverable_wal_stays_corrupt(tmp_path):
    # fail-closed: a present-but-unrecoverable WAL must not smuggle a pass through the probe.
    db = tmp_path / "state.db"
    _crashed_writer_fixture(db, "committed-row")
    _zero_main_header(db)
    db.with_name(db.name + "-wal").write_bytes(b"not-a-wal" * 16)

    result = verify_sqlite_integrity(db, check_header=True, run_pragma=True)

    assert result["valid"] is False


def test_run_pragma_false_keeps_the_header_short_circuit(tmp_path):
    # Callers that opted out of the probe must not get the WAL override for free.
    db = tmp_path / "state.db"
    _crashed_writer_fixture(db, "committed-row")
    _zero_main_header(db)

    result = verify_sqlite_integrity(db, check_header=True, run_pragma=False)

    assert result["valid"] is False
    assert "missing SQLite header magic" in result["message"]


def test_post_update_guard_leaves_readable_wal_database_alone(tmp_path, capsys):
    from hermes_cli.update_cmd_maint import _verify_and_restore_one_state_db

    db = tmp_path / "state.db"
    _crashed_writer_fixture(db, "newer-row")
    _zero_main_header(db)
    # An older pre-update snapshot exists and would otherwise be copied over the live DB.
    snap = _quick_snapshot_root(tmp_path) / "20260101T000000Z"
    snap.mkdir(parents=True)
    plain = sqlite3.connect(snap / "state.db")
    try:
        plain.execute("CREATE TABLE t(x)")
        plain.execute("INSERT INTO t VALUES ('older-row')")
        plain.commit()
    finally:
        plain.close()

    _verify_and_restore_one_state_db(tmp_path, label="test home")

    out = capsys.readouterr().out
    assert "Auto-restored" not in out
    ro = sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True)
    try:
        assert ro.execute("SELECT x FROM t").fetchall() == [("newer-row",)]
    finally:
        ro.close()
