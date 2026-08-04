"""Auto-restore of state.db must not inherit the corrupt database's WAL.

The post-update integrity guard added for #68474 restores ``state.db`` from a
pre-update quick snapshot with a plain ``shutil.copy2``. The snapshot image is
produced by ``backup._safe_copy_db`` via ``sqlite3.backup()``, so it is already
checkpointed and owns no WAL — which is why ``backup._EXCLUDED_SUFFIXES``
deliberately refuses to ship ``-wal`` / ``-shm`` / ``-journal`` inside a
snapshot.

Copying that image over the live path replaces only the main database file. A
``state.db-wal`` left behind by the *old* database — a crashed writer, or a
second Hermes holder the updater's drain did not stop — survives the copy and is
replayed over the fresh image on the next open. The restored file then passes
``PRAGMA integrity_check`` while serving the discarded database's contents, so
the CLI prints "✓ Auto-restored from snapshot" over data the user has lost.

These tests exercise REAL SQLite files, in WAL mode, with a genuinely hot
sidecar.
"""

import ast
import shutil
import sqlite3
from pathlib import Path

import pytest

from hermes_cli.update_cmd import _clear_stale_sqlite_sidecars

OLD_ROWS = 201
SNAPSHOT_ROWS = 400


def _sidecar(db_path: Path, suffix: str) -> Path:
    return db_path.with_name(db_path.name + suffix)


@pytest.fixture()
def live_db_with_hot_wal(tmp_path):
    """A WAL-mode database whose committed rows live in an UNCHECKPOINTED
    ``-wal``, exactly as a force-killed writer leaves them on disk.

    A clean ``close()`` checkpoints and unlinks the WAL, so the on-disk trio is
    copied aside while the connection is still open and put back afterwards.
    """
    live = tmp_path / "state.db"
    holding = tmp_path / "_killed_writer_state"
    holding.mkdir()

    conn = sqlite3.connect(live)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA wal_autocheckpoint=0")
    conn.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY, name TEXT)")
    conn.executemany(
        "INSERT INTO sessions (name) VALUES (?)",
        [(f"old{i}",) for i in range(OLD_ROWS)],
    )
    conn.commit()
    for suffix in ("", "-wal", "-shm"):
        src = _sidecar(live, suffix)
        if src.exists():
            shutil.copy2(src, holding / (live.name + suffix))
    conn.close()

    for suffix in ("", "-wal", "-shm"):
        held = holding / (live.name + suffix)
        if held.exists():
            shutil.copy2(held, _sidecar(live, suffix))

    assert _sidecar(live, "-wal").exists(), "fixture failed to leave a hot WAL"
    return live


@pytest.fixture()
def snapshot_db(tmp_path):
    """A consistent, checkpointed image owning no WAL — what
    ``backup._safe_copy_db`` produces through ``sqlite3.backup()``."""
    source_path = tmp_path / "_snapshot_source.db"
    snapshot = tmp_path / "state-snapshots" / "20260804-pre-update" / "state.db"
    snapshot.parent.mkdir(parents=True)

    source = sqlite3.connect(source_path)
    source.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY, name TEXT)")
    source.executemany(
        "INSERT INTO sessions (name) VALUES (?)",
        [(f"snap{i}",) for i in range(SNAPSHOT_ROWS)],
    )
    source.commit()
    destination = sqlite3.connect(snapshot)
    source.backup(destination)
    destination.close()
    source.close()
    source_path.unlink()

    assert not _sidecar(snapshot, "-wal").exists()
    return snapshot


def _row_count(db_path: Path) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    finally:
        conn.close()


def test_restore_over_hot_wal_serves_snapshot_rows(live_db_with_hot_wal, snapshot_db):
    """The restored database must contain the SNAPSHOT's rows, not the WAL's.

    Without clearing the sidecars the copy silently loses every restored row:
    SQLite replays the old WAL and serves the discarded database instead.
    """
    _clear_stale_sqlite_sidecars(live_db_with_hot_wal)
    shutil.copy2(snapshot_db, live_db_with_hot_wal)

    assert _row_count(live_db_with_hot_wal) == SNAPSHOT_ROWS


def test_stale_wal_is_not_left_beside_the_restored_file(
    live_db_with_hot_wal, snapshot_db
):
    """No sidecar from the discarded database may survive the restore."""
    _clear_stale_sqlite_sidecars(live_db_with_hot_wal)
    shutil.copy2(snapshot_db, live_db_with_hot_wal)

    for suffix in ("-wal", "-shm", "-journal"):
        assert not _sidecar(live_db_with_hot_wal, suffix).exists()


def test_torn_restore_is_what_the_guard_prevents(live_db_with_hot_wal, snapshot_db):
    """Pin the SQLite behaviour that makes the guard necessary.

    Copying the snapshot over a database that still owns a hot WAL yields a file
    that reports ``integrity_check`` clean — so the CLI's ``_restored_ok`` test
    passes and it prints success — while serving the OLD row set.
    """
    from hermes_cli.backup import verify_sqlite_integrity

    shutil.copy2(snapshot_db, live_db_with_hot_wal)  # no sidecar clearing

    assert (
        verify_sqlite_integrity(
            live_db_with_hot_wal, check_header=True, run_pragma=True
        ).get("valid")
        is True
    )
    assert _row_count(live_db_with_hot_wal) == OLD_ROWS
    assert _row_count(live_db_with_hot_wal) != SNAPSHOT_ROWS


def test_clear_is_a_noop_when_no_sidecars_exist(tmp_path):
    """A snapshot-clean destination must not raise (``missing_ok``)."""
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE t (a INTEGER)")
    conn.commit()
    conn.close()

    _clear_stale_sqlite_sidecars(db_path)

    assert db_path.exists()


def test_clear_removes_every_sidecar_suffix_and_spares_the_database(tmp_path):
    db_path = tmp_path / "state.db"
    db_path.write_bytes(b"main-db")
    for suffix in ("-wal", "-shm", "-journal"):
        _sidecar(db_path, suffix).write_bytes(b"stale")

    _clear_stale_sqlite_sidecars(db_path)

    for suffix in ("-wal", "-shm", "-journal"):
        assert not _sidecar(db_path, suffix).exists()
    assert db_path.read_bytes() == b"main-db"


def test_both_auto_restore_call_sites_clear_sidecars_first():
    """Bind the fix to the production call sites, not just the helper.

    Both auto-restore blocks (the ZIP-update path and the git-pull path) copy
    the snapshot with ``_shutil.copy2(_snap_state, _state_path)``. Each one must
    be immediately preceded by the sidecar clear, or the helper is dead code.
    """
    source = Path(__file__).resolve().parents[2] / "hermes_cli" / "update_cmd.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))

    guarded = 0
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        for previous, statement in zip(body, body[1:]):
            if not _is_snapshot_copy(statement):
                continue
            assert _is_sidecar_clear(previous), (
                "auto-restore at line "
                f"{statement.lineno} copies the snapshot without clearing the "
                "destination's stale SQLite sidecars first"
            )
            guarded += 1

    assert guarded == 2, f"expected 2 auto-restore call sites, found {guarded}"


def _is_snapshot_copy(statement) -> bool:
    call = _expression_call(statement)
    if call is None:
        return False
    func = call.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "copy2"
        and isinstance(func.value, ast.Name)
        and func.value.id == "_shutil"
        and bool(call.args)
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "_snap_state"
    )


def _is_sidecar_clear(statement) -> bool:
    call = _expression_call(statement)
    if call is None:
        return False
    return (
        isinstance(call.func, ast.Name)
        and call.func.id == "_clear_stale_sqlite_sidecars"
    )


def _expression_call(statement):
    if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
        return statement.value
    return None
