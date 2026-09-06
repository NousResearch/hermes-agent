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

import shutil
import sqlite3
from pathlib import Path

import pytest

from hermes_cli.update_cmd import (
    _clear_stale_sqlite_sidecars,
    _restore_state_db_from_snapshot,
)

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


def test_restore_helper_serves_snapshot_rows_over_a_hot_wal(
    live_db_with_hot_wal, snapshot_db
):
    """The shared restore helper is what both update paths call.

    It must clear, copy and verify as one unit: after it returns, the database
    has to hold the SNAPSHOT's rows even though the destination still owned a
    hot WAL from the corrupt database.
    """
    assert _restore_state_db_from_snapshot(live_db_with_hot_wal, snapshot_db) is True

    assert _row_count(live_db_with_hot_wal) == SNAPSHOT_ROWS
    for suffix in ("-wal", "-shm", "-journal"):
        assert not _sidecar(live_db_with_hot_wal, suffix).exists()


def test_restore_helper_reports_failure_when_the_restored_copy_is_corrupt(tmp_path):
    """A snapshot that does not survive the copy must return False, so the
    caller prints the failure branch instead of claiming success."""
    state_path = tmp_path / "state.db"
    state_path.write_bytes(b"whatever")
    bad_snapshot = tmp_path / "bad-snapshot.db"
    bad_snapshot.write_bytes(b"\x00" * 4096)

    assert _restore_state_db_from_snapshot(state_path, bad_snapshot) is False


def test_restore_helper_propagates_copy_errors(tmp_path):
    """A missing snapshot raises OSError, which both call sites already catch
    and report as 'Auto-restore file copy failed'."""
    state_path = tmp_path / "state.db"
    state_path.write_bytes(b"whatever")

    with pytest.raises(OSError):
        _restore_state_db_from_snapshot(state_path, tmp_path / "does-not-exist.db")


# ── Multi-profile coverage (#97994) ─────────────────────────────────────


def _make_valid_db(path: Path, rows: int) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY, name TEXT)")
    conn.executemany(
        "INSERT INTO sessions (name) VALUES (?)",
        [(str(i),) for i in range(rows)],
    )
    conn.commit()
    conn.close()


def _make_valid_snapshot(home: Path, snap_id: str, rows: int) -> None:
    snap_dir = home / "state-snapshots" / snap_id
    snap_dir.mkdir(parents=True)
    _make_valid_db(snap_dir / "state.db", rows)


def test_post_update_guard_covers_sibling_profiles(tmp_path, monkeypatch, capsys):
    """#97994: the guard must verify + auto-restore EVERY profile's state.db,
    not just the root home's. Pre-update snapshots already cover siblings
    (#66140); the guard was the missing half."""
    from hermes_cli import update_cmd
    from hermes_cli.backup import _sibling_profile_homes

    root_home = tmp_path / "default-home"
    root_home.mkdir()
    sibling_home = tmp_path / "profiles" / "work"
    sibling_home.mkdir(parents=True)

    # Root DB: valid, with its own snapshot — must be left untouched.
    _make_valid_db(root_home / "state.db", 10)
    root_before = (root_home / "state.db").read_bytes()

    # Sibling: live DB corrupted post-update (the #68474 zeroed signature),
    # with its own VALID pre-update snapshot under its own snapshots dir.
    _make_valid_snapshot(sibling_home, "20260901-pre-update", 25)
    (sibling_home / "state.db").write_bytes(b"\x00" * 4096)

    monkeypatch.setattr(update_cmd, "get_hermes_home", lambda: root_home)
    monkeypatch.setattr(
        "hermes_cli.backup._sibling_profile_homes",
        lambda invoking_home: [("work", sibling_home)],
    )

    update_cmd._verify_and_restore_state_dbs_post_update()

    # Sibling restored from ITS snapshot (snap rows, not zeroed bytes).
    assert _row_count(sibling_home / "state.db") == 25
    # Root DB byte-identical — untouched.
    assert (root_home / "state.db").read_bytes() == root_before
    # Operator-visible restore message mentions the profile.
    out = capsys.readouterr().out
    assert "profile work" in out


def test_post_update_guard_leaves_valid_sibling_dbs_alone(tmp_path, monkeypatch, capsys):
    """A healthy sibling profile must not be touched — the guard only acts
    on corruption."""
    from hermes_cli import update_cmd

    root_home = tmp_path / "default-home"
    root_home.mkdir()
    sibling_home = tmp_path / "profiles" / "work"
    sibling_home.mkdir(parents=True)

    _make_valid_db(root_home / "state.db", 10)
    _make_valid_db(sibling_home / "state.db", 7)
    sibling_before = (sibling_home / "state.db").read_bytes()

    monkeypatch.setattr(update_cmd, "get_hermes_home", lambda: root_home)
    monkeypatch.setattr(
        "hermes_cli.backup._sibling_profile_homes",
        lambda invoking_home: [("work", sibling_home)],
    )

    update_cmd._verify_and_restore_state_dbs_post_update()

    assert (sibling_home / "state.db").read_bytes() == sibling_before
    out = capsys.readouterr().out
    assert "corrupted" not in out


def test_post_update_guard_survives_missing_sibling_snapshot(tmp_path, monkeypatch, capsys):
    """Corrupt sibling with NO snapshot: guard must report and continue,
    never raise into the update tail."""
    from hermes_cli import update_cmd

    root_home = tmp_path / "default-home"
    root_home.mkdir()
    sibling_home = tmp_path / "profiles" / "work"
    sibling_home.mkdir(parents=True)

    _make_valid_db(root_home / "state.db", 10)
    (sibling_home / "state.db").write_bytes(b"\x00" * 4096)

    monkeypatch.setattr(update_cmd, "get_hermes_home", lambda: root_home)
    monkeypatch.setattr(
        "hermes_cli.backup._sibling_profile_homes",
        lambda invoking_home: [("work", sibling_home)],
    )

    # Must not raise even though no snapshot exists to restore from.
    update_cmd._verify_and_restore_state_dbs_post_update()

    out = capsys.readouterr().out
    assert "corrupted" in out
    # Still corrupt (no snapshot) — but the guard completed cleanly.
    assert (sibling_home / "state.db").read_bytes() == b"\x00" * 4096
