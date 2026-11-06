"""Updater restore must publish snapshot rows without replaying the old WAL."""

import logging
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from hermes_cli.update_cmd import (
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


@pytest.fixture(autouse=True)
def _isolate_database_holders(monkeypatch):
    # These fixtures own all DB connections. Do not scan other users' /proc FDs.
    monkeypatch.setattr("hermes_cli.backup_restore._foreign_db_holder_pids", lambda path: [])


def _row_count(db_path: Path) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    finally:
        conn.close()


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


def test_post_update_preserves_readable_hot_wal_with_bad_main_header(
    live_db_with_hot_wal, snapshot_db, capsys
):
    """A damaged main header is not proof that a readable WAL can be discarded."""
    from hermes_cli.backup import verify_sqlite_integrity
    from hermes_cli.update_cmd import _verify_and_restore_one_state_db

    live = live_db_with_hot_wal
    with live.open("r+b") as stream:
        stream.write(b"\x00" * 16)
    main_before = live.read_bytes()
    wal_before = _sidecar(live, "-wal").read_bytes()

    result = verify_sqlite_integrity(live, check_header=True, run_pragma=True)
    assert result["valid"] is None
    assert result["wal_preserve"] is True
    assert "WAL" in result["message"]

    _verify_and_restore_one_state_db(live.parent, label="test home")
    assert "corrupt" not in capsys.readouterr().out.lower()
    assert live.read_bytes() == main_before
    assert _sidecar(live, "-wal").read_bytes() == wal_before
    assert _row_count(live) == OLD_ROWS
    assert _row_count(live) != SNAPSHOT_ROWS


def test_short_main_with_readable_wal_is_not_auto_restored(
    live_db_with_hot_wal, snapshot_db, capsys
):
    from hermes_cli.backup import verify_sqlite_integrity
    from hermes_cli.update_cmd import _verify_and_restore_one_state_db

    live = live_db_with_hot_wal
    live.write_bytes(live.read_bytes()[:64])
    with closing(sqlite3.connect(live.as_uri() + "?mode=ro", uri=True)) as probe:
        assert probe.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert probe.execute("SELECT count(*) FROM sessions").fetchone()[0] == OLD_ROWS
    main_before = live.read_bytes()
    wal_before = _sidecar(live, "-wal").read_bytes()

    assert verify_sqlite_integrity(live)["valid"] is None
    _verify_and_restore_one_state_db(live.parent, label="test home")
    assert "auto-restored" not in capsys.readouterr().out.lower()
    assert live.read_bytes() == main_before
    assert _sidecar(live, "-wal").read_bytes() == wal_before
    assert _row_count(live) == OLD_ROWS


def test_refused_main_header_read_defers_restore(
    live_db_with_hot_wal, snapshot_db, monkeypatch, capsys
):
    from hermes_cli.backup import verify_sqlite_integrity
    from hermes_cli import sqlite_safe_read
    from hermes_cli.update_cmd import _verify_and_restore_one_state_db

    live = live_db_with_hot_wal
    before = live.read_bytes()
    wal_before = _sidecar(live, "-wal").read_bytes()
    original = sqlite_safe_read.read_header_bytes_preopen
    monkeypatch.setattr(
        sqlite_safe_read, "read_header_bytes_preopen",
        lambda path, **kwargs: None if path == live else original(path, **kwargs),
    )
    result = verify_sqlite_integrity(live)
    assert result["valid"] is None
    assert "header" in result["message"]
    _verify_and_restore_one_state_db(live.parent, label="test home")
    assert "auto-restored" not in capsys.readouterr().out.lower()
    assert live.read_bytes() == before
    assert _sidecar(live, "-wal").read_bytes() == wal_before
    assert _row_count(live) == OLD_ROWS


def test_bad_main_header_with_wal_is_indeterminate_when_probe_is_bounded(
    live_db_with_hot_wal
):
    from hermes_cli.backup import verify_sqlite_integrity

    live = live_db_with_hot_wal
    with live.open("r+b") as stream:
        stream.write(b"\x00" * 16)
    bounded = verify_sqlite_integrity(live, max_bytes=1)
    assert bounded["valid"] is None
    assert bounded["wal_preserve"] is True
    assert "full integrity check skipped" in bounded["message"]
    skipped = verify_sqlite_integrity(live, run_pragma=False)
    assert skipped["valid"] is None
    assert skipped["wal_preserve"] is True


def test_bad_main_header_with_wal_defers_when_sqlite_probe_cannot_open(
    live_db_with_hot_wal, monkeypatch
):
    from hermes_cli import backup

    live = live_db_with_hot_wal
    with live.open("r+b") as stream:
        stream.write(b"\x00" * 16)
    monkeypatch.setattr(
        backup, "_query_ro_sqlite", lambda path, fn: (None, sqlite3.OperationalError("database is locked"))
    )
    result = backup.verify_sqlite_integrity(live)
    assert result["valid"] is None and result["wal_preserve"] is True
    assert "locked" in result["message"]


def test_restore_refuses_wal_dependent_source_with_damaged_main_header(
    live_db_with_hot_wal, snapshot_db
):
    from hermes_cli.backup_restore import _safe_restore_db

    source = live_db_with_hot_wal
    with source.open("r+b") as stream:
        stream.write(b"\x00" * 16)
    dst_before = snapshot_db.read_bytes()
    wal_before = _sidecar(source, "-wal").read_bytes()
    assert _safe_restore_db(source, snapshot_db) is False
    assert snapshot_db.read_bytes() == dst_before
    assert _sidecar(source, "-wal").read_bytes() == wal_before
    assert _row_count(snapshot_db) == SNAPSHOT_ROWS


def test_post_update_warns_to_preserve_recoverable_wal(live_db_with_hot_wal, capsys):
    from hermes_cli.update_cmd import _verify_and_restore_one_state_db

    live = live_db_with_hot_wal
    with live.open("r+b") as stream:
        stream.write(b"\x00" * 16)
    _verify_and_restore_one_state_db(live.parent, label="test home")
    out = capsys.readouterr().out.lower()
    assert "wal" in out and "preserve" in out
    assert "auto-restored" not in out


def test_boot_guard_warns_for_readable_wal_with_damaged_main_header(
    live_db_with_hot_wal, monkeypatch, caplog
):
    from hermes_cli.post_update import step_state_db_guard
    import hermes_constants

    live = live_db_with_hot_wal
    with live.open("r+b") as stream:
        stream.write(b"\x00" * 16)
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: live.parent)
    with caplog.at_level(logging.WARNING):
        result = step_state_db_guard()
    assert result["ok"] is False and result["unverified"] is True
    assert "WAL" in caplog.text and "preserve" in caplog.text.lower()
    assert "verification deferred" in caplog.text
    assert "failed integrity check" not in caplog.text


def test_pre_update_snapshot_check_defers_recoverable_wal(
    live_db_with_hot_wal, snapshot_db, monkeypatch, capsys
):
    from hermes_cli.update_cmd_maint import _verify_state_db_after_snapshot
    from hermes_cli import config

    live = live_db_with_hot_wal
    with live.open("r+b") as stream:
        stream.write(b"\x00" * 16)
    wal_before = _sidecar(live, "-wal").read_bytes()
    monkeypatch.setattr(config, "get_hermes_home", lambda: live.parent)
    _verify_state_db_after_snapshot(snapshot_db.parent.name)
    out = capsys.readouterr().out.lower()
    assert "verification deferred" in out
    assert "integrity check failed" not in out
    assert "will be auto-restored" not in out
    assert _sidecar(live, "-wal").read_bytes() == wal_before


def test_restore_helper_serves_snapshot_rows_over_a_hot_wal(
    live_db_with_hot_wal, snapshot_db
):
    """The shared restore helper is what both update paths call.

    It must restore and verify as one unit: after it returns, the database
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

    # Corrupt only the sessions B-tree; the canonical epoch must remain readable.
    _make_valid_snapshot(sibling_home, "20260901-pre-update", 25)
    state = sibling_home / "state.db"
    _make_valid_db(state, 3)
    with sqlite3.connect(state) as conn:
        page = conn.execute("SELECT rootpage FROM sqlite_master WHERE name='sessions'").fetchone()[0]
        page_size = conn.execute("PRAGMA page_size").fetchone()[0]
    conn.close()
    with state.open("r+b") as stream:
        stream.seek((page - 1) * page_size)
        stream.write(b"\x00")

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
    assert "✓ Auto-restored from snapshot" in out


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
