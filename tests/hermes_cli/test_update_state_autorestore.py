"""Updater restore must publish snapshot rows without replaying the old WAL."""

import os
import shutil
import sqlite3
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
    from hermes_cli.backup_integrity import verify_sqlite_integrity

    shutil.copy2(snapshot_db, live_db_with_hot_wal)  # no sidecar clearing

    assert (
        verify_sqlite_integrity(
            live_db_with_hot_wal, check_header=True, run_pragma=True
        ).get("valid")
        is True
    )
    assert _row_count(live_db_with_hot_wal) == OLD_ROWS
    assert _row_count(live_db_with_hot_wal) != SNAPSHOT_ROWS


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


def _wreck_sessions_btree(state: Path) -> None:
    """Corrupt the sessions B-tree so integrity fails but the header/epoch stay readable."""
    with sqlite3.connect(state) as conn:
        page = conn.execute("SELECT rootpage FROM sqlite_master WHERE name='sessions'").fetchone()[0]
        page_size = conn.execute("PRAGMA page_size").fetchone()[0]
    with state.open("r+b") as stream:
        stream.seek((page - 1) * page_size)
        stream.write(b"\x00")



def test_post_update_guard_covers_sibling_profiles(tmp_path, monkeypatch, capsys):
    """#97994: the guard must verify + auto-restore EVERY profile's state.db,
    not just the root home's. Pre-update snapshots already cover siblings
    (#66140); the guard was the missing half. Restores only this update's
    snapshot (#147938), never an older sibling directory."""
    from hermes_cli import backup, update_cmd

    root_home = tmp_path / "default-home"
    root_home.mkdir()
    sibling_home = tmp_path / "profiles" / "work"
    sibling_home.mkdir(parents=True)

    # Root DB: valid, with its own snapshot — must be left untouched.
    _make_valid_db(root_home / "state.db", 10)
    root_before = (root_home / "state.db").read_bytes()

    # Capture this update's snapshot, then add a decoy older/newer directory.
    state = sibling_home / "state.db"
    _make_valid_db(state, 25)
    snapshot_id = backup.create_quick_snapshot(label="pre-update", hermes_home=sibling_home)
    assert snapshot_id
    _make_valid_snapshot(sibling_home, "99999999-other-snapshot", 50)
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

    update_cmd._verify_and_restore_state_dbs_post_update(sibling_snapshots={"work": snapshot_id})

    # Sibling restored from THIS update's snapshot (25 rows), not the decoy (50).
    assert _row_count(sibling_home / "state.db") == 25
    # Root DB byte-identical — untouched.
    assert (root_home / "state.db").read_bytes() == root_before
    # Operator-visible restore message mentions the profile.
    out = capsys.readouterr().out
    assert "profile work" in out
    assert "✓ Auto-restored from snapshot" in out
    assert snapshot_id in out


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
    assert "verification failed" not in out
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
    assert "verification failed" in out
    # Still corrupt (no this-update snapshot) — but the guard completed cleanly.
    assert (sibling_home / "state.db").read_bytes() == b"\x00" * 4096


# ── #147938 combined guard: lock ≠ corrupt, freshness, honest restore label ──


def test_unreadable_header_on_healthy_db_does_not_restore(tmp_path, monkeypatch, capsys):
    """Defect 1: a failed header read must fall through to PRAGMA, not accuse corruption."""
    from hermes_cli import backup, sqlite_safe_read, update_cmd

    home = tmp_path / "home"
    home.mkdir()
    state = home / "state.db"
    _make_valid_db(state, 10)
    # Stale snapshot that must never be selected even if the guard misfires.
    _make_valid_snapshot(home, "20260715-pre-update", 2)
    before = state.read_bytes()
    real_read = sqlite_safe_read.read_header_bytes_preopen

    def refuse_live(path, **kwargs):
        if Path(path) == state:
            return None
        return real_read(path, **kwargs)

    monkeypatch.setattr(sqlite_safe_read, "read_header_bytes_preopen", refuse_live)
    monkeypatch.setattr(update_cmd, "get_hermes_home", lambda: home)
    monkeypatch.setattr(backup, "_sibling_profile_homes", lambda _h: [])

    # Even with a recorded snapshot id, a healthy DB must not be restored.
    snap_id = backup.create_quick_snapshot(label="pre-update", hermes_home=home)
    assert snap_id
    # Recreate the richer live DB after the snapshot so a mistaken restore is visible.
    state.unlink()
    _make_valid_db(state, 10)
    before = state.read_bytes()
    update_cmd._verify_and_restore_state_dbs_post_update(snap_id)

    assert state.read_bytes() == before
    out = capsys.readouterr().out
    assert "verification failed" not in out
    assert "corrupted" not in out
    assert "Auto-restored" not in out


def test_no_this_update_snapshot_leaves_corrupt_db_unchanged(tmp_path, monkeypatch, capsys):
    """Defect 2: without this update's snapshot, never fall back to a months-old image."""
    from hermes_cli import backup, update_cmd

    home = tmp_path / "home"
    home.mkdir()
    state = home / "state.db"
    _make_valid_db(state, 10)
    stale = backup.create_quick_snapshot(label="pre-update", hermes_home=home)
    assert stale
    # Destroy the live header so verify returns a definite False.
    state.write_bytes(b"\x00" * max(state.stat().st_size, 4096))
    damaged = state.read_bytes()
    monkeypatch.setattr(update_cmd, "get_hermes_home", lambda: home)
    monkeypatch.setattr(backup, "_sibling_profile_homes", lambda _h: [])

    update_cmd._verify_and_restore_state_dbs_post_update(None)

    assert state.read_bytes() == damaged
    out = capsys.readouterr().out
    assert "left unchanged" in out
    assert "Auto-restored" not in out


def test_this_update_snapshot_restores_genuine_corruption(tmp_path, monkeypatch, capsys):
    """Genuine corruption still restores — but only from this update's snapshot."""
    from hermes_cli import backup, update_cmd

    home = tmp_path / "home"
    home.mkdir()
    state = home / "state.db"
    _make_valid_db(state, 10)
    snap_id = backup.create_quick_snapshot(label="pre-update", hermes_home=home)
    assert snap_id
    _make_valid_snapshot(home, "20260715-old", 2)  # decoy
    # Keep the canonical epoch readable so _safe_restore_db will accept the restore.
    _wreck_sessions_btree(state)
    monkeypatch.setattr(update_cmd, "get_hermes_home", lambda: home)
    monkeypatch.setattr(backup, "_sibling_profile_homes", lambda _h: [])

    update_cmd._verify_and_restore_state_dbs_post_update(snap_id)

    assert _row_count(state) == 10
    out = capsys.readouterr().out
    assert f"Auto-restored from snapshot {snap_id}" in out


def test_post_restore_success_despite_unreadable_header(tmp_path, monkeypatch, capsys):
    """Defect 3: a completed restore must not be reported as 'did not complete'."""
    from hermes_cli import backup, sqlite_safe_read, update_cmd
    from hermes_cli.update_cmd_maint import _restore_state_db_from_snapshot

    home = tmp_path / "home"
    home.mkdir()
    state = home / "state.db"
    _make_valid_db(state, 3)
    snap_id = backup.create_quick_snapshot(label="pre-update", hermes_home=home)
    snap_state = home / "state-snapshots" / snap_id / "state.db"
    assert snap_state.is_file()
    # Destination stays a real DB (epoch readable) so restore authority admits the write;
    # only the post-restore header probe is refused.
    _make_valid_db(home / "live-extra.db", 1)
    # Grow live slightly so a successful restore from the 3-row snapshot is observable.
    # (epoch remains readable — do not zero the file.)
    real_read = sqlite_safe_read.read_header_bytes_preopen

    def refuse_live(path, **kwargs):
        if Path(path) == state:
            return None
        return real_read(path, **kwargs)

    monkeypatch.setattr(sqlite_safe_read, "read_header_bytes_preopen", refuse_live)

    assert _restore_state_db_from_snapshot(state, snap_state) is True
    assert _row_count(state) == 3


@pytest.mark.parametrize("max_bytes", [1, 2 << 30])
def test_integrity_probe_treats_an_exclusive_lock_as_indeterminate(tmp_path, max_bytes):
    """A busy schema probe or full PRAGMA cannot establish corruption."""
    from hermes_cli.backup_integrity import verify_sqlite_integrity

    db = tmp_path / "state.db"
    _make_valid_db(db, 3)
    holder = sqlite3.connect(db)
    try:
        holder.execute("BEGIN EXCLUSIVE")
        result = verify_sqlite_integrity(db, run_pragma=True, max_bytes=max_bytes)
    finally:
        holder.rollback()
        holder.close()

    assert result["valid"] is None
    assert verify_sqlite_integrity(db)["valid"] is True


def test_update_and_boot_guards_defer_a_locked_healthy_db(tmp_path, monkeypatch, capsys):
    """Update must explain a deferred check without entering the restore path."""
    from hermes_cli import update_cmd
    from hermes_cli.post_update import step_state_db_guard
    from hermes_cli.update_cmd_maint import _verify_state_db_after_snapshot

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    db = home / "state.db"
    _make_valid_db(db, 3)
    # Provide a this-update snapshot id so a mistaken restore path would fire.
    from hermes_cli import backup
    snap_id = backup.create_quick_snapshot(label="pre-update", hermes_home=home)
    assert snap_id
    before = db.read_bytes()
    monkeypatch.setattr(
        "hermes_cli.update_cmd_maint._restore_state_db_from_snapshot",
        lambda *args: pytest.fail("a locked database must not be restored"),
    )

    holder = sqlite3.connect(db)
    try:
        holder.execute("BEGIN EXCLUSIVE")
        _verify_state_db_after_snapshot(snap_id)
        update_cmd._verify_and_restore_one_state_db(home, label="default home", snapshot_id=snap_id)
        assert step_state_db_guard() == {"ok": True, "skipped": "integrity-indeterminate"}
    finally:
        holder.rollback()
        holder.close()

    out = capsys.readouterr().out
    assert "verification deferred" in out
    assert "corrupted" not in out
    assert "integrity check FAILED" not in out
    assert db.read_bytes() == before


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0, reason="needs a non-root chmod 000")
def test_unreadable_header_and_unopenable_db_is_indeterminate(tmp_path, monkeypatch):
    """A Windows sharing lock can refuse BOTH the raw read and SQLite's open (CANTOPEN):
    with no NOTADB/CORRUPT evidence the verdict is None, never "corrupt" (#147938)."""
    from hermes_cli import sqlite_safe_read
    from hermes_cli.backup_integrity import verify_sqlite_integrity

    db = tmp_path / "state.db"
    _make_valid_db(db, 3)
    monkeypatch.setattr(sqlite_safe_read, "read_header_bytes_preopen", lambda *a, **k: None)
    db.chmod(0)
    try:
        result = verify_sqlite_integrity(db, check_header=True, run_pragma=True)
    finally:
        db.chmod(0o644)

    assert result["valid"] is None, result


def test_short_header_read_is_indeterminate(tmp_path, monkeypatch):
    """A torn 1–15-byte header read must not accuse corruption (sd2389 / #150275).

    Fall through like OSError/lock: with no PRAGMA asked, the verdict is None.
    """
    from hermes_cli import sqlite_safe_read
    from hermes_cli.backup_integrity import _SQLITE_HEADER, verify_sqlite_integrity

    db = tmp_path / "state.db"
    _make_valid_db(db, 3)
    short = _SQLITE_HEADER[:15]
    assert len(short) < len(_SQLITE_HEADER)
    monkeypatch.setattr(
        sqlite_safe_read, "read_header_bytes_preopen", lambda *a, **k: short
    )

    result = verify_sqlite_integrity(db, check_header=True, run_pragma=False)

    assert result["valid"] is None, result
    assert "incomplete header read" in result["message"]


def test_short_header_read_falls_through_to_pragma_on_healthy_db(tmp_path, monkeypatch):
    """A short header read on a healthy DB still passes via the PRAGMA probe."""
    from hermes_cli import sqlite_safe_read
    from hermes_cli.backup_integrity import _SQLITE_HEADER, verify_sqlite_integrity

    db = tmp_path / "state.db"
    _make_valid_db(db, 3)
    monkeypatch.setattr(
        sqlite_safe_read,
        "read_header_bytes_preopen",
        lambda *a, **k: _SQLITE_HEADER[:8],
    )

    result = verify_sqlite_integrity(db, check_header=True, run_pragma=True)

    assert result["valid"] is True, result


def test_full_length_wrong_magic_is_corrupt(tmp_path, monkeypatch):
    """A full-length non-SQLite header remains a definite False."""
    from hermes_cli import sqlite_safe_read
    from hermes_cli.backup_integrity import _SQLITE_HEADER, verify_sqlite_integrity

    db = tmp_path / "state.db"
    _make_valid_db(db, 3)
    wrong = b"not a database!!"
    assert len(wrong) == len(_SQLITE_HEADER)
    assert wrong != _SQLITE_HEADER
    monkeypatch.setattr(
        sqlite_safe_read, "read_header_bytes_preopen", lambda *a, **k: wrong
    )

    result = verify_sqlite_integrity(db, check_header=True, run_pragma=False)

    assert result["valid"] is False, result
    assert "missing SQLite header magic" in result["message"]


def test_restore_helper_defers_when_snapshot_reverify_is_indeterminate(
    tmp_path, monkeypatch, capsys
):
    """Windows flip-flop: a passing check then a transient None on the restore
    helper's second verify must hold back — never call the snapshot corrupt."""
    from hermes_cli import backup_integrity
    from hermes_cli.update_cmd_maint import _restore_state_db_from_snapshot

    state = tmp_path / "state.db"
    snap = tmp_path / "snap.db"
    _make_valid_db(state, 10)
    _make_valid_db(snap, 3)
    before = state.read_bytes()
    real = backup_integrity.verify_sqlite_integrity

    def flip_flop(path, **kwargs):
        if Path(path) == snap:
            return {
                "valid": None,
                "message": "header check skipped: file temporarily unreadable",
                "size": snap.stat().st_size,
            }
        return real(path, **kwargs)

    monkeypatch.setattr(backup_integrity, "verify_sqlite_integrity", flip_flop)

    assert _restore_state_db_from_snapshot(state, snap) is False

    assert state.read_bytes() == before
    out = capsys.readouterr().out
    assert "verification deferred" in out
    assert "corrupted" not in out
    assert "failed integrity" not in out


def test_caller_defers_when_snapshot_verify_is_indeterminate(
    tmp_path, monkeypatch, capsys
):
    """The post-update guard must not label an indeterminate snapshot check as failed."""
    from hermes_cli import backup, backup_integrity, update_cmd

    home = tmp_path / "home"
    home.mkdir()
    state = home / "state.db"
    _make_valid_db(state, 10)
    snap_id = backup.create_quick_snapshot(label="pre-update", hermes_home=home)
    assert snap_id
    snap_state = home / "state-snapshots" / snap_id / "state.db"
    _wreck_sessions_btree(state)
    damaged = state.read_bytes()
    real = backup_integrity.verify_sqlite_integrity

    def snap_indeterminate(path, **kwargs):
        if Path(path) == snap_state:
            return {
                "valid": None,
                "message": "header check skipped: file temporarily unreadable",
                "size": snap_state.stat().st_size,
            }
        return real(path, **kwargs)

    monkeypatch.setattr(backup_integrity, "verify_sqlite_integrity", snap_indeterminate)
    monkeypatch.setattr(update_cmd, "get_hermes_home", lambda: home)
    monkeypatch.setattr(backup, "_sibling_profile_homes", lambda _h: [])
    monkeypatch.setattr(
        "hermes_cli.update_cmd_maint._restore_state_db_from_snapshot",
        lambda *a, **k: pytest.fail("indeterminate snapshot must not enter restore"),
    )

    update_cmd._verify_and_restore_state_dbs_post_update(snap_id)

    assert state.read_bytes() == damaged
    out = capsys.readouterr().out
    assert "verification deferred" in out
    assert "failed state.db verification" not in out
    assert "Auto-restored" not in out

def test_after_snapshot_indeterminate_snapshot_is_not_reported_valid(
    tmp_path, monkeypatch, capsys
):
    """#147938: an unreadable recovery snapshot is not a verified recovery point."""
    from hermes_cli import backup, backup_integrity
    from hermes_cli.update_cmd_maint import _verify_state_db_after_snapshot

    home = tmp_path / "home"
    home.mkdir()
    state = home / "state.db"
    _make_valid_db(state, 3)
    snapshot_id = backup.create_quick_snapshot(label="pre-update", hermes_home=home)
    assert snapshot_id
    snapshot = home / "state-snapshots" / snapshot_id / "state.db"
    assert snapshot.is_file()

    # The live DB is genuinely damaged, but the saved snapshot's second check
    # cannot decide (e.g. a transient Windows sharing violation).
    _wreck_sessions_btree(state)
    real_verify = backup_integrity.verify_sqlite_integrity

    def indeterminate_snapshot(path, **kwargs):
        if Path(path) == snapshot:
            return {
                "valid": None,
                "message": "temporary sharing violation",
                "size": snapshot.stat().st_size,
            }
        return real_verify(path, **kwargs)

    monkeypatch.setattr(backup_integrity, "verify_sqlite_integrity", indeterminate_snapshot)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: home)

    _verify_state_db_after_snapshot(snapshot_id)

    out = capsys.readouterr().out
    assert "state.db integrity check FAILED after snapshot" in out
    assert "Snapshot copy verification deferred: temporary sharing violation" in out
    assert "Snapshot copy is valid" not in out
    assert "will be auto-restored" not in out
    assert "Snapshot copy ALSO failed integrity" not in out


def test_after_snapshot_valid_snapshot_reports_auto_restore(
    tmp_path, monkeypatch, capsys
):
    """Corrupt live DB + verified snapshot still promises auto-restore."""
    from hermes_cli import backup, backup_integrity
    from hermes_cli.update_cmd_maint import _verify_state_db_after_snapshot

    home = tmp_path / "home"
    home.mkdir()
    state = home / "state.db"
    _make_valid_db(state, 3)
    snapshot_id = backup.create_quick_snapshot(label="pre-update", hermes_home=home)
    assert snapshot_id
    snapshot = home / "state-snapshots" / snapshot_id / "state.db"
    assert snapshot.is_file()

    _wreck_sessions_btree(state)
    # Leave snapshot verification real so valid=True.
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: home)

    _verify_state_db_after_snapshot(snapshot_id)

    out = capsys.readouterr().out
    assert "state.db integrity check FAILED after snapshot" in out
    assert "Snapshot copy is valid — continuing update." in out
    assert "will be auto-restored" in out
    assert "verification deferred" not in out
    assert "ALSO failed integrity" not in out
    # Pre-update status path never restores; live DB stays wrecked.
    assert backup_integrity.verify_sqlite_integrity(state)["valid"] is False


def test_after_snapshot_corrupt_snapshot_reports_also_failed(
    tmp_path, monkeypatch, capsys
):
    """Corrupt live DB + corrupt snapshot must not claim a recovery point."""
    from hermes_cli import backup, backup_integrity
    from hermes_cli.update_cmd_maint import _verify_state_db_after_snapshot

    home = tmp_path / "home"
    home.mkdir()
    state = home / "state.db"
    _make_valid_db(state, 3)
    snapshot_id = backup.create_quick_snapshot(label="pre-update", hermes_home=home)
    assert snapshot_id
    snapshot = home / "state-snapshots" / snapshot_id / "state.db"
    assert snapshot.is_file()

    _wreck_sessions_btree(state)
    _wreck_sessions_btree(snapshot)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: home)

    _verify_state_db_after_snapshot(snapshot_id)

    out = capsys.readouterr().out
    assert "state.db integrity check FAILED after snapshot" in out
    assert "Snapshot copy ALSO failed integrity" in out
    assert "Snapshot copy is valid" not in out
    assert "will be auto-restored" not in out
    assert "verification deferred" not in out
