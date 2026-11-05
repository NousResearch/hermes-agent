"""``BackupAdmin`` against the legacy backup, restore and downgrade-guard functions: each case builds two identical
files (or hands the same file twice), runs the legacy function on one and the admin method on the other, and
compares the results and the files they leave. The restore, row-count and downgrade-guard functions are main's, from
``_main_reference``."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path

import pytest

from hermes_cli.backup import verify_sqlite_integrity
from hermes_cli.backup_sqlite import _safe_copy_db, preflight_state_db
from persistence import PermanentStoreError, ProfileScope
from persistence.sqlite import SQLiteBackend
from persistence.sqlite.admin import SQLiteStoreAdmin
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.hermes_cli.test_session_recovery import _corrupt_middle_table_leaf, _make_page_spanning_source
from tests.persistence_sqlite.admin._main_reference import (
    _count_session_rows, _has_ledger_rows, _pre_runtime_snapshot, _safe_restore_db,
)

_DUMPED = ("sessions", "messages", "runtime_epoch")


def _store_seed(path: Path, *, sessions: int = 2, epochs: int = 0) -> Path:
    """Sessions with three messages each and ``epochs`` runtime epochs, written through the backend's store."""
    path.parent.mkdir(parents=True, exist_ok=True)
    scope = ProfileScope(profile_dir=path.parent, name="default")
    backend = SQLiteBackend({scope.hermes_home_key: path})
    try:
        store = backend.open(scope)
        for number in range(sessions):
            sid = f"backup-{path.parent.name}-{number}"
            seed_session(store, sid, "cli")
            seed_messages(store, sid, *({"role": "user" if turn % 2 == 0 else "assistant",
                                         "content": f"payload {number} {turn}"} for turn in range(3)))
        for _ in range(epochs):
            store.runtime.begin_epoch(instance_id="owner")
    finally:
        backend.close()
    return path


def _clone(src: Path, *destinations: Path) -> None:
    """Self-contained copies of ``src`` (no sidecars), the same bytes in every destination."""
    for dst in destinations:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(src)) as source, closing(sqlite3.connect(dst)) as target:
            source.backup(target)


def _dump(path: Path) -> dict:
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        dump = {t: sorted(conn.execute(f"SELECT * FROM {t}").fetchall(), key=repr) for t in _DUMPED if t in tables}
        dump["journal_mode"] = conn.execute("PRAGMA journal_mode").fetchone()[0]
        return dump


def _digest(path: Path) -> tuple[str, int]:
    return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns


def _zeroed(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * 4096)
    return path


# backup_to


def test_backup_to_copies_the_same_snapshot(tmp_path):
    source = _store_seed(tmp_path / "home" / "state.db")
    legacy, admin = tmp_path / "legacy.db", tmp_path / "admin.db"
    assert _safe_copy_db(source, legacy) is True
    assert SQLiteStoreAdmin(source).backup_to(admin) is True
    assert _dump(admin) == _dump(legacy) == _dump(source)
    assert _dump(admin)["messages"]
    assert (admin.stat().st_mode & 0o777) == (legacy.stat().st_mode & 0o777) == 0o600


def test_backup_to_missing_source_fails_and_leaves_no_destination(tmp_path):
    source = tmp_path / "home" / "state.db"
    legacy, admin = tmp_path / "legacy.db", tmp_path / "admin.db"
    assert _safe_copy_db(source, legacy) is False
    assert SQLiteStoreAdmin(source).backup_to(admin) is False
    assert not legacy.exists() and not admin.exists()


def test_backup_to_locked_source_times_out_the_same(tmp_path):
    source = tmp_path / "locked.db"
    with closing(sqlite3.connect(source)) as conn:
        conn.execute("CREATE TABLE t(x)")
        conn.execute("INSERT INTO t VALUES (1)")
        conn.commit()
    holder = sqlite3.connect(source, isolation_level=None)
    try:
        holder.execute("BEGIN EXCLUSIVE")
        legacy, admin = tmp_path / "legacy.db", tmp_path / "admin.db"
        assert _safe_copy_db(source, legacy, timeout_seconds=0.2) is False
        assert SQLiteStoreAdmin(source).backup_to(admin, timeout_seconds=0.2) is False
        assert not legacy.exists() and not admin.exists()
    finally:
        holder.close()


# preflight_backup


def _emergency_files(home: Path) -> list[str]:
    return sorted(re.sub(r"\d{4}-\d\d-\d\dT[\d-]+Z-\d+", "<stamp>", p.name) for p in home.iterdir())


def test_preflight_backup_publishes_and_prunes_the_same(tmp_path):
    seed = _store_seed(tmp_path / "seed" / "state.db")
    homes = (tmp_path / "legacy", tmp_path / "admin")
    old = [f"state.db.pre-update-emergency-2020-01-0{day}T00-00-00-000000Z-1.bak" for day in (1, 2, 3)]
    for home in homes:
        _clone(seed, home / "state.db")
        for name in old:
            (home / name).write_bytes(b"old")
    legacy = preflight_state_db(homes[0])
    admin = SQLiteStoreAdmin(homes[1] / "state.db").preflight_backup()
    left = [sorted(p.name for p in home.iterdir()) for home in homes]
    assert _emergency_files(homes[1]) == _emergency_files(homes[0])
    assert set(admin) == set(legacy) == {"path", "bytes"}
    assert admin["bytes"] == legacy["bytes"] > 0
    assert Path(admin["path"]).parent == homes[1] and Path(legacy["path"]).parent == homes[0]
    assert [name in left[1] for name in old] == [name in left[0] for name in old] == [False, False, True]
    assert _dump(Path(admin["path"])) == _dump(Path(legacy["path"])) == _dump(seed)



def test_preflight_backup_reclaims_abandoned_staging_the_same(tmp_path):
    seed = _store_seed(tmp_path / "seed" / "state.db")
    homes = (tmp_path / "legacy", tmp_path / "admin")
    hours_ago = time.time() - 3 * 3600
    for home in homes:
        _clone(seed, home / "state.db")
        abandoned = home / "state.db.pre-update-emergency-killed.partial"
        abandoned.write_bytes(b"x" * 4096)
        os.utime(abandoned, (hours_ago, hours_ago))
        (home / "state.db.pre-update-emergency-running.partial").write_bytes(b"x" * 4096)
    preflight_state_db(homes[0])
    SQLiteStoreAdmin(homes[1] / "state.db").preflight_backup()
    staging = [sorted(p.name for p in home.glob("state.db.pre-update-emergency-*.partial*")) for home in homes]
    assert staging[1] == staging[0] == ["state.db.pre-update-emergency-running.partial"]


def _rows_db(path: Path, rows: int, journal_mode: str) -> Path:
    with closing(sqlite3.connect(path)) as setup:
        setup.execute(f"PRAGMA journal_mode={journal_mode}")
        setup.execute("CREATE TABLE messages (body TEXT)")
        setup.executemany("INSERT INTO messages VALUES (?)", [("x" * 4000,)] * rows)
        setup.commit()
    return path


def _copy_in_thread(copy, timeout: float, stop: threading.Event) -> tuple[bool, list]:
    """Run ``copy`` on a thread; whether it finished within ``timeout``, and its result. On a timeout ``stop`` is
    set, which ends the copy, and the thread is joined before its connections can be closed."""
    result: list = []
    thread = threading.Thread(target=lambda: result.append(copy()), daemon=True)
    thread.start()
    thread.join(timeout)
    finished = not thread.is_alive()
    stop.set()
    thread.join()
    return finished, result


@pytest.mark.parametrize("side", ["legacy", "admin"])
def test_a_wal_snapshot_finishes_while_another_connection_keeps_writing(tmp_path, side):
    # A stepped backup restarts on every foreign write and never finishes under load.
    db = _rows_db(tmp_path / "state.db", 4000, "WAL")
    snapshot = tmp_path / "snapshot.db"
    copy = ((lambda: _safe_copy_db(db, snapshot)) if side == "legacy"
            else (lambda: SQLiteStoreAdmin(db).backup_to(snapshot)))
    stop = threading.Event()

    def write_continuously():
        with closing(sqlite3.connect(db, timeout=5)) as writer:
            while not stop.is_set():
                writer.execute("INSERT INTO messages VALUES ('live turn')")
                writer.commit()
                stop.wait(0.005)

    writer = threading.Thread(target=write_continuously)
    writer.start()
    try:
        finished, result = _copy_in_thread(copy, 20, stop)
    finally:
        stop.set()
        writer.join()
    assert finished and result == [True]
    with closing(sqlite3.connect(snapshot)) as copied:
        assert copied.execute("PRAGMA integrity_check").fetchone() == ("ok",)


def test_a_rollback_journal_snapshot_lets_a_live_writer_commit_between_steps(tmp_path, monkeypatch):
    # A rollback-journal reader blocks every commit while it holds the read lock, so a
    # one-step copy of a large state.db would stall the live backend's writes for the whole copy.
    db = _rows_db(tmp_path / "state.db", 2000, "DELETE")
    admin = SQLiteStoreAdmin(db)
    # A held lock fails at once; the copy thread commits through it between steps.
    writer = sqlite3.connect(db, timeout=0, isolation_level=None, check_same_thread=False)
    unread_pages_at_commit, write_errors = [], []
    real_connect = sqlite3.connect
    stop = threading.Event()

    class WriterCommitsBetweenSteps(sqlite3.Connection):
        def backup(self, target, *, pages=-1, progress=None, name="main", sleep=0.25):
            def between_steps(status, remaining, total):
                if stop.is_set():
                    raise RuntimeError("the test stopped the copy")
                if status == sqlite3.SQLITE_OK:
                    try:
                        writer.execute("INSERT INTO messages VALUES ('live turn')")
                        unread_pages_at_commit.append(remaining)
                    except sqlite3.OperationalError as exc:
                        write_errors.append(str(exc))
                if progress is not None:
                    progress(status, remaining, total)
            return super().backup(target, pages=pages, progress=between_steps, name=name, sleep=sleep)

    def connect(database, *args, **kwargs):
        if "mode=ro" in str(database):
            kwargs["factory"] = WriterCommitsBetweenSteps
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", connect)
    snapshot = tmp_path / "snapshot.db"
    try:
        finished, result = _copy_in_thread(lambda: admin.backup_to(snapshot), 30, stop)
    finally:
        writer.close()
    assert finished, "the copy restarted on every commit and never ended"
    assert write_errors == []
    # The writer got in before the copy had read the whole database.
    assert any(remaining > 0 for remaining in unread_pages_at_commit)
    # Each commit restarts the copy; the restarts are bounded and end in a snapshot or a refusal.
    if result == [True]:
        with closing(real_connect(snapshot)) as copied:
            assert copied.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    else:
        assert result == [False] and not snapshot.exists()


def test_a_wal_snapshot_runs_its_one_step_off_the_calling_thread(tmp_path, monkeypatch):
    # A signal handler runs only on the main thread between bytecodes, so the one C backup step runs elsewhere
    # and a SIGTERM during the copy is serviced while it runs.
    db = _rows_db(tmp_path / "state.db", 10, "WAL")
    admin = SQLiteStoreAdmin(db)
    steps = []
    real_connect = sqlite3.connect

    class RecordingSteps(sqlite3.Connection):
        def backup(self, target, *, pages=-1, progress=None, name="main", sleep=0.25):
            steps.append((pages, threading.get_ident()))
            return super().backup(target, pages=pages, progress=progress, name=name, sleep=sleep)

    def connect(database, *args, **kwargs):
        if "mode=ro" in str(database):
            kwargs["factory"] = RecordingSteps
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", connect)
    assert admin.backup_to(tmp_path / "snapshot.db") is True
    assert [pages for pages, _ in steps] == [-1]
    assert steps[0][1] != threading.get_ident()


def test_preflight_backup_without_a_file_reports_the_same(tmp_path):
    homes = (tmp_path / "legacy", tmp_path / "admin")
    for home in homes:
        home.mkdir()
    assert SQLiteStoreAdmin(homes[1] / "state.db").preflight_backup() == preflight_state_db(homes[0])
    assert [list(home.iterdir()) for home in homes] == [[], []]


def test_preflight_backup_of_a_zeroed_file_raises_the_same(tmp_path):
    homes = (tmp_path / "legacy", tmp_path / "admin")
    for home in homes:
        _zeroed(home / "state.db")
    with pytest.raises(RuntimeError) as legacy:
        preflight_state_db(homes[0])
    with pytest.raises(RuntimeError) as admin:
        SQLiteStoreAdmin(homes[1] / "state.db").preflight_backup()
    assert str(admin.value) == str(legacy.value)
    assert _emergency_files(homes[1]) == _emergency_files(homes[0]) == ["state.db"]


# verify_file


def _too_small(path: Path) -> Path:
    path.write_bytes(b"SQLite format 3\0" + b"\0" * 20)
    return path


def _damaged(path: Path) -> Path:
    messages_root, _ = _make_page_spanning_source(path, message_count=120)
    _corrupt_middle_table_leaf(path, messages_root)
    return path


@pytest.mark.parametrize("build, valid", [
    (lambda p: _store_seed(p), True),
    (lambda p: p, False),
    (_too_small, False),
    (_zeroed, False),
    (_damaged, False),
], ids=["valid", "missing", "too-small", "zeroed", "damaged-leaf"])
def test_verify_file_matches(tmp_path, build, valid):
    path = build(tmp_path / "state.db")
    legacy = verify_sqlite_integrity(path)
    assert SQLiteStoreAdmin.verify_file(path) == legacy
    assert legacy["valid"] is valid
    for max_bytes in (0, 200):
        assert SQLiteStoreAdmin.verify_file(path, max_bytes=max_bytes) == verify_sqlite_integrity(path, max_bytes=max_bytes)


# restore_from


def _restore_pair(tmp_path, *, destination_epochs: int, snapshot_epochs: int):
    """Two identical (destination, snapshot) pairs: the destination holds more sessions than the snapshot."""
    snapshot_seed = _store_seed(tmp_path / "snapshot-seed" / "state.db", sessions=1, epochs=snapshot_epochs)
    destination_seed = _store_seed(tmp_path / "destination-seed" / "state.db", sessions=3,
                                   epochs=destination_epochs)
    pairs = []
    for side in ("legacy", "admin"):
        destination, snapshot = tmp_path / side / "state.db", tmp_path / f"{side}-snapshots" / "state.db"
        _clone(destination_seed, destination)
        _clone(snapshot_seed, snapshot)
        pairs.append((destination, snapshot))
    return pairs


def _messages(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING]


def test_restore_from_restores_the_snapshot_pages(tmp_path):
    (legacy_dst, legacy_snap), (admin_dst, admin_snap) = _restore_pair(tmp_path, destination_epochs=0,
                                                                         snapshot_epochs=0)
    before = _dump(admin_dst)
    assert _safe_restore_db(legacy_snap, legacy_dst) is True
    assert SQLiteStoreAdmin(admin_dst, read_only=False).restore_from(admin_snap) is True
    assert _dump(admin_dst) == _dump(legacy_dst) == _dump(admin_snap) != before
    assert (admin_dst.stat().st_mode & 0o777) == (legacy_dst.stat().st_mode & 0o777)


def test_restore_from_stages_the_destination_epoch_floor_into_a_copy(tmp_path):
    (legacy_dst, legacy_snap), (admin_dst, admin_snap) = _restore_pair(tmp_path, destination_epochs=3,
                                                                         snapshot_epochs=1)
    snapshot_before = _digest(admin_snap)
    assert _safe_restore_db(legacy_snap, legacy_dst) is True
    assert SQLiteStoreAdmin(admin_dst, read_only=False).restore_from(admin_snap) is True
    assert _dump(admin_dst) == _dump(legacy_dst)
    assert _dump(admin_dst)["runtime_epoch"] == [(1, 3, "offline-restore")]
    assert _dump(admin_dst)["sessions"] == _dump(admin_snap)["sessions"]
    assert _digest(admin_snap) == snapshot_before
    assert _dump(admin_snap)["runtime_epoch"] == [(1, 1, "owner")]
    assert sorted(p.name for p in admin_dst.parent.iterdir()) == sorted(p.name for p in legacy_dst.parent.iterdir())
    assert not list(admin_dst.parent.glob(".restore-epoch-*"))


def test_restore_from_keeps_a_higher_snapshot_epoch(tmp_path):
    (legacy_dst, legacy_snap), (admin_dst, admin_snap) = _restore_pair(tmp_path, destination_epochs=1,
                                                                         snapshot_epochs=2)
    assert _safe_restore_db(legacy_snap, legacy_dst) is True
    assert SQLiteStoreAdmin(admin_dst, read_only=False).restore_from(admin_snap) is True
    assert _dump(admin_dst) == _dump(legacy_dst) == _dump(admin_snap)
    assert _dump(admin_dst)["runtime_epoch"] == [(1, 2, "owner")]


def test_restore_from_refuses_an_unreadable_destination_epoch(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    snapshot = _store_seed(tmp_path / "snapshots" / "state.db")
    results, logs = {}, {}
    for side in ("legacy", "admin"):
        destination = tmp_path / side / "state.db"
        destination.parent.mkdir()
        destination.write_bytes(b"not a database " * 300)
        caplog.clear()
        if side == "legacy":
            results[side] = _safe_restore_db(snapshot, destination)
        else:
            results[side] = SQLiteStoreAdmin(destination, read_only=False).restore_from(snapshot)
        logs[side] = [m.replace(str(destination), "<dst>") for m in _messages(caplog)]
        assert destination.read_bytes() == b"not a database " * 300
    assert results == {"legacy": False, "admin": False}
    assert logs["admin"] == logs["legacy"]
    assert logs["admin"] == [
        "Cannot read the canonical runtime epoch at <dst>; refusing in-place restore. Use `hermes sessions recover "
        "--source <snapshot> --output <separate-path>` to salvage transcripts into a separate output."
    ]


def test_restore_from_refuses_a_zeroed_snapshot(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    (legacy_dst, _), (admin_dst, _) = _restore_pair(tmp_path, destination_epochs=0, snapshot_epochs=0)
    snapshot = _zeroed(tmp_path / "zeroed" / "state.db")
    before = _digest(admin_dst)
    caplog.clear()
    assert _safe_restore_db(snapshot, legacy_dst) is False
    legacy_logs = _messages(caplog)
    caplog.clear()
    assert SQLiteStoreAdmin(admin_dst, read_only=False).restore_from(snapshot) is False
    assert _messages(caplog) == legacy_logs and legacy_logs
    assert _digest(admin_dst) == before


def test_restore_from_refuses_while_maintenance_is_reserved(tmp_path, caplog):
    from gateway.runtime_ownership import exclusive_maintenance

    caplog.set_level(logging.DEBUG)
    (legacy_dst, legacy_snap), (admin_dst, admin_snap) = _restore_pair(tmp_path, destination_epochs=0,
                                                                         snapshot_epochs=0)
    held, release, done = threading.Event(), threading.Event(), threading.Event()

    def hold() -> None:
        with exclusive_maintenance([legacy_dst.parent, admin_dst.parent]):
            held.set()
            release.wait(10)
        done.set()

    worker = threading.Thread(target=hold)
    worker.start()
    try:
        assert held.wait(10)
        before = _digest(admin_dst)
        caplog.clear()
        assert _safe_restore_db(legacy_snap, legacy_dst) is False
        legacy_logs = [m.replace(str(legacy_dst.parent), "<home>") for m in _messages(caplog)]
        caplog.clear()
        assert SQLiteStoreAdmin(admin_dst, read_only=False).restore_from(admin_snap) is False
        assert [m.replace(str(admin_dst.parent), "<home>") for m in _messages(caplog)] == legacy_logs
        assert legacy_logs
        assert _digest(admin_dst) == before
    finally:
        release.set()
        worker.join(10)
    assert done.is_set()


def test_restore_from_on_a_read_only_handle_refuses(tmp_path):
    (_, _), (admin_dst, admin_snap) = _restore_pair(tmp_path, destination_epochs=0, snapshot_epochs=0)
    before = _digest(admin_dst)
    with pytest.raises(PermanentStoreError):
        SQLiteStoreAdmin(admin_dst).restore_from(admin_snap)
    assert _digest(admin_dst) == before


# session_row_counts


def _plain(path: Path) -> Path:
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE notes(x)")
        conn.commit()
    return path


@pytest.mark.parametrize("build, expected", [
    (lambda p: _store_seed(p, sessions=2), (2, 6)),
    (lambda p: p, None),
    (_plain, None),
    (_zeroed, None),
], ids=["seeded", "missing", "not-a-session-store", "zeroed"])
def test_session_row_counts_match(tmp_path, build, expected):
    path = build(tmp_path / "state.db")
    assert SQLiteStoreAdmin(path).session_row_counts() == _count_session_rows(path) == expected


# has_runtime_ledger_rows


def _ledger(path: Path, table: str) -> Path:
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE session_admissions(id)")
        conn.execute("CREATE TABLE worker_executions(id)")
        conn.execute(f"INSERT INTO {table} VALUES (1)")
        conn.commit()
    return path


@pytest.mark.parametrize("build, expected", [
    (lambda p: p, False),
    (lambda p: _store_seed(p), False),
    (_plain, False),
    (_zeroed, False),
    (lambda p: _ledger(p, "session_admissions"), True),
    (lambda p: _ledger(p, "worker_executions"), True),
], ids=["missing", "empty-ledgers", "no-ledger-tables", "zeroed", "admission-row", "execution-row"])
def test_has_runtime_ledger_rows_matches(tmp_path, build, expected):
    path = build(tmp_path / "state.db")
    assert SQLiteStoreAdmin.has_runtime_ledger_rows(path) is _has_ledger_rows(path) is expected



# has_runtime_ledger_table


def _snapshot(home: Path, name: str, build) -> Path:
    from hermes_cli.backup import _quick_snapshot_root

    path = _quick_snapshot_root(home) / name / "state.db"
    path.parent.mkdir(parents=True)
    build(path)
    return path


def _sessions_only(path: Path) -> Path:
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
    return path


def test_has_runtime_ledger_table_answers_the_downgrade_guard_snapshot_probe(tmp_path):
    with_ledger, home = tmp_path / "with" / "home", tmp_path / "without" / "home"
    current = _snapshot(with_ledger, "20260101", _store_seed)
    assert _pre_runtime_snapshot(with_ledger) is None
    assert SQLiteStoreAdmin.has_runtime_ledger_table(current) is True
    old = _snapshot(home, "20260101", _sessions_only)
    assert _pre_runtime_snapshot(home) == "20260101"
    assert SQLiteStoreAdmin.has_runtime_ledger_table(old) is False


def test_has_runtime_ledger_table_raises_on_a_file_the_guard_skips(tmp_path):
    from persistence import PersistenceError

    home = tmp_path / "home"
    garbage = _snapshot(home, "20260101", lambda path: path.write_bytes(b"not a database" * 512))
    assert _pre_runtime_snapshot(home) is None
    with pytest.raises(PersistenceError, match="file is not a database"):
        SQLiteStoreAdmin.has_runtime_ledger_table(garbage)
