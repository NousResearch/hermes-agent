"""``DoctorAdmin`` against the doctor, repair, journal-mode and change-watcher functions: each case runs the function and the admin method on two identical copies of one seeded file (or on the same file twice when the probe reads only), and compares the results and the files they leave. A held file is held by a child process with an open read transaction, started and stopped around the case. The doctor's write-health probe and session count are main's, from ``_main_reference``."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from contextlib import closing, contextmanager
from pathlib import Path

import pytest

from gateway.lifecycle_ledger import _state_db_journal_mode, check_state_db_integrity
from persistence.sqlite._state.dbfile import collect_state_db_stats, count_db_holders, iter_deleted_sqlite_sidecar_holders
from persistence.sqlite._state.holders import foreign_state_db_holders, held_store_refusal
from persistence.sqlite._state.repair import (
    _db_opens_cleanly, _exclusive_repair_db_guard, _live_writer_holds_db, repair_state_db_schema,
    state_db_has_structural_damage,
)
from persistence import PermanentStoreError, ProfileScope, StoreContention
from persistence.sqlite import SQLiteBackend
from persistence.sqlite.admin import SQLiteStoreAdmin, _doctor
from persistence.sqlite.admin._doctor import SQLiteRepairGuard
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.persistence_sqlite.admin import _main_reference
from tests.persistence_sqlite.admin._main_reference import _session_count, _write_health_reason
from tui_gateway.change_watcher import _newest_mtime_ns

_HOLD = """
import sqlite3, sys
conn = sqlite3.connect(sys.argv[1], isolation_level=None)
conn.execute("BEGIN")
conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
print("ready", flush=True)
sys.stdin.read()
"""


def _seed(path: Path, *, rows: int = 120) -> Path:
    from hermes_state import SessionDB

    path.parent.mkdir(parents=True, exist_ok=True)
    db = SessionDB(db_path=path)
    try:
        db.create_session("s1", source="cli")
        for i in range(rows):
            db.append_message("s1", "user", f"hello {i} alpha beta " + "lorem " * 30)
    finally:
        db.close()
    with closing(sqlite3.connect(path)) as raw:
        raw.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return path


def _store_seed(path: Path, *, rows: int = 120) -> Path:
    """One session of ``rows`` user messages written through the backend's store, its WAL checkpointed into the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    scope = ProfileScope(profile_dir=path.parent, name="default")
    backend = SQLiteBackend({scope.hermes_home_key: path})
    try:
        store = backend.open(scope)
        seed_session(store, "s1", "cli")
        seed_messages(store, "s1", *({"role": "user", "content": f"hello {i} alpha beta " + "lorem " * 30}
                                     for i in range(rows)))
    finally:
        backend.close()
    with closing(sqlite3.connect(path)) as raw:
        raw.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return path


def _clone(src: Path, *destinations: Path) -> None:
    for dst in destinations:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(src)) as source, closing(sqlite3.connect(dst)) as target:
            source.backup(target)


def _dump(path: Path) -> dict:
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
        return {t: sorted(conn.execute(f"SELECT * FROM {t}").fetchall(), key=repr) for t in ("sessions", "messages")}


@pytest.fixture
def store_pair(tmp_path):
    """``(legacy path, admin path)``: two identical copies of one seeded store, each in its own home."""
    seed = _store_seed(tmp_path / "seed" / "state.db")
    legacy, admin = tmp_path / "legacy" / "state.db", tmp_path / "admin" / "state.db"
    _clone(seed, legacy, admin)
    return legacy, admin


def _tear_sessions_btree(path: Path) -> None:
    """Overwrite the cell pointers of the ``sessions`` root page: canonical-table damage, not FTS."""
    with closing(sqlite3.connect(path)) as raw:
        page_size = raw.execute("PRAGMA page_size").fetchone()[0]
        root = raw.execute("SELECT rootpage FROM sqlite_master WHERE name='sessions'").fetchone()[0]
    with open(path, "r+b") as f:
        f.seek((root - 1) * page_size + 8)
        f.write(b"\xff\xff" * 8)


def _corrupt_fts_index_data(path: Path) -> None:
    """Garbage FTS5 shadow blocks: reads and integrity_check pass, the MATCH and write probes fail."""
    with closing(sqlite3.connect(path, isolation_level=None)) as conn:
        conn.execute("UPDATE messages_fts_data SET block = X'DEADBEEFDEADBEEF'")


def _corrupt_duplicate_fts(path: Path) -> None:
    """A duplicate ``messages_fts`` row in sqlite_master: every statement fails with a malformed schema."""
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("PRAGMA writable_schema=ON")
        conn.execute(
            "INSERT INTO sqlite_master (type, name, tbl_name, rootpage, sql) "
            "SELECT type, name, tbl_name, rootpage, sql FROM sqlite_master WHERE name='messages_fts'")
        conn.commit()


@contextmanager
def _held(*paths: Path):
    """One child process per path holding it open inside a read transaction until the block exits."""
    procs = [subprocess.Popen([sys.executable, "-c", _HOLD, str(path)], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, text=True) for path in paths]
    try:
        for proc in procs:
            assert proc.stdout.readline().strip() == "ready"
        yield procs
    finally:
        for proc in procs:
            proc.kill()
            proc.wait(timeout=10)


def _admin(path: Path, *, read_only: bool = True) -> SQLiteStoreAdmin:
    return SQLiteStoreAdmin(path, read_only=read_only)


# holders and stats


def test_probes_on_a_quiet_file(store_pair):
    path = store_pair[0]
    admin = _admin(path)
    assert admin.stats() == collect_state_db_stats(path)
    assert admin.stats()["sessions"] == 1 and admin.stats()["messages"] == 120
    assert admin.holder_count() == count_db_holders(path)
    assert admin.deleted_sidecar_holders() == iter_deleted_sqlite_sidecar_holders(path) == []
    assert admin.foreign_holders() == foreign_state_db_holders(path) == []
    assert admin.held_refusal(command="optimize-storage") is held_store_refusal(path, command="optimize-storage") is None
    assert admin.live_writer_holds() is _live_writer_holds_db(path) is False


def test_probes_on_a_held_file(store_pair):
    path = store_pair[0]
    admin = _admin(path)
    with _held(path) as (proc,):
        holders = foreign_state_db_holders(path)
        assert holders and {pid for pid, _ in holders} == {proc.pid}
        assert admin.foreign_holders() == holders
        assert admin.holder_count() == count_db_holders(path) >= 1
        for force_hint in ("--force", None):
            refusal = held_store_refusal(path, command="optimize-storage", force_hint=force_hint)
            assert refusal is not None and str(proc.pid) in refusal
            assert admin.held_refusal(command="optimize-storage", force_hint=force_hint) == refusal
        assert admin.live_writer_holds() is _live_writer_holds_db(path) is True


@pytest.mark.platforms("linux")
def test_deleted_sidecar_holders(store_pair):
    """A child keeps the WAL sidecars open while their paths are unlinked: a retired generation."""
    path = store_pair[0]
    with _held(path) as (proc,):
        for suffix in ("-wal", "-shm"):
            Path(f"{path}{suffix}").unlink()
        expected = iter_deleted_sqlite_sidecar_holders(path)
        assert expected and {pid for pid, _ in expected} == {proc.pid}
        assert _admin(path).deleted_sidecar_holders() == expected


def test_change_stamp(store_pair, tmp_path):
    path = store_pair[0]
    expected = _newest_mtime_ns((path, path.with_name(f"{path.name}-wal")))
    assert expected is not None and _admin(path).change_stamp() == expected
    missing = tmp_path / "missing" / "state.db"
    assert _admin(missing).change_stamp() is _newest_mtime_ns((missing, Path(f"{missing}-wal"))) is None


# health probes


@pytest.mark.parametrize("damage", [None, _corrupt_fts_index_data, _tear_sessions_btree])
def test_opens_cleanly_and_structural_damage(store_pair, damage):
    legacy, admin = store_pair
    if damage is not None:
        damage(legacy)
        damage(admin)
    expected = _db_opens_cleanly(legacy)
    assert _admin(admin).opens_cleanly() == expected
    assert (expected is None) is (damage is None)
    structural = state_db_has_structural_damage(legacy)
    assert _admin(admin).has_structural_damage() is structural
    assert structural is (damage is _tear_sessions_btree)
    if not structural:
        assert _dump(admin) == _dump(legacy)


@pytest.mark.parametrize("should_fix", [False, True])
def test_write_health_reason_on_a_quiet_file(store_pair, should_fix):
    legacy, admin = store_pair
    assert _write_health_reason(legacy, should_fix=should_fix) is None
    assert _admin(admin).write_health_reason(should_fix=should_fix) is None
    _corrupt_fts_index_data(legacy)
    _corrupt_fts_index_data(admin)
    expected = _write_health_reason(legacy, should_fix=should_fix)
    assert expected is not None
    assert _admin(admin).write_health_reason(should_fix=should_fix) == expected
    assert _dump(admin) == _dump(legacy)


def test_write_health_reason_on_a_held_file_probes_a_snapshot(store_pair):
    path = store_pair[0]
    _corrupt_fts_index_data(path)
    before = _dump(path)
    with _held(path):
        expected = _write_health_reason(path, should_fix=False)
        assert expected is not None
        assert _admin(path).write_health_reason(should_fix=False) == expected
    assert _dump(path) == before


# exclusive repair guard


def _write_refused(path: Path) -> bool:
    with closing(sqlite3.connect(path, timeout=0.0, isolation_level=None)) as other:
        try:
            other.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError:
            return True
        other.execute("ROLLBACK")
        return False


def test_exclusive_repair_guard_excludes_other_writers(store_pair):
    legacy, admin = store_pair
    with _exclusive_repair_db_guard(legacy) as (guard, error):
        assert guard is not None and error is None
        legacy_refused = _write_refused(legacy)
    with _admin(admin).exclusive_repair_guard() as value:
        assert isinstance(value, SQLiteRepairGuard)
        admin_refused = _write_refused(admin)
    assert admin_refused is legacy_refused is True
    assert _write_refused(legacy) is _write_refused(admin) is False


def test_exclusive_repair_guard_on_a_held_file_raises_contention(store_pair):
    legacy, admin = store_pair
    with _held(legacy, admin):
        with _exclusive_repair_db_guard(legacy) as (guard, error):
            assert guard is None and error is not None
        entered = False
        with pytest.raises(StoreContention) as raised:
            with _admin(admin).exclusive_repair_guard():
                entered = True
    assert not entered
    assert str(raised.value) == str(error)
    assert isinstance(raised.value.__cause__, type(error))


# integrity check and journal mode


@pytest.mark.parametrize("damage", [None, _tear_sessions_btree])
@pytest.mark.parametrize("deadline_s", [None, 0.0, 60.0])
def test_integrity_check(store_pair, damage, deadline_s):
    legacy, admin = store_pair
    if damage is not None:
        damage(legacy)
        damage(admin)
    expected = check_state_db_integrity(home=legacy.parent, deadline_s=deadline_s)
    assert _admin(admin).integrity_check(deadline_s=deadline_s) == expected
    if damage is not None and deadline_s != 0.0:
        assert expected != "ok"


def test_integrity_check_of_a_missing_file(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    assert _admin(home / "state.db").integrity_check() == check_state_db_integrity(home=home) == "absent"
    assert list(home.iterdir()) == []


@pytest.mark.parametrize("mode", ["wal", "delete"])
def test_journal_mode(store_pair, mode):
    legacy, admin = store_pair
    for path in store_pair:
        with closing(sqlite3.connect(path, isolation_level=None)) as conn:
            conn.execute(f"PRAGMA journal_mode={mode}")
    assert _admin(admin).journal_mode() == _state_db_journal_mode(legacy) == mode


# schema repair


def _normalized(report: dict, home: Path) -> dict:
    out = dict(report)
    if out.get("backup_path"):
        out["backup_path"] = Path(out["backup_path"]).relative_to(home).as_posix().split(".malformed-backup")[0]
    return out


def test_repair_schema_of_a_malformed_schema(store_pair):
    legacy, admin = store_pair
    _corrupt_duplicate_fts(legacy)
    _corrupt_duplicate_fts(admin)
    assert _db_opens_cleanly(legacy) is not None
    expected = repair_state_db_schema(legacy)
    actual = _admin(admin, read_only=False).repair_schema()
    assert expected["repaired"] is True
    assert _normalized(actual, admin.parent) == _normalized(expected, legacy.parent)
    assert _dump(admin) == _dump(legacy)
    assert _db_opens_cleanly(admin) is None


def test_repair_schema_without_backup_and_of_a_missing_file(store_pair, tmp_path):
    legacy, admin = store_pair
    _corrupt_duplicate_fts(legacy)
    _corrupt_duplicate_fts(admin)
    expected = repair_state_db_schema(legacy, backup=False)
    assert _admin(admin, read_only=False).repair_schema(backup=False) == expected
    assert expected["backup_path"] is None and expected["repaired"] is True
    missing = tmp_path / "missing" / "state.db"
    assert _admin(missing, read_only=False).repair_schema() == {
        **repair_state_db_schema(missing), "error": f"{missing} does not exist"}


def test_repair_schema_on_a_read_only_handle_refuses(store_pair):
    admin = store_pair[1]
    _corrupt_duplicate_fts(admin)
    with pytest.raises(PermanentStoreError):
        _admin(admin).repair_schema()
    assert _db_opens_cleanly(admin) is not None


def test_integrity_check_claims_and_renews_the_startup_lease_as_the_legacy_check(store_pair, monkeypatch):
    import hermes_startup_watchdog
    from persistence.sqlite.admin import _doctor

    calls: list[tuple[float, str]] = []
    monkeypatch.setattr(
        hermes_startup_watchdog, "report_startup_progress",
        lambda seconds, *, phase: calls.append((seconds, phase)),
    )
    monkeypatch.setattr(_doctor, "_INTEGRITY_CHECK_LEASE_RENEW_S", 0.0)
    monkeypatch.setattr(_doctor, "_INTEGRITY_CHECK_PROGRESS_OPS", 1_000)
    legacy, admin = store_pair
    expected = check_state_db_integrity(home=legacy.parent)
    legacy_calls, calls[:] = list(calls), []
    assert _admin(admin).integrity_check() == expected == "ok"
    assert calls == legacy_calls
    assert len(calls) > 1 and set(calls) == {(900.0, "state_db_unclean_integrity_check")}


def test_the_repair_guard_checkpoints_the_wal_as_the_legacy_guard(store_pair):
    sizes = []
    for path in store_pair:
        with closing(sqlite3.connect(path, isolation_level=None)) as conn:
            conn.execute("PRAGMA journal_mode=wal")
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            # The frames written below stay in the WAL after this connection closes.
            conn.setconfig(sqlite3.SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE, True)
            conn.execute("PRAGMA wal_autocheckpoint=0")
            conn.execute("CREATE TABLE IF NOT EXISTS filler (x BLOB)")
            conn.executemany("INSERT INTO filler VALUES (?)", [(b"x" * 4000,) for _ in range(50)])
        sizes.append(path.stat().st_size)
    legacy, admin = store_pair
    with _exclusive_repair_db_guard(legacy) as (guard, _error):
        guard.execute("PRAGMA wal_checkpoint(PASSIVE)")
    with _admin(admin).exclusive_repair_guard() as guard:
        guard.wal_checkpoint_passive()
    assert admin.stat().st_size == legacy.stat().st_size > sizes[0]
    assert _dump(admin) == _dump(legacy)


@pytest.mark.parametrize("should_fix", [False, True])
def test_write_health_reason_skips_a_held_file_over_the_snapshot_limit_as_the_legacy(store_pair, monkeypatch, should_fix):
    path = store_pair[0]
    printed = []
    monkeypatch.setattr(_main_reference, "_WRITE_PROBE_SNAPSHOT_MAX_BYTES", 0)
    monkeypatch.setattr(_main_reference, "check_info", printed.append)
    monkeypatch.setattr(_doctor, "_WRITE_PROBE_SNAPSHOT_MAX_BYTES", 0)
    with _held(path):
        expected = _write_health_reason(path, should_fix=should_fix)
        answer = _admin(path).write_health_reason(should_fix=should_fix)
    if should_fix:
        assert answer == expected and printed == []
    else:
        assert expected is None and printed == [_doctor.WRITE_PROBE_SKIPPED]
        assert answer == _doctor.WRITE_PROBE_SKIPPED


def test_session_row_count_matches_the_doctor_count(store_pair):

    legacy, admin = store_pair
    expected = _session_count(legacy)
    assert expected > 0 and _admin(admin).session_row_count() == expected


def test_session_row_count_of_a_file_that_is_no_database_raises_the_doctor_message(tmp_path):
    from persistence import PersistenceError

    path = tmp_path / "state.db"
    path.write_bytes(b"not a database" * 512)
    with pytest.raises(sqlite3.DatabaseError) as legacy:
        _session_count(path)
    with pytest.raises(PersistenceError) as admin:
        _admin(path).session_row_count()
    assert str(admin.value) == str(legacy.value)


def test_the_doctor_answers_with_the_admin_contracts_guard_and_skip_notice():
    from persistence.core import admin

    assert _doctor.WRITE_PROBE_SKIPPED is admin.WRITE_PROBE_SKIPPED
    assert admin.RepairGuard in SQLiteRepairGuard.__mro__
