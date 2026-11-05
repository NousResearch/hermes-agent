"""``MaintenanceAdmin`` against the legacy ``SessionDB``: each case seeds one store, copies it, runs the legacy method
on one copy and the admin operation on the other, and compares the results and the files."""

from __future__ import annotations

import os
import select
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from persistence.sqlite._state.errors import StateDbCorruptError
from persistence.sqlite._state.repair import _release_auto_maintenance_lock, _try_acquire_auto_maintenance_lock
from persistence import CorruptStoreError, PermanentStoreError
from persistence.sqlite import _registry
from persistence.sqlite.admin import SQLiteStoreAdmin
from session_retention import run_auto_prune_and_vacuum
from tests.persistence_sqlite._legacy_store import legacy_store
from tests.persistence_sqlite.admin._main_reference import _print_fts_optimize_available_notice
from tests.persistence_sqlite.conftest import store_for

__all__ = ["legacy_store"]

DAY = 86400.0
MEDIA_ROOT = "/nonexistent/native-inputs"


def _copy(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    src, dst = sqlite3.connect(source), sqlite3.connect(target)
    try:
        src.backup(dst)
    finally:
        src.close()
        dst.close()
    return target


def _copies(tmp_path: Path, seed) -> tuple[Path, Path]:
    """Two identical copies of one seeded store: the legacy side's and the admin's."""
    seed_path = tmp_path / "seed" / "state.db"
    seed_path.parent.mkdir()
    seed(seed_path)
    return _copy(seed_path, tmp_path / "legacy" / "state.db"), _copy(seed_path, tmp_path / "admin" / "state.db")


def _pragma(path: Path, name: str):
    conn = sqlite3.connect(path)
    try:
        return conn.execute(f"PRAGMA {name}").fetchone()[0]
    finally:
        conn.close()


def _rows(path: Path, sql: str) -> list:
    conn = sqlite3.connect(path)
    try:
        return sorted(conn.execute(sql).fetchall(), key=repr)
    finally:
        conn.close()


def _seed_with_deletes(path: Path) -> None:
    """A store with FTS rows and a large delete, so VACUUM has pages to give back."""
    from hermes_state import SessionDB

    with SessionDB(db_path=path) as db:
        for n in range(4):
            db.create_session(f"s{n}", source="cli")
            for i in range(60):
                db.append_message(f"s{n}", role="user", content=f"zebra quokka {n}-{i} " + "x" * 2000)
        db._execute_write(lambda conn: conn.execute("DELETE FROM messages WHERE session_id IN ('s0', 's1', 's2')"))


def _seed_prunable(path: Path) -> None:
    """One ended session older than any retention, so automatic maintenance prunes and considers VACUUM."""
    from hermes_state import SessionDB

    with SessionDB(db_path=path) as db:
        db.create_session("old", source="cli")
        db.append_message("old", role="user", content="y" * 5000)
        old = time.time() - 200 * DAY
        db._execute_write(lambda conn: conn.execute(
            "UPDATE sessions SET started_at = ?, ended_at = ?, end_reason = 'user_exit', last_activity_at = NULL "
            "WHERE id = 'old'", (old, old + 60)))
        db._execute_write(lambda conn: conn.execute("UPDATE messages SET timestamp = ?", (old,)))


def _legacy_inline_store(path: Path, n: int = 1200) -> None:
    """The legacy-v22 inline FTS layout, with an id gap from a pruned range."""
    from hermes_state_common import LEGACY_FTS_SQL, LEGACY_FTS_TRIGRAM_SQL, SCHEMA_SQL

    ids = [i + 1 + (100_000 if i >= 600 else 0) for i in range(n)]
    raw = sqlite3.connect(path)
    try:
        raw.executescript(SCHEMA_SQL + LEGACY_FTS_SQL + LEGACY_FTS_TRIGRAM_SQL)
        raw.execute("INSERT INTO sessions (id, source, started_at) VALUES ('s1', 'cli', 1000.0)")
        raw.executemany(
            "INSERT INTO messages (id, session_id, timestamp, role, content) VALUES (?, 's1', 1000.0, 'user', ?)",
            [(mid, f"legacy payload {mid} zebra quokka " + "inline copy " * 40) for mid in ids],
        )
        raw.commit()
    finally:
        raw.close()


def _current_store(path: Path) -> None:
    from hermes_state import SessionDB

    with SessionDB(db_path=path) as db:
        db.create_session("s1", source="cli")
        db.append_message("s1", role="user", content="zebra quokka")


# ── VACUUM ─────────────────────────────────────────────────────────────────


def test_vacuum_merges_fts_and_reclaims_pages_like_the_legacy_store(tmp_path, legacy_store):
    from hermes_state import SessionDB

    legacy_path, admin_path = _copies(tmp_path, _seed_with_deletes)
    pages_before = _pragma(legacy_path, "page_count")
    assert _pragma(legacy_path, "freelist_count") > 0

    db = SessionDB(db_path=legacy_path)
    try:
        legacy = db.vacuum()
    finally:
        db.close()
    admin = SQLiteStoreAdmin(admin_path, read_only=False).vacuum()

    assert admin == legacy and admin > 0
    for path in (legacy_path, admin_path):
        assert _pragma(path, "page_count") < pages_before
        assert _pragma(path, "freelist_count") == 0
    assert _pragma(admin_path, "page_count") == _pragma(legacy_path, "page_count")
    assert _pragma(admin_path, "journal_mode") == _pragma(legacy_path, "journal_mode")
    sql = "SELECT session_id, content FROM messages"
    assert _rows(admin_path, sql) == _rows(legacy_path, sql)
    match = "SELECT rowid FROM messages_fts WHERE messages_fts MATCH 'zebra'"
    assert _rows(admin_path, match) == _rows(legacy_path, match) != []


def test_read_only_vacuum_is_refused_and_writes_nothing(tmp_path, legacy_store):
    from hermes_state import SessionDB

    legacy_path, admin_path = _copies(tmp_path, _seed_with_deletes)
    before = {path: path.read_bytes() for path in (legacy_path, admin_path)}

    db = SessionDB(db_path=legacy_path, read_only=True)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            db.vacuum()
    finally:
        db.close()
    with pytest.raises(PermanentStoreError):
        SQLiteStoreAdmin(admin_path, read_only=True).vacuum()

    assert {path: path.read_bytes() for path in before} == before


def test_vacuum_on_a_quarantined_store_raises_before_any_rewrite(tmp_path, legacy_store):
    from hermes_state import SessionDB, StateDbCorruptError as LegacyCorruptError

    legacy_path, admin_path = _copies(tmp_path, _seed_with_deletes)
    damage = sqlite3.DatabaseError("database disk image is malformed")

    db = SessionDB(db_path=legacy_path)
    holder = _registry.acquire(admin_path)
    try:
        with pytest.raises(LegacyCorruptError):
            db._halt_db_corrupt(damage)
        with pytest.raises(StateDbCorruptError):
            holder._halt_db_corrupt(damage)
        pages = {path: _pragma(path, "page_count") for path in (legacy_path, admin_path)}
        with pytest.raises(LegacyCorruptError) as legacy:
            db.vacuum()
        with pytest.raises(CorruptStoreError) as admin:
            SQLiteStoreAdmin(admin_path, read_only=False).vacuum()
        assert str(admin.value) == str(legacy.value).replace(str(legacy_path), str(admin_path))
        assert {path: _pragma(path, "page_count") for path in pages} == pages
    finally:
        db.close()
        _registry.release_or_close(holder)


# ── page statistics ────────────────────────────────────────────────────────


@pytest.mark.parametrize("read_only", [True, False], ids=["read-only", "writable"])
def test_page_statistics_match_the_legacy_store(tmp_path, read_only, legacy_store):
    from hermes_state import SessionDB

    legacy_path, admin_path = _copies(tmp_path, _seed_with_deletes)
    db = SessionDB(db_path=legacy_path, read_only=read_only)
    try:
        legacy = (db.logical_size_bytes(), db._freelist_ratio())
    finally:
        db.close()
    admin = SQLiteStoreAdmin(admin_path, read_only=read_only)

    assert (admin.logical_size_bytes(), admin.freelist_ratio()) == legacy
    assert legacy[0] > 0 and legacy[1] > 0


def test_page_statistics_of_a_missing_file_are_none_and_create_nothing(tmp_path):
    path = tmp_path / "absent" / "state.db"
    admin = SQLiteStoreAdmin(path, read_only=True)
    assert (admin.logical_size_bytes(), admin.freelist_ratio()) == (None, None)
    assert not path.exists()


# ── auto-maintenance admission ─────────────────────────────────────────────


def _prunable_pair(tmp_path: Path, store_path: Path) -> tuple[Path, Path]:
    """One prunable store, copied to the legacy side's path and to ``store_path``."""
    seed_path = tmp_path / "seed" / "state.db"
    seed_path.parent.mkdir()
    _seed_prunable(seed_path)
    return _copy(seed_path, tmp_path / "legacy" / "state.db"), _copy(seed_path, store_path)


def _auto_maintenance(legacy, store, store_path: Path) -> tuple[dict, dict]:
    """The legacy pass on ``legacy`` and the store's pass on ``store``; the freelist floor is disabled so only a
    holder can stop the VACUUM."""
    expected = legacy.maybe_auto_prune_and_vacuum(retention_days=90, min_vacuum_freelist_ratio=0.0)
    result = run_auto_prune_and_vacuum(
        store, SQLiteStoreAdmin(store_path, read_only=False), store_home=store_path.parent, media_root=MEDIA_ROOT,
        retention_days=90, min_vacuum_freelist_ratio=0.0)
    return expected, result


def _without_ratio(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "freelist_ratio"}


@pytest.mark.parametrize("sibling", [False, True], ids=["alone", "beside-a-sibling-handle"])
def test_auto_maintenance_vacuums_through_the_writer_it_holds_like_the_legacy_pass(tmp_path, backend, scope_a,
                                                                                   sibling, legacy_store):
    import hermes_state_registry

    legacy_path, store_path = _prunable_pair(tmp_path, store_for(scope_a))
    legacy = hermes_state_registry.acquire(legacy_path)
    siblings = (hermes_state_registry.acquire(legacy_path), _registry.acquire(store_path)) if sibling else ()
    try:
        expected, result = _auto_maintenance(legacy, backend.open(scope_a), store_path)
    finally:
        hermes_state_registry.release(legacy)
        if siblings:
            hermes_state_registry.release(siblings[0])
            _registry.release(siblings[1])

    assert expected["pruned"] == 1 and expected["vacuumed"] is True and "vacuum_skipped_holders" not in expected
    assert _without_ratio(result) == _without_ratio(expected)
    assert _pragma(store_path, "freelist_count") == _pragma(legacy_path, "freelist_count") == 0


_HOLDER = """
import sqlite3, sys
connections = [sqlite3.connect(path) for path in sys.argv[1:]]
for conn in connections:
    conn.execute("SELECT count(*) FROM sessions").fetchall()
print("ready", flush=True)
sys.stdin.read(1)
"""


@pytest.mark.platforms("linux")
def test_auto_maintenance_skips_vacuum_while_another_process_holds_the_store_like_the_legacy_pass(
        tmp_path, backend, scope_a, legacy_store):
    import hermes_state_registry

    legacy_path, store_path = _prunable_pair(tmp_path, store_for(scope_a))
    legacy = hermes_state_registry.acquire(legacy_path)
    holder = subprocess.Popen([sys.executable, "-c", _HOLDER, str(legacy_path), str(store_path)],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert select.select([holder.stdout], [], [], 10)[0] and holder.stdout.readline().strip() == "ready"
        expected, result = _auto_maintenance(legacy, backend.open(scope_a), store_path)
    finally:
        holder.stdin.write("x")
        holder.stdin.close()
        holder.wait(timeout=10)
        hermes_state_registry.release(legacy)

    assert expected["pruned"] == 1 and expected["vacuumed"] is False and expected["vacuum_skipped_holders"] >= 1
    assert _without_ratio(result) == _without_ratio(expected)


def test_auto_maintenance_lock_excludes_the_legacy_pass_until_released(tmp_path, legacy_store):
    from hermes_state import SessionDB

    legacy_path, admin_path = _copies(tmp_path, _seed_prunable)
    admin = SQLiteStoreAdmin(legacy_path, read_only=True)

    handle = admin.try_acquire_auto_maintenance_lock()
    assert handle is not None
    try:
        assert _try_acquire_auto_maintenance_lock(legacy_path) is None
        assert admin.try_acquire_auto_maintenance_lock() is None
        db = SessionDB(db_path=legacy_path)
        try:
            assert db.maybe_auto_prune_and_vacuum(retention_days=90)["skipped"] is True
        finally:
            db.close()
    finally:
        admin.release_auto_maintenance_lock(handle)

    legacy_handle = _try_acquire_auto_maintenance_lock(legacy_path)
    assert legacy_handle is not None
    try:
        assert admin.try_acquire_auto_maintenance_lock() is None
    finally:
        _release_auto_maintenance_lock(legacy_handle)
    other = SQLiteStoreAdmin(admin_path)
    other_handle = other.try_acquire_auto_maintenance_lock()
    assert other_handle is not None
    other.release_auto_maintenance_lock(other_handle)


# ── FTS storage rewrite ────────────────────────────────────────────────────


def _event_key(event: dict) -> tuple:
    return tuple(sorted((k, v) for k, v in event.items() if k != "elapsed"))


def _schema(path: Path) -> list:
    return _rows(path, "SELECT type, name, sql FROM sqlite_master")


def test_optimize_fts_storage_rewrites_a_legacy_layout_like_the_legacy_store(tmp_path, legacy_store):
    from hermes_state import SessionDB
    from hermes_state_common import FTS_STORAGE_VERSION

    legacy_path, admin_path = _copies(tmp_path, _legacy_inline_store)
    pages_before = _pragma(legacy_path, "page_count")
    legacy_events, admin_events = [], []

    db = SessionDB(db_path=legacy_path)
    try:
        assert db.fts_optimize_available() is True
        legacy = db.optimize_fts_storage(progress_cb=legacy_events.append, vacuum=True)
        legacy_after = db.fts_optimize_available()
    finally:
        db.close()
    admin = SQLiteStoreAdmin(admin_path, read_only=False)
    assert admin.fts_optimize_available() is True
    result = admin.optimize_fts_storage(progress_cb=admin_events.append, vacuum=True)

    assert result == legacy == {"ok": True, "vacuumed": True}
    assert admin.fts_optimize_available() is legacy_after is False
    assert [_event_key(e) for e in admin_events] == [_event_key(e) for e in legacy_events]
    phases = [e["phase"] for e in legacy_events]
    assert {"backfill", "teardown", "vacuum"} <= set(phases) and phases[-1] == "done"
    assert _schema(admin_path) == _schema(legacy_path)
    # Every key, and the value of every FTS key (the others identify each file and its first open).
    meta = "SELECT key, CASE WHEN key LIKE 'fts%' THEN value END FROM state_meta"
    assert _rows(admin_path, meta) == _rows(legacy_path, meta)
    assert ("fts_storage_version", str(FTS_STORAGE_VERSION)) in _rows(admin_path, meta)
    match = "SELECT rowid FROM messages_fts WHERE messages_fts MATCH 'zebra'"
    assert _rows(admin_path, match) == _rows(legacy_path, match)
    assert len(_rows(admin_path, match)) == 1200
    for path in (legacy_path, admin_path):
        assert _pragma(path, "page_count") < pages_before
        assert not _rows(path, "SELECT name FROM sqlite_master WHERE name LIKE 'fts_v22_trash_%'")


def test_optimize_fts_storage_without_vacuum_matches_and_reports_nothing_vacuumed(tmp_path, legacy_store):
    from hermes_state import SessionDB

    legacy_path, admin_path = _copies(tmp_path, _legacy_inline_store)
    db = SessionDB(db_path=legacy_path)
    try:
        legacy = db.optimize_fts_storage(vacuum=False)
    finally:
        db.close()
    result = SQLiteStoreAdmin(admin_path, read_only=False).optimize_fts_storage(vacuum=False)

    assert result == legacy == {"ok": True, "vacuumed": None}
    assert _schema(admin_path) == _schema(legacy_path)


def test_read_only_optimize_is_refused_like_the_legacy_store(tmp_path, legacy_store):
    from hermes_state import SessionDB

    legacy_path, admin_path = _copies(tmp_path, _legacy_inline_store)
    before = {path: path.read_bytes() for path in (legacy_path, admin_path)}
    db = SessionDB(db_path=legacy_path, read_only=True)
    try:
        legacy = (db.fts_optimize_available(), db.optimize_fts_storage())
    finally:
        db.close()
    admin = SQLiteStoreAdmin(admin_path, read_only=True)

    assert (admin.fts_optimize_available(), admin.optimize_fts_storage()) == legacy
    assert legacy == (False, {"ok": False, "reason": "read_only"})
    assert {path: path.read_bytes() for path in before} == before


# ── FTS layout probe ───────────────────────────────────────────────────────


def _legacy_drift(path: Path) -> list[str]:
    """The update notice's probe: a read-only store open, then the layout check on its connection."""
    from hermes_state import SessionDB
    from hermes_state_fts import fts_layout_drift

    db = SessionDB(db_path=path, read_only=True)
    try:
        return fts_layout_drift(db._conn)
    finally:
        db.close()


@pytest.mark.parametrize("read_only", [True, False], ids=["read-only", "writable"])
def test_fts_layout_drift_reports_a_legacy_inline_layout(tmp_path, read_only, legacy_store):
    legacy_path, admin_path = _copies(tmp_path, _legacy_inline_store)
    before = admin_path.read_bytes()

    drift = SQLiteStoreAdmin(admin_path, read_only=read_only).fts_layout_drift()

    assert drift == _legacy_drift(legacy_path) == ["messages_fts", "messages_fts_trigram"]
    assert admin_path.read_bytes() == before


def test_fts_layout_drift_is_empty_on_a_current_layout(tmp_path, legacy_store):
    legacy_path, admin_path = _copies(tmp_path, _current_store)
    assert SQLiteStoreAdmin(admin_path).fts_layout_drift() == _legacy_drift(legacy_path) == []


@pytest.mark.parametrize("read_only", [True, False], ids=["read-only", "writable"])
def test_fts_layout_drift_of_a_missing_file_is_empty_and_creates_nothing(tmp_path, read_only, legacy_store):
    path = tmp_path / "absent" / "state.db"
    with pytest.raises(sqlite3.OperationalError):
        _legacy_drift(path)
    assert SQLiteStoreAdmin(path, read_only=read_only).fts_layout_drift() == []
    assert not path.exists()



# ── FTS storage notice ─────────────────────────────────────────────────────


def _with_meta(key: str):
    def seed(path: Path) -> None:
        from hermes_state import SessionDB

        _current_store(path)
        with SessionDB(db_path=path) as db:
            db.set_meta(key, "1")
    return seed


def _with_trash(path: Path) -> None:
    _current_store(path)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE fts_v22_trash_1 (x)")
    conn.close()


def _legacy_notice(path: Path, monkeypatch, capsys) -> str:
    """Main's update-command notice for the store at ``path``, with the file grown sparse past its size floor."""
    with open(path, "r+b") as handle:
        handle.truncate(600 * 1024 * 1024)
    monkeypatch.setenv("HERMES_HOME", str(path.parent))
    capsys.readouterr()
    _print_fts_optimize_available_notice()
    return capsys.readouterr().out


@pytest.mark.parametrize("seed", [_current_store, _legacy_inline_store, _with_meta("fts_rebuild_high_water"),
                                  _with_meta("fts_cjk_rebuild_high_water"), _with_trash],
                         ids=["current", "legacy-inline", "backfill-marker", "cjk-marker", "trash-table"])
def test_fts_storage_notice_reads_what_the_update_notice_decides_on(tmp_path, monkeypatch, capsys, seed,
                                                                    legacy_store):
    legacy_path, admin_path = _copies(tmp_path, seed)
    before = admin_path.read_bytes()

    notice = SQLiteStoreAdmin(admin_path).fts_storage_notice()
    printed = _legacy_notice(legacy_path, monkeypatch, capsys)

    assert admin_path.read_bytes() == before
    assert notice["drift"] == _legacy_drift(legacy_path)
    assert (printed == "") is (not notice["drift"] and not notice["interrupted"])
    assert ("optimization incomplete" in printed) is notice["interrupted"]
    assert ("old storage layout" in printed) is (notice["legacy_inline"] and not notice["interrupted"])
    if seed is _current_store:
        assert notice == {"drift": [], "legacy_inline": False, "interrupted": False}
    elif seed is _legacy_inline_store:
        assert notice["legacy_inline"] is True and notice["drift"]
    else:
        assert notice["interrupted"] is True


def test_fts_storage_notice_of_a_missing_file_raises_and_creates_nothing(tmp_path):
    from persistence import PersistenceError

    path = tmp_path / "absent" / "state.db"
    with pytest.raises(PersistenceError):
        SQLiteStoreAdmin(path).fts_storage_notice()
    assert not path.exists()
