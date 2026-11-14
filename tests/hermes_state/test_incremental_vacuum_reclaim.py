"""Incremental space reclaim on a live state.db (``reclaim_free_pages_incrementally``).

A full VACUUM holds the write lock for the whole rewrite, so it can only run with the store quiet.
``auto_vacuum=INCREMENTAL`` lets freed pages go back to the disk in short batches beside live
writers. New stores are created INCREMENTAL; an existing store switches on its next full VACUUM
(``hermes sessions optimize``), which is the only operation SQLite allows to change the mode.
"""

from __future__ import annotations

import sqlite3
import threading

from hermes_state import SessionDB


def _fill_and_prune(db: SessionDB, rows: int = 60, sid: str = "old") -> None:
    db.create_session(sid, "cli")
    for _ in range(rows):
        db.append_message(sid, "user", "x" * 20_000)
    db.end_session(sid, "done")
    assert db.prune_sessions(older_than_days=0) == 1


def _pragma(db: SessionDB, name: str) -> int:
    return db._conn.execute(f"PRAGMA {name}").fetchone()[0]


def _legacy_store(path) -> None:
    """A store created before this change: auto_vacuum NONE."""
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE legacy_seed(x)")
    conn.commit()
    conn.close()


def test_a_new_store_is_created_incremental(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        assert _pragma(db, "auto_vacuum") == 2
    finally:
        db.close()


def test_a_new_store_is_silent_about_the_journal_mode_switch(tmp_path, monkeypatch, caplog):
    """The auto_vacuum pragma writes page 1 before WAL is enabled. A store that was empty a moment
    earlier made no journal-mode choice, so the #89293 "was delete, switched to WAL" WARNING must
    not fire on it, and the pragma must still take."""
    import hermes_state_wal
    monkeypatch.setattr(hermes_state_wal, "is_sqlite_wal_reset_vulnerable", lambda **kwargs: False)
    monkeypatch.setattr(hermes_state_wal, "_journal_upgrade_warned_paths", set())
    with caplog.at_level("WARNING", logger="hermes_state"):
        db = SessionDB(db_path=tmp_path / "state.db")
    try:
        assert _pragma(db, "auto_vacuum") == 2
        assert db._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert [r.getMessage() for r in caplog.records if r.levelno >= 30] == []
    finally:
        db.close()


def test_reclaim_returns_free_pages_and_keeps_the_store_intact(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("keep", "cli")
        db.append_message("keep", "user", "survives")
        _fill_and_prune(db)
        pages, free = _pragma(db, "page_count"), _pragma(db, "freelist_count")
        assert free > 256

        result = db.reclaim_free_pages_incrementally(batch_pages=64)

        assert result["incremental"] is True and "error" not in result
        assert result["batches"] > 1, "one batch drained everything: the pass is not bounded"
        assert result["reclaimed_pages"] >= free - 64
        assert _pragma(db, "page_count") <= pages - result["reclaimed_pages"]
        assert _pragma(db, "integrity_check") == "ok"
        assert [m["content"] for m in db.get_messages("keep")] == ["survives"]
    finally:
        db.close()


def test_a_writer_lands_between_batches(tmp_path):
    """Each batch is its own short transaction: a second connection's write commits while the pass
    is still running (the next batch waits for it). Holding the lock across batches deadlocks the
    writer here, and the bounded wait below turns that into a failure."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        _fill_and_prune(db)
        first_batch_done, landed = threading.Event(), threading.Event()
        seen_before_second = []
        real_execute_write = db._execute_write

        def _execute_write(fn, *args, **kwargs):
            if first_batch_done.is_set() and not seen_before_second:
                seen_before_second.append(landed.wait(10))
            out = real_execute_write(fn, *args, **kwargs)
            first_batch_done.set()
            return out

        def _write_between_batches():
            assert first_batch_done.wait(10)
            other = sqlite3.connect(tmp_path / "state.db", timeout=5)
            other.execute("INSERT INTO state_meta(key, value) VALUES ('probe', 'x')")
            other.commit()
            other.close()
            landed.set()

        db._execute_write = _execute_write
        writer = threading.Thread(target=_write_between_batches)
        writer.start()
        result = db.reclaim_free_pages_incrementally(batch_pages=16, pause_s=0)
        writer.join(10)
        assert seen_before_second == [True], "the write could not land between two batches"
        assert "error" not in result and result["batches"] > 1
        assert db.get_meta("probe") == "x"
    finally:
        db.close()


def test_a_legacy_store_is_left_alone_with_one_hint(tmp_path, caplog):
    path = tmp_path / "state.db"
    _legacy_store(path)
    db = SessionDB(db_path=path)
    try:
        assert _pragma(db, "auto_vacuum") == 0
        _fill_and_prune(db)
        pages = _pragma(db, "page_count")
        with caplog.at_level("INFO", logger="hermes_state"):
            first = db.reclaim_free_pages_incrementally()
            second = db.reclaim_free_pages_incrementally()
        assert first["incremental"] is False and first["reclaimed_pages"] == 0
        assert second["reclaimed_pages"] == 0
        assert _pragma(db, "page_count") == pages
        hints = [r for r in caplog.records if "hermes sessions optimize" in r.getMessage()]
        assert len(hints) == 1
    finally:
        db.close()


def test_a_full_vacuum_switches_a_legacy_store_to_incremental(tmp_path):
    """``hermes sessions optimize`` (SessionDB.vacuum) is the one-time conversion."""
    path = tmp_path / "state.db"
    _legacy_store(path)
    db = SessionDB(db_path=path)
    try:
        _fill_and_prune(db)
        db.vacuum()
        assert _pragma(db, "auto_vacuum") == 2
        assert _pragma(db, "integrity_check") == "ok"
        # From now on a later prune is reclaimed without a stop.
        _fill_and_prune(db, sid="old2")
    finally:
        db.close()


def test_gateway_flag_prunes_without_the_full_rewrite(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("old", "cli")
        db.end_session("old", "done")
        calls = []
        monkeypatch.setattr(db, "vacuum", lambda: calls.append(True))
        result = db.maybe_auto_prune_and_vacuum(
            retention_days=0, min_interval_hours=0, min_vacuum_interval_days=0,
            min_vacuum_freelist_ratio=-1.0, full_vacuum=False)
        assert result["pruned"] == 1 and result["vacuumed"] is False
        assert calls == []
    finally:
        db.close()
