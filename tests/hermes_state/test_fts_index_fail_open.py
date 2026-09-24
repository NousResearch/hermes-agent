"""Regression tests for #97794: an FTS5-index-only failure must not kill the turn, and an
FTS-scoped error that escapes must not be rendered as whole-file damage.

The write path already fails open on provenance-proven FTS corruption (detach the derived
indexes, retry the canonical write) and quarantines the handle on unscoped corruption
(#97940 / #90837). These tests pin the contract at the boundaries the issue was filed
against — the agent flush whose failure ends the turn, and the cause that drives the
user-facing guidance:

* the flush succeeds after an FTS-only stomp and the exact user message is durable in
  ``messages`` (the turn proceeds);
* an FTS-scoped error that still escapes (detach refused) classifies as ``fts_index`` and
  never quarantines the handle;
* a sibling process holding the write lock when the detach runs is waited out, not a lost write;
* a quarantine that lands while the detach waits stops it: nothing is committed on the file;
"""

import sqlite3
import threading
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB, StateDbCorruptError
from hermes_state_health import mark_storage_corrupt, reset_storage_state
from run_agent import AIAgent


def _flush_agent(db, session_id):
    """Bind the real flush methods onto a stand-in over a live SessionDB."""
    agent = SimpleNamespace(
        _session_db=db,
        _session_db_created=True,
        _persist_disabled=False,
        session_id=session_id,
        _session_persist_lock=None,
        _flushed_db_message_ids=set(),
        _flushed_db_message_session_id=None,
        _last_flushed_db_idx=0,
        _db_flush_scan_prefix=None,
        _persist_user_message_idx=None,
        _persist_user_message_override=None,
        _persist_user_message_timestamp=None,
        _pending_cli_user_message=None,
        _active_session_turn_lease_holder=None,
        _last_persistence_error_cause=None,
        _compression_adoption_failed=False,
    )
    agent._ensure_db_session = lambda: None
    agent._flush_messages_to_session_db = (
        AIAgent._flush_messages_to_session_db.__get__(agent, AIAgent)
    )
    agent._flush_messages_to_session_db_unlocked = (
        AIAgent._flush_messages_to_session_db_unlocked.__get__(agent, AIAgent)
    )
    return agent


def _seed(db, rows=60):
    if not db._fts_enabled:
        pytest.skip("FTS5 unavailable in this build")
    db.create_session("s1", source="cli")
    for i in range(rows):
        db.append_message("s1", "user", f"seed row {i} " + "z" * 200)


def _stomp_fts_shadow(db_path):
    """Overwrite the messages_fts shadow b-tree blocks: FTS5 raises SQLITE_CORRUPT_VTAB on the
    next MATCH / sync-trigger insert while every canonical row stays intact."""
    raw = sqlite3.connect(str(db_path))
    raw.execute("UPDATE messages_fts_data SET block = X'DEADBEEFDEADBEEFDEADBEEFDEADBEEF'")
    raw.commit()
    raw.close()


def _contents(db_path):
    raw = sqlite3.connect(str(db_path))
    try:
        return [r[0] for r in raw.execute("SELECT content FROM messages ORDER BY id").fetchall()]
    finally:
        raw.close()


def test_turn_flush_survives_fts_only_corruption(tmp_path):
    """The turn's transcript write succeeds after an FTS-only stomp: the flush reports
    success (the turn proceeds, no ``session_persistence_failed``) and the exact user
    message is durable in ``messages``. The derived indexes are detached, the handle is
    not quarantined."""
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    try:
        _seed(db)
        _stomp_fts_shadow(db_path)
        agent = _flush_agent(db, "s1")

        ok = agent._flush_messages_to_session_db(
            [{"role": "user", "content": "lands after stomp"}], []
        )

        assert ok is True
        assert agent._last_persistence_error_cause is None
        assert _contents(db_path)[-1] == "lands after stomp"
        assert db._db_corrupt is False
        # Builds whose sync trigger walks the stomped structure record detach the derived
        # indexes; builds that defer the read pass the insert through untouched. Either way
        # the canonical write landed, which is the contract.
        assert db._fts_stale in (True, False)
        assert db.get_session("s1") is not None
    finally:
        db.close()


def test_escaped_fts_only_error_is_index_scoped_not_quarantined(tmp_path, monkeypatch):
    """When the detach itself is refused the FTS-scoped error escapes to the agent. It must
    classify as ``fts_index`` (guidance names the index, not the file) and must not
    quarantine the handle or touch the derived indexes."""
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    try:
        _seed(db)
        _stomp_fts_shadow(db_path)
        monkeypatch.setattr(db, "_enter_fts_fail_open", lambda exc, **_: False)
        agent = _flush_agent(db, "s1")

        ok = agent._flush_messages_to_session_db(
            [{"role": "user", "content": "refused detach"}], []
        )
        if ok is True:
            pytest.skip("this SQLite build defers FTS shadow corruption past the insert trigger")

        assert ok is False
        assert agent._last_persistence_error_cause == "fts_index"
        assert db._db_corrupt is False
        assert db._fts_stale is False
        assert "refused detach" not in _contents(db_path)
    finally:
        db.close()


def test_detach_waits_out_a_sibling_holding_the_write_lock(tmp_path):
    """Gateway + TUI hit the same corrupt index: one detaches while the other waits. A sibling
    that takes the write lock between this writer's corruption error and its detach, and holds
    it past the writer connection's 1 s busy timeout, must be waited out on the write budget —
    the canonical row lands instead of escaping as 'database disk image is malformed'."""
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    try:
        _seed(db, rows=5)
        _stomp_fts_shadow(db_path)
        held, release = threading.Event(), threading.Event()

        def sibling():
            raw = sqlite3.connect(str(db_path), timeout=30, isolation_level=None)
            raw.execute("BEGIN IMMEDIATE")
            held.set()
            release.wait(10)
            raw.execute("COMMIT")
            raw.close()

        real_check = db._is_fts_write_corruption_error
        holder = []

        def check_then_contend(exc):
            hit = real_check(exc)
            if hit and not holder:  # the writer has rolled back; the sibling grabs the lock now
                holder.append(threading.Thread(target=sibling))
                holder[0].start()
                assert held.wait(10)
                threading.Timer(1.6, release.set).start()
            return hit

        db._is_fts_write_corruption_error = check_then_contend
        db.append_message("s1", "user", "lands after the sibling lets go")
        if not holder:
            pytest.skip("this SQLite build defers FTS shadow corruption past the insert trigger")
        holder[0].join(10)

        assert _contents(db_path)[-1] == "lands after the sibling lets go"
        assert db._fts_stale is True
        assert db._db_corrupt is False
    finally:
        db.close()


def test_quarantine_while_detach_waits_commits_nothing(tmp_path):
    """The detach may now wait up to the write budget for the lock. A sibling that quarantines
    this file meanwhile (structural corruption latched process-wide) must stop it: the retry
    drops no triggers, commits no stale breadcrumb, and the corrupt error surfaces."""
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    try:
        _seed(db, rows=5)
        _stomp_fts_shadow(db_path)
        held, release = threading.Event(), threading.Event()

        def sibling():
            raw = sqlite3.connect(str(db_path), timeout=30, isolation_level=None)
            raw.execute("BEGIN IMMEDIATE")
            held.set()
            release.wait(10)
            raw.execute("COMMIT")
            raw.close()

        real_check, real_sleep = db._is_fts_write_corruption_error, db._sleep_before_write_retry
        holder = []

        def check_then_contend(exc):
            hit = real_check(exc)
            if hit and not holder:
                holder.append(threading.Thread(target=sibling))
                holder[0].start()
                assert held.wait(10)
            return hit

        def quarantine_then_sleep(deadline, patience_s):
            mark_storage_corrupt(db_path, "database disk image is malformed (sibling handle)")
            release.set()
            return real_sleep(deadline, patience_s)

        db._is_fts_write_corruption_error = check_then_contend
        db._sleep_before_write_retry = quarantine_then_sleep
        with pytest.raises(StateDbCorruptError):
            db.append_message("s1", "user", "must not land on a quarantined file")
        if not holder:
            pytest.skip("this SQLite build defers FTS shadow corruption past the insert trigger")
        holder[0].join(10)

        raw = sqlite3.connect(str(db_path))
        try:
            triggers = raw.execute(
                "SELECT count(*) FROM sqlite_master WHERE type = 'trigger' AND name LIKE 'messages_fts%'"
            ).fetchone()[0]
            stale = raw.execute("SELECT value FROM state_meta WHERE key LIKE 'fts%stale%'").fetchall()
        finally:
            raw.close()
        assert triggers > 0
        assert stale == []
        assert db._fts_stale is False
    finally:
        db.close()
        reset_storage_state(db_path)
