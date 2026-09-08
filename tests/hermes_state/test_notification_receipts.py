"""Native notification transaction foundation; no provider or board worker."""
import sqlite3

import pytest

from hermes_state import SessionDB
from hermes_state_errors import SessionTurnLeaseLostError


@pytest.fixture
def delivery(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("origin", source="tui")
    assert db.try_acquire_session_turn_lease("origin", "owner")
    yield db
    db.close()


def envelope(db, **changes):
    from hermes_state_errors import _STATE_DB_GENERATION_KEY
    with db._read_ctx() as conn:
        generation = conn.execute("SELECT value FROM state_meta WHERE key = ?",
                                  (_STATE_DB_GENERATION_KEY,)).fetchone()[0]
    identity = dict(store_path=str(db.db_path), store_generation=generation,
                    board_path=str(db.db_path.parent / "board.db"), board_generation="board-gen",
                    task_id="task", event_id=2, origin_session_id="origin", platform="tui",
                    thread_id=None, subscription_generation="subscription-gen")
    return identity | changes


def admit(db, identity=None, holder="owner"):
    return db.append_notification_once("origin", identity=identity or envelope(db),
                                       content="terminal result", turn_lease_holder=holder)


def test_atomic_receipt_reopen_and_compaction(delivery):
    db = delivery
    assert callable(getattr(db, "append_notification_once", None)), "native atomic admission missing"
    message_id = admit(db)
    assert admit(db) == message_id
    assert len(db.get_messages("origin")) == 1
    assert db.get_session("origin")["message_count"] == 1
    db.archive_and_compact("origin", [{"role": "system", "content": "summary"}])
    db.close()
    reopened = SessionDB(db.db_path)
    try:
        assert admit(reopened) == message_id
        assert len(reopened.get_messages("origin")) == 1
        with reopened._read_ctx() as conn:
            assert conn.execute("SELECT count(*) FROM notification_receipts").fetchone()[0] == 1
    finally:
        reopened.close()


@pytest.mark.parametrize("failure_table", ["messages", "notification_receipts"])
def test_transaction_rollback(delivery, failure_table):
    db = delivery
    assert callable(getattr(db, "append_notification_once", None)), "native atomic admission missing"
    # Deterministic SQLite abort at either insert, including AFTER message insert.
    db._execute_write(lambda conn: conn.execute(
        f"CREATE TEMP TRIGGER fail_insert BEFORE INSERT ON {failure_table} "
        "BEGIN SELECT RAISE(ABORT, 'fixture abort'); END"))
    with pytest.raises(sqlite3.IntegrityError, match="fixture abort"):
        admit(db)
    assert db.get_messages("origin") == []
    assert db.get_session("origin")["message_count"] == 0
    with db._read_ctx() as conn:
        assert conn.execute("SELECT count(*) FROM notification_receipts").fetchone()[0] == 0
    db._execute_write(lambda conn: conn.execute("DROP TRIGGER fail_insert"))
    assert admit(db) > 0


@pytest.mark.parametrize("holder", [None, "", "competing", "expired", "unknown"])
def test_ownership_refused_without_mutating_lease(delivery, holder):
    db = delivery
    assert callable(getattr(db, "append_notification_once", None)), "native atomic admission missing"
    if holder == "expired":
        db._execute_write(lambda conn: conn.execute(
            "UPDATE session_turn_leases SET holder = 'expired', expires_at = 0"))
    if holder == "unknown":
        db.release_session_turn_lease("origin", "owner")
    with db._read_ctx() as conn:
        before = [tuple(row) for row in conn.execute("SELECT * FROM session_turn_leases")]
    with pytest.raises(SessionTurnLeaseLostError):
        admit(db, holder=holder)
    assert db.get_messages("origin") == []
    with db._read_ctx() as conn:
        assert [tuple(row) for row in conn.execute("SELECT * FROM session_turn_leases")] == before


@pytest.mark.parametrize("field,value", [("store_generation", "replacement"),
    ("store_path", "/different/state.db"), ("origin_session_id", "other"),
    ("platform", "telegram"), ("thread_id", "other-thread"), ("board_generation", ""),
    ("event_id", True), ("event_id", 0)])
def test_unknown_or_wrong_identity_refused(delivery, field, value):
    assert callable(getattr(delivery, "append_notification_once", None)), "native atomic admission missing"
    with pytest.raises(ValueError):
        admit(delivery, envelope(delivery, **{field: value}))
    assert delivery.get_messages("origin") == []


def test_exact_identity_and_board_alias(delivery):
    db = delivery
    assert callable(getattr(db, "append_notification_once", None)), "native atomic admission missing"
    first = admit(db)
    alias = db.db_path.parent / "alias"
    alias.symlink_to(db.db_path.parent, target_is_directory=True)
    assert admit(db, envelope(db, board_path=str(alias / "board.db"))) == first
    for change in ({"subscription_generation": "new"}, {"board_generation": "new"},
                   {"event_id": 3}, {"task_id": "another"}):
        assert admit(db, envelope(db, **change)) != first
    assert len(db.get_messages("origin")) == 5


@pytest.mark.parametrize("boundary", ["before_commit", "after_commit"])
def test_process_death_and_fresh_runtime_retry(tmp_path, boundary):
    import json
    import os
    import subprocess
    import sys

    path = tmp_path / "crash.db"
    db = SessionDB(path)
    db.create_session("origin", source="tui")
    identity = envelope(db)
    db.close()
    code = '''
import json, os, sys
from pathlib import Path
from hermes_state import SessionDB
path, boundary, raw = sys.argv[1:]
db = SessionDB(Path(path))
assert db.try_acquire_session_turn_lease("origin", "owner")
if boundary == "before_commit":
    db._conn.create_function("fixture_die", 0, lambda: os._exit(73))
    db._execute_write(lambda c: c.execute("CREATE TEMP TRIGGER die BEFORE INSERT ON notification_receipts BEGIN SELECT fixture_die(); END"))
db.append_notification_once("origin", identity=json.loads(raw), content="terminal result", turn_lease_holder="owner")
os._exit(74)
'''
    result = subprocess.run([sys.executable, "-c", code, str(path), boundary, json.dumps(identity)],
                            env=dict(os.environ), capture_output=True, text=True, timeout=20)
    assert result.returncode == (73 if boundary == "before_commit" else 74), result.stderr
    reopened = SessionDB(path)
    try:
        assert len(reopened.get_messages("origin")) == (0 if boundary == "before_commit" else 1)
        message_id = admit(reopened, identity)
        reopened.close()
        fresh = SessionDB(path)
        try:
            assert admit(fresh, identity) == message_id
            assert len(fresh.get_messages("origin")) == 1
            assert fresh.get_session("origin")["message_count"] == 1
        finally:
            fresh.close()
    finally:
        reopened.close()


def test_additive_migration_repeat_and_store_separation(delivery, tmp_path):
    db = delivery
    # Equivalent old native schema: the additive receipt table does not exist.
    db._execute_write(lambda c: c.execute("DROP TABLE notification_receipts"))
    db.close()
    first = SessionDB(db.db_path)
    try:
        message_id = admit(first)
        first.close()
        second = SessionDB(db.db_path)
        try:
            assert admit(second) == message_id
            assert len(second.get_messages("origin")) == 1
            other = SessionDB(tmp_path / "other-profile" / "state.db")
            try:
                other.create_session("origin", source="tui")
                assert other.try_acquire_session_turn_lease("origin", "owner")
                with pytest.raises(ValueError):
                    admit(other, envelope(second))
                assert other.get_messages("origin") == []
                assert admit(other) > 0
            finally:
                other.close()
        finally:
            second.close()
    finally:
        first.close()


@pytest.mark.parametrize("guard_time", [100.5, 101.0, 102.0])
def test_notification_strict_expiry_boundary(delivery, monkeypatch, guard_time):
    from types import SimpleNamespace
    import hermes_state_notifications as notifications
    import hermes_state_messages as messages

    db = delivery
    db._execute_write(lambda c: c.execute("UPDATE session_turn_leases SET expires_at=101"))
    with db._read_ctx() as c:
        lease_before = [tuple(r) for r in c.execute("SELECT * FROM session_turn_leases")]
    counters_before = db.get_session("origin")
    monkeypatch.setattr(notifications, "time", SimpleNamespace(time=lambda: 100.0))
    monkeypatch.setattr(messages, "time", SimpleNamespace(time=lambda: guard_time))
    if guard_time < 101:
        first = admit(db)
        assert admit(db) == first
        assert len(db.get_messages("origin")) == 1
        assert db.get_session("origin")["message_count"] == counters_before["message_count"] + 1
    else:
        with pytest.raises(SessionTurnLeaseLostError):
            admit(db)
        assert db.get_messages("origin") == []
        assert db.get_session("origin") == counters_before
        assert not db._conn.in_transaction
    with db._read_ctx() as c:
        assert [tuple(r) for r in c.execute("SELECT * FROM session_turn_leases")] == lease_before
        assert c.execute("SELECT count(*) FROM notification_receipts").fetchone()[0] == (1 if guard_time < 101 else 0)


def test_notification_closed_compression_parent_refused(delivery):
    from hermes_state_errors import CompressionSessionClosedError
    db = delivery
    db.end_session("origin", "compression")
    with db._read_ctx() as c:
        before = [tuple(r) for r in c.execute("SELECT * FROM session_turn_leases")]
    counters_before = db.get_session("origin")
    with pytest.raises(CompressionSessionClosedError):
        admit(db)
    assert db.get_messages("origin") == []
    assert db.get_session("origin") == counters_before
    with db._read_ctx() as c:
        assert [tuple(r) for r in c.execute("SELECT * FROM session_turn_leases")] == before
        assert c.execute("SELECT count(*) FROM notification_receipts").fetchone()[0] == 0


def test_concurrent_handles_share_one_receipt(delivery):
    from concurrent.futures import ThreadPoolExecutor
    db = delivery
    other = SessionDB(db.db_path)
    try:
        with ThreadPoolExecutor(max_workers=2) as workers:
            results = list(workers.map(admit, [db, other]))
        assert results[0] == results[1]
        assert len(db.get_messages("origin")) == 1
    finally:
        other.close()
