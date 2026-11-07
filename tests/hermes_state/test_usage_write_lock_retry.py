"""A SQLite writer lock timeout leaves accepted token deltas retryable."""

import sqlite3

import pytest

from hermes_state import SessionDB


@pytest.fixture
def database(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    db = SessionDB(home / "state.db")
    yield db
    try:
        db.close()
    except TimeoutError:
        pass


@pytest.fixture
def delete_database(tmp_path, monkeypatch):
    home = tmp_path / "hermes-delete"
    home.mkdir()
    (home / "config.yaml").write_text("database:\n  journal_mode: delete\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    db = SessionDB(home / "state.db")
    assert not db._wal_active
    yield db
    try:
        db.close()
    except TimeoutError:
        pass


def test_begin_immediate_lock_timeout_requeues_unapplied_accounting(database, monkeypatch):
    db = database
    expected = {"one": 3, "two": 7}
    for session_id in expected:
        db.create_session(session_id, "cli")

    monkeypatch.setattr(db, "_token_writer_loop", lambda: None)
    for session_id, tokens in expected.items():
        db.queue_token_counts(
            session_id, input_tokens=tokens, api_call_count=1,
            model="route-model", billing_provider="route-provider",
        )
        db._token_writer_thread.join(5)
    assert db._token_writer_thread is not None and not db._token_writer_thread.is_alive()

    db._TRANSCRIPT_WRITE_PATIENCE_S = 0.05
    db._conn.execute("PRAGMA busy_timeout=0")
    callback_calls = []
    original_execute_write = db._execute_write

    def observe_callback(callback, *args, **kwargs):
        def observed(conn):
            callback_calls.append(True)
            return callback(conn)
        return original_execute_write(observed, *args, **kwargs)

    monkeypatch.setattr(db, "_execute_write", observe_callback)
    sibling = sqlite3.connect(db.db_path, timeout=0, isolation_level=None)
    try:
        sibling.execute("BEGIN IMMEDIATE")
        first = db.flush_token_counts()
        first_state = {
            "callbacks": len(callback_calls),
            "uncertain": [sid for sid, _ in db._token_writer_uncertain],
            "queued": [sid for sid, _ in db._token_queue],
            "marker_held": db._token_pending_marker is not None,
            "has_pending": db.has_pending_token_counts(),
            "counts": {
                sid: db.get_session(sid)["input_tokens"]
                for sid in expected
            },
        }
    finally:
        if sibling.in_transaction:
            sibling.execute("ROLLBACK")
        sibling.close()

    retry = db.flush_token_counts()
    route_rows = [
        tuple(row) for row in db._read_all(
            "SELECT session_id, model, billing_provider, input_tokens, api_call_count "
            "FROM session_model_usage ORDER BY session_id"
        )
    ]
    counts = {sid: db.get_session(sid)["input_tokens"] for sid in expected}
    queue_empty = not db._token_queue and not db._token_writer_uncertain and not db._token_writer_busy
    pending_clear = not db.has_pending_token_counts()
    try:
        db.close()
        closed = db._conn is None
    except TimeoutError:
        closed = False
    assert {
        "first_flush": first,
        "first_state": first_state,
        "retry_flush": retry,
        "callbacks": len(callback_calls),
        "counts": counts,
        "route_rows": route_rows,
        "queue_empty": queue_empty,
        "pending_clear": pending_clear,
        "closed": closed,
    } == {
        "first_flush": False,
        "first_state": {
            "callbacks": 0,
            "uncertain": [],
            "queued": ["one", "two"],
            "marker_held": True,
            "has_pending": True,
            "counts": {"one": 0, "two": 0},
        },
        "retry_flush": True,
        "callbacks": 2,
        "counts": expected,
        "route_rows": [
            ("one", "route-model", "route-provider", 3, 1),
            ("two", "route-model", "route-provider", 7, 1),
        ],
        "queue_empty": True,
        "pending_clear": True,
        "closed": True,
    }



def test_commit_lock_timeout_retries_only_after_real_rollback(delete_database, monkeypatch):
    db = delete_database
    db._conn.execute("PRAGMA busy_timeout=0")
    expected = {"one": 3, "two": 7}
    for session_id in expected:
        db.create_session(session_id, "cli")

    monkeypatch.setattr(db, "_token_writer_loop", lambda: None)
    for session_id, tokens in expected.items():
        db.queue_token_counts(
            session_id, input_tokens=tokens, api_call_count=1,
            model="route-model", billing_provider="route-provider",
        )
        db._token_writer_thread.join(5)

    db._TRANSCRIPT_WRITE_PATIENCE_S = 0.05
    sibling = sqlite3.connect(db.db_path, timeout=0, isolation_level=None)
    sibling.execute("BEGIN")
    sibling.execute("SELECT id FROM sessions ORDER BY id").fetchall()

    callback_calls = []
    commit_failures = []
    inner = db._conn

    class CommitObserver:
        def __getattr__(self, name):
            return getattr(inner, name)

        def commit(self):
            try:
                return inner.commit()
            except sqlite3.OperationalError as exc:
                commit_failures.append((exc.sqlite_errorcode, inner.in_transaction))
                raise

    db._conn = CommitObserver()
    original_execute_write = db._execute_write

    def observe_callback(callback, *args, **kwargs):
        def observed(conn):
            callback_calls.append(True)
            return callback(conn)
        return original_execute_write(observed, *args, **kwargs)

    monkeypatch.setattr(db, "_execute_write", observe_callback)
    try:
        first = db.flush_token_counts()
        first_state = {
            "callbacks": len(callback_calls),
            "commit_failures": list(commit_failures),
            "uncertain": [sid for sid, _ in db._token_writer_uncertain],
            "queued": [sid for sid, _ in db._token_queue],
            "counts": {sid: inner.execute(
                "SELECT input_tokens FROM sessions WHERE id=?", (sid,)
            ).fetchone()[0] for sid in expected},
            "transaction_open": inner.in_transaction,
        }
    finally:
        sibling.execute("ROLLBACK")
        sibling.close()
        db._conn = inner

    retry = db.flush_token_counts()
    counts = {sid: db.get_session(sid)["input_tokens"] for sid in expected}
    route_rows = [
        tuple(row) for row in db._read_all(
            "SELECT session_id, model, billing_provider, input_tokens, api_call_count "
            "FROM session_model_usage ORDER BY session_id"
        )
    ]
    queue_empty = not db._token_queue and not db._token_writer_uncertain and not db._token_writer_busy
    pending_clear = not db.has_pending_token_counts()
    try:
        db.close()
        closed = db._conn is None
    except TimeoutError:
        closed = False

    assert {
        "first_flush": first,
        "first_state": first_state,
        "retry_flush": retry,
        "callbacks_after_retry": len(callback_calls),
        "counts": counts,
        "route_rows": route_rows,
        "queue_empty": queue_empty,
        "pending_clear": pending_clear,
        "closed": closed,
    } == {
        "first_flush": False,
        "first_state": {
            "callbacks": first_state["callbacks"],
            "commit_failures": first_state["commit_failures"],
            "uncertain": [],
            "queued": ["one", "two"],
            "counts": {"one": 0, "two": 0},
            "transaction_open": False,
        },
        "retry_flush": True,
        "callbacks_after_retry": first_state["callbacks"] + 2,
        "counts": expected,
        "route_rows": [
            ("one", "route-model", "route-provider", 3, 1),
            ("two", "route-model", "route-provider", 7, 1),
        ],
        "queue_empty": True,
        "pending_clear": True,
        "closed": True,
    }
    assert first_state["callbacks"] > 0
    assert first_state["commit_failures"]
    assert all(
        code is not None and code & 0xFF == sqlite3.SQLITE_BUSY and was_active
        for code, was_active in first_state["commit_failures"]
    )




def test_postcommit_lineage_refusal_stays_uncertain(database, monkeypatch):
    from hermes_state_lineage import SessionLineageError

    db = database
    db.create_session("committed", "cli")
    monkeypatch.setattr(db, "_token_writer_loop", lambda: None)
    db.queue_token_counts("committed", input_tokens=3)
    db._token_writer_thread.join(5)
    update = db.update_token_counts

    def commit_then_refuse(session_id, **kwargs):
        update(session_id, **kwargs)
        raise SessionLineageError("post-commit refusal")

    monkeypatch.setattr(db, "update_token_counts", commit_then_refuse)
    assert db.flush_token_counts() is False
    assert db.get_session("committed")["input_tokens"] == 3
    assert [sid for sid, _ in db._token_writer_uncertain] == ["committed"]
    assert not db._token_queue
    assert db._token_pending_marker is not None
    assert db.flush_token_counts() is False
    assert db.get_session("committed")["input_tokens"] == 3


def test_saved_timeout_rethrown_after_commit_stays_uncertain(database, monkeypatch):
    db = database
    db.create_session("committed", "cli")
    db._TRANSCRIPT_WRITE_PATIENCE_S = 0.01
    db._conn.execute("PRAGMA busy_timeout=0")
    monkeypatch.setattr(db, "_token_writer_loop", lambda: None)
    db.queue_token_counts("committed", input_tokens=3)
    db._token_writer_thread.join(5)
    update = db.update_token_counts
    sibling = sqlite3.connect(db.db_path, timeout=0, isolation_level=None)

    def timeout_then_commit(session_id, **kwargs):
        sibling.execute("BEGIN IMMEDIATE")
        try:
            with pytest.raises(sqlite3.OperationalError) as failed:
                db._execute_write(lambda conn: None, patience_s=0.01)
        finally:
            sibling.execute("ROLLBACK")
        update(session_id, **kwargs)
        raise failed.value

    monkeypatch.setattr(db, "update_token_counts", timeout_then_commit)
    try:
        assert db.flush_token_counts() is False
        assert db.get_session("committed")["input_tokens"] == 3
        assert [sid for sid, _ in db._token_writer_uncertain] == ["committed"]
        assert not db._token_queue
        assert db._token_pending_marker is not None
    finally:
        sibling.close()


@pytest.mark.parametrize("rollback_mode", ["raises", "no_op"])
def test_failed_rollback_is_quarantined(delete_database, monkeypatch, rollback_mode):
    db = delete_database
    db.create_session("uncertain", "cli")
    db._conn.execute("PRAGMA busy_timeout=0")
    db._TRANSCRIPT_WRITE_PATIENCE_S = 0
    monkeypatch.setattr(db, "_token_writer_loop", lambda: None)
    db.queue_token_counts("uncertain", input_tokens=3)
    db._token_writer_thread.join(5)
    sibling = sqlite3.connect(db.db_path, timeout=0, isolation_level=None)
    sibling.execute("BEGIN")
    sibling.execute("SELECT * FROM sessions").fetchall()
    inner = db._conn
    calls = []

    class BrokenRollback:
        def __getattr__(self, name):
            return getattr(inner, name)

        def rollback(self):
            calls.append("rollback")
            if rollback_mode == "raises":
                raise sqlite3.OperationalError("injected rollback failure")

    db._conn = BrokenRollback()
    try:
        assert db.flush_token_counts() is False
        assert calls == ["rollback"]
        assert inner.in_transaction
        assert [sid for sid, _ in db._token_writer_uncertain] == ["uncertain"]
        assert not db._token_queue
        assert db._token_pending_marker is not None
        assert db.flush_token_counts() is False
    finally:
        sibling.execute("ROLLBACK")
        sibling.close()
        inner.rollback()
        db._conn = inner


def test_real_commit_then_raise_stays_uncertain(database, monkeypatch):
    db = database
    db.create_session("uncertain", "cli")
    monkeypatch.setattr(db, "_token_writer_loop", lambda: None)
    db.queue_token_counts("uncertain", input_tokens=3)
    db._token_writer_thread.join(5)
    inner = db._conn
    commits = []

    class CommitThenRaise:
        def __getattr__(self, name):
            return getattr(inner, name)

        def commit(self):
            inner.commit()
            commits.append(True)
            raise sqlite3.OperationalError("injected failure after commit")

    db._conn = CommitThenRaise()
    try:
        assert db.flush_token_counts() is False
        assert len(commits) == 1
        assert inner.execute("SELECT input_tokens FROM sessions WHERE id='uncertain'").fetchone()[0] == 3
        assert [sid for sid, _ in db._token_writer_uncertain] == ["uncertain"]
        assert db._token_pending_marker is not None
        assert db.flush_token_counts() is False
        with pytest.raises(TimeoutError):
            db.close()
        assert len(commits) == 1
    finally:
        db._conn = inner


def test_committed_write_then_second_begin_timeout_stays_uncertain(database, monkeypatch):
    db = database
    db.create_session("uncertain", "cli")
    db._conn.execute("PRAGMA busy_timeout=0")
    monkeypatch.setattr(db, "_token_writer_loop", lambda: None)
    db.queue_token_counts("uncertain", input_tokens=3)
    db._token_writer_thread.join(5)
    update = db.update_token_counts
    sibling = sqlite3.connect(db.db_path, timeout=0, isolation_level=None)

    def first_commits_second_fails(session_id, **kwargs):
        update(session_id, **kwargs)
        sibling.execute("BEGIN IMMEDIATE")
        try:
            db._execute_write(lambda conn: None, patience_s=0.01)
        finally:
            sibling.execute("ROLLBACK")

    monkeypatch.setattr(db, "update_token_counts", first_commits_second_fails)
    try:
        assert db.flush_token_counts() is False
        assert db.get_session("uncertain")["input_tokens"] == 3
        assert [sid for sid, _ in db._token_writer_uncertain] == ["uncertain"]
        assert not db._token_queue
        assert db.flush_token_counts() is False
        assert db.get_session("uncertain")["input_tokens"] == 3
    finally:
        sibling.close()


@pytest.mark.parametrize("failed_probe", [1, 2])
def test_failed_transaction_state_probe_keeps_original_error_and_uncertainty(delete_database, monkeypatch, failed_probe):
    from hermes_state_usage import _TokenBatchInterrupted

    db = delete_database
    db.create_session("probe", "cli")
    db._conn.execute("PRAGMA busy_timeout=0")
    db._TRANSCRIPT_WRITE_PATIENCE_S = 0
    sibling = sqlite3.connect(db.db_path, timeout=0, isolation_level=None)
    sibling.execute("BEGIN")
    sibling.execute("SELECT * FROM sessions").fetchall()
    inner = db._conn
    probes = []

    class ProbeFailure:
        def __getattr__(self, name):
            return getattr(inner, name)

        @property
        def in_transaction(self):
            probes.append(True)
            if len(probes) == failed_probe:
                raise RuntimeError("injected state probe failure")
            return inner.in_transaction

    db._conn = ProbeFailure()
    try:
        with pytest.raises(_TokenBatchInterrupted) as failure:
            db._apply_token_batch([("probe", {"input_tokens": 3})])
        assert isinstance(failure.value.cause, sqlite3.OperationalError)
        assert failure.value.cause.sqlite_errorcode == sqlite3.SQLITE_BUSY
        assert [sid for sid, _ in failure.value.uncertain] == ["probe"]
        assert not inner.in_transaction
    finally:
        sibling.execute("ROLLBACK")
        sibling.close()
        db._conn = inner
