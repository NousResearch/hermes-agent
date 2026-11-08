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



def test_two_begin_ioerrs_before_callback_requeue_without_uncertainty(database, monkeypatch):
    db = database
    expected = {"one": 3, "two": 7}
    for session_id in expected:
        db.create_session(session_id, "cli")

    monkeypatch.setattr(db, "_token_writer_loop", lambda: None)
    for session_id, tokens in expected.items():
        db.queue_token_counts(session_id, input_tokens=tokens, api_call_count=1)
        db._token_writer_thread.join(5)

    inner = db._conn
    begin_errors, callbacks = [], []

    class BeginFault:
        def __getattr__(self, name):
            return getattr(inner, name)

        def execute(self, sql, *args, **kwargs):
            if sql == "BEGIN IMMEDIATE" and len(begin_errors) < 2:
                error = sqlite3.OperationalError("disk I/O error before callback")
                error.sqlite_errorcode = sqlite3.SQLITE_IOERR
                begin_errors.append(error.sqlite_errorcode)
                raise error
            return inner.execute(sql, *args, **kwargs)

    db._conn = BeginFault()
    original_execute_write = db._execute_write

    def observe_callback(callback, *args, **kwargs):
        def observed(conn):
            callbacks.append(True)
            return callback(conn)
        return original_execute_write(observed, *args, **kwargs)

    monkeypatch.setattr(db, "_execute_write", observe_callback)
    first_flush = db.flush_token_counts()
    first_state = {
        "callbacks": len(callbacks),
        "begin_errors": list(begin_errors),
        "rows": {sid: inner.execute(
            "SELECT input_tokens FROM sessions WHERE id=?", (sid,)
        ).fetchone()[0] for sid in expected},
        "transaction_open": inner.in_transaction,
        "uncertain": [sid for sid, _ in db._token_writer_uncertain],
        "queued": [sid for sid, _ in db._token_queue],
    }

    db._conn = inner
    second_flush = db.flush_token_counts()
    rows = {sid: inner.execute(
        "SELECT input_tokens FROM sessions WHERE id=?", (sid,)
    ).fetchone()[0] for sid in expected}
    usage_rows = [tuple(row) for row in db._read_all(
        "SELECT session_id, input_tokens, api_call_count FROM session_model_usage ORDER BY session_id"
    )]

    assert first_flush is False
    assert first_state == {
        "callbacks": 0,
        "begin_errors": [sqlite3.SQLITE_IOERR, sqlite3.SQLITE_IOERR],
        "rows": {"one": 0, "two": 0},
        "transaction_open": False,
        "uncertain": [],
        "queued": ["one", "two"],
    }
    assert second_flush is True
    assert callbacks == [True, True]
    assert rows == expected
    assert usage_rows == [("one", 3, 1), ("two", 7, 1)]
    assert not db._token_queue and not db._token_writer_uncertain
    assert not db.has_pending_token_counts()


@pytest.mark.parametrize("state_mode", ["probe-error", "active"])
def test_begin_ioerr_with_unknown_transaction_state_is_not_retried(
    database, monkeypatch, state_mode,
):
    db = database
    session_id = "unknown-begin"
    db.create_session(session_id, "cli")
    monkeypatch.setattr(db, "_token_writer_loop", lambda: None)
    db.queue_token_counts(session_id, input_tokens=3)
    db._token_writer_thread.join(5)

    inner = db._conn
    begin_attempts, callback_calls = [], []

    class UnknownBeginState:
        def __getattr__(self, name):
            return getattr(inner, name)

        @property
        def in_transaction(self):
            if begin_attempts and state_mode == "probe-error":
                raise RuntimeError("injected transaction state probe failure")
            if begin_attempts and state_mode == "active":
                return True
            return inner.in_transaction

        def execute(self, sql, *args, **kwargs):
            if sql == "BEGIN IMMEDIATE":
                begin_attempts.append(sql)
                if len(begin_attempts) == 1:
                    error = sqlite3.OperationalError("disk I/O error at BEGIN")
                    error.sqlite_errorcode = sqlite3.SQLITE_IOERR
                    raise error
            return inner.execute(sql, *args, **kwargs)

    db._conn = UnknownBeginState()
    original_execute_write = db._execute_write

    def observe_callback(callback, *args, **kwargs):
        def observed(conn):
            callback_calls.append(True)
            return callback(conn)
        return original_execute_write(observed, *args, **kwargs)

    monkeypatch.setattr(db, "_execute_write", observe_callback)
    first_flush = db.flush_token_counts()
    db._conn = inner
    row_after_first = inner.execute(
        "SELECT input_tokens FROM sessions WHERE id=?", (session_id,)
    ).fetchone()[0]
    first_uncertain = [sid for sid, _ in db._token_writer_uncertain]
    first_queued = [sid for sid, _ in db._token_queue]
    second_flush = db.flush_token_counts()
    row_after_second = inner.execute(
        "SELECT input_tokens FROM sessions WHERE id=?", (session_id,)
    ).fetchone()[0]

    assert first_flush is False
    assert len(begin_attempts) == 1
    assert callback_calls == []
    assert row_after_first == row_after_second == 0
    assert first_uncertain == [session_id]
    assert first_queued == []
    assert second_flush is False
    assert [sid for sid, _ in db._token_writer_uncertain] == [session_id]


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




def test_sqlite_full_after_verified_rollback_retries_once(database, monkeypatch):
    db = database
    session_id = "full-session"
    db.create_session(session_id, "cli")
    monkeypatch.setattr(db, "_token_writer_loop", lambda: None)
    db.queue_token_counts(
        session_id, input_tokens=5, api_call_count=1,
        model="route-model", billing_provider="route-provider",
    )
    db._token_writer_thread.join(5)
    assert db._token_writer_thread is not None and not db._token_writer_thread.is_alive()

    full = sqlite3.OperationalError("database or disk is full")
    full.sqlite_errorcode = sqlite3.SQLITE_FULL
    original_execute_write = db._execute_write
    inject = [True]

    def inject_full_after_callback(callback, *args, **kwargs):
        def callback_then_full(conn):
            result = callback(conn)
            if inject[0]:
                inject[0] = False
                raise full
            return result
        return original_execute_write(callback_then_full, *args, **kwargs)

    monkeypatch.setattr(db, "_execute_write", inject_full_after_callback)
    first_flush = db.flush_token_counts()
    uri = db.db_path.resolve().as_uri() + "?mode=ro"
    reader = sqlite3.connect(uri, uri=True, timeout=0)
    try:
        reader.execute("PRAGMA query_only = ON")
        row_after_first = reader.execute(
            "SELECT input_tokens FROM sessions WHERE id=?", (session_id,)
        ).fetchone()[0]
    finally:
        reader.close()
    first_state = {
        "transaction_open": db._conn.in_transaction,
        "uncertain": [sid for sid, _ in db._token_writer_uncertain],
        "queued": [sid for sid, _ in db._token_queue],
    }

    monkeypatch.setattr(db, "_execute_write", original_execute_write)
    second_flush = db.flush_token_counts()
    uri = db.db_path.resolve().as_uri() + "?mode=ro"
    reader = sqlite3.connect(uri, uri=True, timeout=0)
    try:
        reader.execute("PRAGMA query_only = ON")
        row_after_retry = reader.execute(
            "SELECT input_tokens FROM sessions WHERE id=?", (session_id,)
        ).fetchone()[0]
        usage_rows = [tuple(row) for row in reader.execute(
            "SELECT session_id, input_tokens, api_call_count FROM session_model_usage"
        )]
    finally:
        reader.close()

    assert {
        "sqlite_errorcode": full.sqlite_errorcode,
        "first_flush": first_flush,
        "row_after_first": row_after_first,
        "transaction_open_after_first": first_state["transaction_open"],
        "uncertain_after_first": first_state["uncertain"],
        "queued_after_first": first_state["queued"],
        "second_flush": second_flush,
        "row_after_retry": row_after_retry,
        "usage_rows": usage_rows,
        "uncertain_after_retry": [sid for sid, _ in db._token_writer_uncertain],
        "queued_after_retry": [sid for sid, _ in db._token_queue],
    } == {
        "sqlite_errorcode": sqlite3.SQLITE_FULL,
        "first_flush": False,
        "row_after_first": 0,
        "transaction_open_after_first": False,
        "uncertain_after_first": [],
        "queued_after_first": [session_id],
        "second_flush": True,
        "row_after_retry": 5,
        "usage_rows": [(session_id, 5, 1)],
        "uncertain_after_retry": [],
        "queued_after_retry": [],
    }


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


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("point", ["rollback", "before_probe", "after_probe"])
def test_commit_busy_cleanup_interrupt_propagates_and_quarantines(delete_database, monkeypatch, interrupt, point):
    db = delete_database
    for sid in ("current", "suffix"):
        db.create_session(sid, "cli")
    db._conn.execute("PRAGMA busy_timeout=0")
    db._TRANSCRIPT_WRITE_PATIENCE_S = 0
    monkeypatch.setattr(db, "_token_writer_loop", lambda: None)
    for sid, tokens in (("current", 3), ("suffix", 7)):
        db.queue_token_counts(sid, input_tokens=tokens)
        db._token_writer_thread.join(5)
    sibling = sqlite3.connect(db.db_path, timeout=0, isolation_level=None)
    sibling.execute("BEGIN")
    sibling.execute("SELECT * FROM sessions").fetchall()
    inner = db._conn
    commit_errors, probes = [], []

    class InterruptedCleanup:
        def __getattr__(self, name):
            return getattr(inner, name)

        @property
        def in_transaction(self):
            probes.append(True)
            if point == ("before_probe" if len(probes) == 1 else "after_probe"):
                raise interrupt("cleanup interrupted")
            return inner.in_transaction

        def commit(self):
            try:
                return inner.commit()
            except sqlite3.OperationalError as exc:
                commit_errors.append(exc.sqlite_errorcode)
                raise

        def rollback(self):
            if point == "rollback":
                raise interrupt("cleanup interrupted")
            return inner.rollback()

    db._conn = InterruptedCleanup()
    try:
        with pytest.raises(interrupt, match="cleanup interrupted"):
            db.flush_token_counts()
        assert commit_errors == [sqlite3.SQLITE_BUSY]
        assert [sid for sid, _ in db._token_writer_uncertain] == ["current"]
        assert [sid for sid, _ in db._token_queue] == ["suffix"]
        assert db._token_pending_marker is not None
        assert db.flush_token_counts() is False
        with pytest.raises(TimeoutError):
            db.close()
        assert commit_errors == [sqlite3.SQLITE_BUSY]
    finally:
        sibling.execute("ROLLBACK")
        sibling.close()
        inner.rollback()
        db._conn = inner


def _queue_for_deadline(db, monkeypatch):
    db.create_session("deadline", "cli")
    monkeypatch.setattr(db, "_token_writer_loop", lambda: None)
    db.queue_token_counts("deadline", input_tokens=3, model="m", billing_provider="p")
    db._token_writer_thread.join(5)
    db._TRANSCRIPT_WRITE_PATIENCE_S = 0.5


def _hold_write_lock(db):
    import threading
    ready = threading.Event()
    release = threading.Event()
    def holder():
        sibling = sqlite3.connect(db.db_path, timeout=0, isolation_level=None)
        try:
            sibling.execute("BEGIN IMMEDIATE")
            ready.set()
            release.wait(0.4)
            sibling.execute("ROLLBACK")
        finally:
            sibling.close()
    thread = threading.Thread(target=holder)
    thread.start()
    assert ready.wait(2)
    return release, thread


def test_async_session_getter_does_not_drain_queued_accounting(database, monkeypatch):
    import asyncio
    import time

    db = database
    _queue_for_deadline(db, monkeypatch)
    release, holder = _hold_write_lock(db)
    async def read():
        loop = asyncio.get_running_loop()
        beat = asyncio.Event()
        beat_at = []
        loop.call_later(0.01, lambda: (beat_at.append(time.monotonic()), beat.set()))
        start = time.monotonic()
        result = db.get_session("deadline")
        returned = time.monotonic()
        await beat.wait()
        return result, returned - start, beat_at[0] - start
    try:
        result, elapsed, beat_delay = asyncio.run(read())
        assert elapsed < 0.2 and beat_delay < 0.2
        assert result["input_tokens"] == 0
        assert [sid for sid, _ in db._token_queue] == ["deadline"]
        assert not db._token_writer_uncertain and db._token_pending_marker is not None
    finally:
        release.set()
        holder.join(2)
    assert db.flush_token_counts()
    assert db.get_session("deadline")["input_tokens"] == 3
    db.update_session_billing_route("deadline", provider="p", base_url="u", billing_mode="m")
    db.close()


def test_positive_flush_budget_bounds_sqlite_wait(database, monkeypatch):
    import time

    db = database
    _queue_for_deadline(db, monkeypatch)
    release, holder = _hold_write_lock(db)
    try:
        start = time.monotonic()
        result = db.flush_token_counts(timeout=0.03)
        elapsed = time.monotonic() - start
        assert result is False and elapsed < 0.2
        assert [sid for sid, _ in db._token_queue] == ["deadline"]
        assert not db._token_writer_uncertain and db._token_pending_marker is not None
        assert getattr(db._token_batch_write_context, "deadline", None) is None
        assert db._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 1000
        assert db._TRANSCRIPT_WRITE_PATIENCE_S == 0.5
        assert db._conn.execute("SELECT input_tokens FROM sessions WHERE id='deadline'").fetchone()[0] == 0
    finally:
        release.set()
        holder.join(2)
    assert db.flush_token_counts()
    assert db.get_session("deadline")["input_tokens"] == 3


@pytest.mark.parametrize("lock_kind", ["lineage", "advisory", "connection", "queue"])
def test_positive_flush_budget_bounds_process_locks(database, monkeypatch, lock_kind):
    import threading
    import time
    from hermes_state_lineage import (
        session_lineage_guard, _advisory_lock_file, _managed_root_and_current, _lineage_lock_paths,
    )

    db = database
    _queue_for_deadline(db, monkeypatch)
    ready = threading.Event()
    release = threading.Event()
    def hold():
        if lock_kind == "lineage":
            guard = session_lineage_guard(db.db_path, "deadline")
        elif lock_kind == "advisory":
            root, _ = _managed_root_and_current(db.db_path)
            guard = _advisory_lock_file(_lineage_lock_paths(root, ("deadline",))[0], 2)
        elif lock_kind == "connection":
            guard = db._lock
        else:
            guard = db._token_queue_cond
        with guard:
            ready.set()
            release.wait(0.4)
    holder = threading.Thread(target=hold)
    holder.start()
    assert ready.wait(2)
    try:
        start = time.monotonic()
        assert not db.flush_token_counts(timeout=0.03)
        assert time.monotonic() - start < 0.2
        assert [sid for sid, _ in db._token_queue] == ["deadline"]
        assert not db._token_writer_uncertain and db._token_pending_marker is not None
    finally:
        release.set()
        holder.join(2)
    assert db.flush_token_counts()
    assert db.get_session("deadline")["input_tokens"] == 3


@pytest.mark.parametrize("timeout", [0, -1])
def test_poll_does_not_claim_even_without_sqlite_contention(database, monkeypatch, timeout):
    db = database
    _queue_for_deadline(db, monkeypatch)
    assert not db.flush_token_counts(timeout=timeout)
    assert [sid for sid, _ in db._token_queue] == ["deadline"]
    assert not db._token_writer_uncertain
    assert db._conn.execute("SELECT input_tokens FROM sessions WHERE id='deadline'").fetchone()[0] == 0
    assert db.flush_token_counts()
    assert db.get_session("deadline")["input_tokens"] == 3
