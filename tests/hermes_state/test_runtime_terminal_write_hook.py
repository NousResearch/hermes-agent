"""Transactional contract for the private runtime terminal-write callback."""
import json
import sqlite3

import pytest

from hermes_state import SessionDB
from hermes_state_terminal import RESULT_PREFIX
import hermes_state_runtime as rt


class CallbackFailure(RuntimeError):
    pass


@pytest.fixture
def runtime(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s", source="test")
    epoch = rt.begin_runtime_epoch(db, instance_id="owner")
    try:
        yield db, epoch
    finally:
        db.close()


def _admit(db, epoch, request_id):
    return rt.admit_session_input(
        db,
        epoch=epoch,
        principal_id="human",
        session_id="s",
        request_id=request_id,
        payload={"text": request_id},
    )


def _prepare_transition(db, epoch, transition):
    admitted = _admit(db, epoch, transition)
    result = None
    worker_id = None
    if transition == "cancel":
        row = admitted
        expected_outcome = "cancelled"
        kwargs = {"epoch": epoch, "admission_id": row["admission_id"]}
    else:
        row = rt.claim_session_input(db, epoch=epoch, session_id="s")
        worker_id = "worker-" + transition
        rt.register_worker_execution(
            db,
            epoch=epoch,
            execution_id=worker_id,
            session_id="s",
            generation=row["generation"],
            kind="compute",
            adoption_secret="private",
        )
        if transition == "settle":
            result = {
                "result": {"final_response": "done", "messages": []},
                "usage": {"input_tokens": 1},
            }
            expected_outcome = "completed"
            kwargs = {
                "epoch": epoch,
                "admission_id": row["admission_id"],
                "generation": row["generation"],
                "outcome": expected_outcome,
                "result": result,
            }
        else:
            epoch = rt.begin_runtime_epoch(db, instance_id="replacement")
            assert rt.recover_session_inputs(db, epoch=epoch) == 1
            row = rt.get_session_admission(db, admission_id=row["admission_id"])
            expected_outcome = "interrupted"
            kwargs = {
                "epoch": epoch,
                "admission_id": row["admission_id"],
                "generation": row["generation"],
            }
    return row, kwargs, expected_outcome, result, worker_id


def _invoke(transition, db, kwargs, callback):
    if transition == "settle":
        return rt.settle_session_input(db, **kwargs, _terminal_write=callback)
    if transition == "cancel":
        return rt.cancel_session_input(db, **kwargs, _terminal_write=callback)
    return rt.resolve_unknown_session_input(db, **kwargs, _terminal_write=callback)


@pytest.mark.parametrize("transition", ["settle", "cancel", "resolve"])
def test_terminal_write_commits_metadata_with_each_terminal_transition(runtime, transition):
    db, epoch = runtime
    before, kwargs, expected_outcome, expected_result, worker_id = _prepare_transition(
        db, epoch, transition
    )
    seen = []
    key = "test.terminal_write.success." + transition

    def terminal_write(conn, row, outcome, result):
        seen.append((conn, row, outcome, result))
        assert conn.in_transaction
        conn.execute(
            "INSERT INTO state_meta(key,value) VALUES(?,?)",
            (key, json.dumps({"admission_id": row["admission_id"], "outcome": outcome})),
        )

    settled = _invoke(transition, db, kwargs, terminal_write)

    assert len(seen) == 1
    conn, row, outcome, result = seen[0]
    assert conn is db._conn
    assert row == before
    assert outcome == expected_outcome
    assert result is expected_result
    assert (settled["status"], settled["outcome"]) == ("terminal", expected_outcome)
    with sqlite3.connect(db.db_path) as peer:
        saved = peer.execute("SELECT value FROM state_meta WHERE key=?", (key,)).fetchone()
        assert json.loads(saved[0]) == {
            "admission_id": before["admission_id"],
            "outcome": expected_outcome,
        }
        if expected_result is not None:
            result_row = peer.execute(
                "SELECT value FROM state_meta WHERE key=?",
                (RESULT_PREFIX + before["admission_id"],),
            ).fetchone()
            assert json.loads(result_row[0]) == expected_result
        if worker_id is not None:
            assert peer.execute(
                "SELECT status FROM worker_executions WHERE execution_id=?", (worker_id,)
            ).fetchone()[0] == "terminal"


@pytest.mark.parametrize("transition", ["settle", "cancel", "resolve"])
def test_terminal_write_failure_rolls_back_the_entire_transition(runtime, transition):
    db, epoch = runtime
    before, kwargs, _expected_outcome, _expected_result, worker_id = _prepare_transition(
        db, epoch, transition
    )
    key = "test.terminal_write.rollback." + transition
    session_before = db.get_session("s")
    worker_before = (
        db._read_one(
            "SELECT status FROM worker_executions WHERE execution_id=?", (worker_id,)
        )[0]
        if worker_id is not None
        else None
    )
    calls = 0

    def terminal_write(conn, row, outcome, result):
        nonlocal calls
        calls += 1
        conn.execute("INSERT INTO state_meta(key,value) VALUES(?,?)", (key, outcome))
        raise CallbackFailure(transition)

    with pytest.raises(CallbackFailure, match=transition):
        _invoke(transition, db, kwargs, terminal_write)

    assert calls == 1
    assert rt.get_session_admission(db, admission_id=before["admission_id"]) == before
    session_after = db.get_session("s")
    assert session_after["runtime_generation"] == session_before["runtime_generation"]
    assert session_after["runtime_revision"] == session_before["runtime_revision"]
    assert db._read_one("SELECT value FROM state_meta WHERE key=?", (key,)) is None
    assert db._read_one(
        "SELECT value FROM state_meta WHERE key=?",
        (RESULT_PREFIX + before["admission_id"],),
    ) is None
    if worker_id is not None:
        assert db._read_one(
            "SELECT status FROM worker_executions WHERE execution_id=?", (worker_id,)
        )[0] == worker_before


def test_terminal_write_is_never_called_before_terminal_fences(runtime):
    db, epoch = runtime
    calls = []

    def terminal_write(*args):
        calls.append(args)

    queued = _admit(db, epoch, "queued")
    with pytest.raises(rt.RuntimeStoreError, match="invalid_params"):
        rt.settle_session_input(
            db,
            epoch=epoch,
            admission_id=queued["admission_id"],
            generation=0,
            outcome="not-an-outcome",
            _terminal_write=terminal_write,
        )
    for invoke in (
        lambda: rt.settle_session_input(
            db,
            epoch=epoch,
            admission_id="missing",
            generation=0,
            outcome="completed",
            _terminal_write=terminal_write,
        ),
        lambda: rt.cancel_session_input(
            db, epoch=epoch, admission_id="missing", _terminal_write=terminal_write
        ),
        lambda: rt.resolve_unknown_session_input(
            db,
            epoch=epoch,
            admission_id="missing",
            generation=0,
            _terminal_write=terminal_write,
        ),
    ):
        with pytest.raises(rt.RuntimeStoreError, match="not_found"):
            invoke()
    with pytest.raises(rt.RuntimeStoreError, match="stale_epoch"):
        rt.cancel_session_input(
            db,
            epoch=epoch + 1,
            admission_id=queued["admission_id"],
            _terminal_write=terminal_write,
        )
    with pytest.raises(rt.RuntimeStoreError, match="stale_generation"):
        rt.settle_session_input(
            db,
            epoch=epoch,
            admission_id=queued["admission_id"],
            generation=0,
            outcome="completed",
            _terminal_write=terminal_write,
        )
    with pytest.raises(rt.RuntimeStoreError, match="stale_generation"):
        rt.resolve_unknown_session_input(
            db,
            epoch=epoch,
            admission_id=queued["admission_id"],
            generation=0,
            _terminal_write=terminal_write,
        )

    started = rt.claim_session_input(db, epoch=epoch, session_id="s")
    with pytest.raises(rt.RuntimeStoreError, match="stale_generation"):
        rt.settle_session_input(
            db,
            epoch=epoch,
            admission_id=started["admission_id"],
            generation=started["generation"] + 1,
            outcome="completed",
            _terminal_write=terminal_write,
        )
    with pytest.raises(rt.RuntimeStoreError, match="stale_generation"):
        rt.cancel_session_input(
            db,
            epoch=epoch,
            admission_id=started["admission_id"],
            _terminal_write=terminal_write,
        )
    assert calls == []


@pytest.mark.parametrize("transition", ["settle", "cancel", "resolve"])
def test_none_keeps_the_existing_terminal_transition_behavior(runtime, transition):
    db, epoch = runtime
    _before, kwargs, expected_outcome, _result, _worker_id = _prepare_transition(
        db, epoch, transition
    )
    settled = _invoke(transition, db, kwargs, None)
    assert (settled["status"], settled["outcome"]) == ("terminal", expected_outcome)


def test_cancel_replay_does_not_invoke_or_rewrite_terminal_metadata(runtime):
    db, epoch = runtime
    admitted = _admit(db, epoch, "cancel-replay")
    first = rt.cancel_session_input(db, epoch=epoch, admission_id=admitted["admission_id"])
    session_before = db.get_session("s")
    calls = []

    replayed = rt.cancel_session_input(
        db,
        epoch=epoch,
        admission_id=admitted["admission_id"],
        _terminal_write=lambda *args: calls.append(args),
    )

    assert replayed == first
    assert calls == []
    session_after = db.get_session("s")
    assert session_after["runtime_generation"] == session_before["runtime_generation"]
    assert session_after["runtime_revision"] == session_before["runtime_revision"]


def test_terminal_write_retry_replays_callback_but_commits_its_sql_once(runtime):
    db, epoch = runtime
    admitted = _admit(db, epoch, "retry")
    started = rt.claim_session_input(db, epoch=epoch, session_id="s")
    key = "test.terminal_write.retry"
    attempts = 0
    peer = sqlite3.connect(db.db_path, timeout=0, isolation_level=None)

    def terminal_write(conn, row, outcome, result):
        nonlocal attempts
        attempts += 1
        conn.execute(
            "INSERT INTO state_meta(key,value) VALUES(?,'1') "
            "ON CONFLICT(key) DO UPDATE SET value=CAST(value AS INTEGER)+1",
            (key,),
        )
        if attempts == 1:
            # This second connection cannot acquire a writer lock while the callback's
            # transaction is active. Its real SQLITE_BUSY makes SessionDB roll back and
            # retry the whole callback; no timing loop or daemon is involved.
            peer.execute("BEGIN IMMEDIATE")

    try:
        settled = rt.settle_session_input(
            db,
            epoch=epoch,
            admission_id=admitted["admission_id"],
            generation=started["generation"],
            outcome="completed",
            _terminal_write=terminal_write,
        )
    finally:
        peer.close()

    assert settled["status"] == "terminal"
    assert attempts == 2
    assert db._read_one("SELECT value FROM state_meta WHERE key=?", (key,))[0] == "1"
