"""Accounting barriers observe settlement, not only queue emptiness (PR #149132)."""
import atexit
import threading
import time
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = SessionDB(tmp_path / "state.db")
    store.create_session("s", "cli")
    yield store
    store._stop_token_writer(5)
    # Dispose deliberately retained invalid/uncertain fixture work, never replay it.
    with store._token_queue_cond:
        assert not store._token_writer_busy
        store._token_queue.clear()
        store._token_writer_uncertain.clear()
        store._release_idle_token_marker()
    if store._token_atexit_hook:
        atexit.unregister(store._token_atexit_hook)
    store.close()


def _retain(db, monkeypatch, **values):
    with monkeypatch.context() as paused:
        paused.setattr(db, "_token_writer_loop", lambda: None)
        db.queue_token_counts("s", **values)
        db._token_writer_thread.join(5)
        assert not db._token_writer_thread.is_alive()


def _totals(db):
    return (
        db._read_one("SELECT input_tokens FROM sessions WHERE id='s'")[0],
        db._read_one("SELECT coalesce(sum(input_tokens), 0) FROM session_model_usage WHERE session_id='s'")[0],
    )


def test_waiting_flush_fails_after_last_item_commit_ack_is_lost(db, monkeypatch):
    entered, release, waiting = (threading.Event() for _ in range(3))
    results, errors = [], []
    conn = db._conn

    class LostAckConnection:
        def __getattr__(self, name):
            return getattr(conn, name)

        def commit(self):
            entered.set()
            assert release.wait(10), "commit was never released"
            conn.commit()
            raise OSError("fixture lost commit acknowledgement")

    def flush():
        try:
            results.append(db.flush_token_counts(10))
        except BaseException as error:
            errors.append(error)

    waiter = threading.Thread(target=flush)
    original_wait = db._token_queue_cond.wait

    def observed_wait(timeout=None):
        if threading.current_thread() is waiter:
            waiting.set()
        return original_wait(timeout)

    monkeypatch.setattr(db, "_conn", LostAckConnection())
    monkeypatch.setattr(db._token_queue_cond, "wait", observed_wait)
    db.queue_token_counts("s", input_tokens=7, model="m", api_call_count=1)
    writer = db._token_writer_thread
    waiter.start()
    try:
        assert entered.wait(5)
        assert waiting.wait(5), "flush never waited on the active commit"
    finally:
        release.set()
        waiter.join(10)
        writer.join(10)
        monkeypatch.setattr(db, "_conn", conn)
    assert not waiter.is_alive() and not writer.is_alive()
    assert not errors
    assert _totals(db) == (7, 7)
    assert len(db._token_writer_uncertain) == 1
    assert db.flush_token_counts(0) is False
    assert results == [False]


@pytest.mark.parametrize("deadline_expired", [False, True])
def test_caller_drain_waits_for_respawned_successor_with_same_deadline(db, monkeypatch, deadline_expired):
    import hermes_state_usage

    clock_offset = 0.0
    monkeypatch.setattr(hermes_state_usage, "time", SimpleNamespace(
        monotonic=lambda: time.monotonic() + clock_offset, time=time.time,
    ))
    _retain(db, monkeypatch, input_tokens=3, model="first", api_call_count=1)
    first, release_first, second, release_second = (threading.Event() for _ in range(4))
    waiting, decision, finished = (threading.Event() for _ in range(3))
    results, errors = [], []
    original_batch = db._apply_token_batch

    def paused_batch(batch):
        entered, release = (first, release_first) if batch[0][1]["model"] == "first" else (second, release_second)
        entered.set()
        assert release.wait(10), "claimed batch was never released"
        return original_batch(batch)

    def flush():
        try:
            results.append(db.flush_token_counts(10))
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()
            decision.set()

    caller = threading.Thread(target=flush)
    original_apply, original_wait = db._apply_claimed_batch, db._token_queue_cond.wait

    def successor_claims_before_caller_checks(batch):
        nonlocal clock_offset
        result = original_apply(batch)
        if threading.current_thread() is caller:
            assert second.wait(5), "respawned writer never claimed its successor"
            if deadline_expired:
                clock_offset = 11.0
        return result

    def observed_wait(timeout=None):
        if threading.current_thread() is caller and second.is_set():
            waiting.set()
            decision.set()
        return original_wait(timeout)

    monkeypatch.setattr(db, "_apply_token_batch", paused_batch)
    monkeypatch.setattr(db, "_apply_claimed_batch", successor_claims_before_caller_checks)
    monkeypatch.setattr(db._token_queue_cond, "wait", observed_wait)
    caller.start()
    try:
        assert first.wait(5)
        db.queue_token_counts("s", input_tokens=7, model="second", api_call_count=1)
        release_first.set()
        assert second.wait(5)
        assert decision.wait(5), "caller neither waited nor returned"
        if deadline_expired:
            assert finished.is_set() and not waiting.is_set(), "caller reset its deadline after draining"
        else:
            assert waiting.is_set() and not finished.is_set(), f"flush returned early: {results}"
    finally:
        release_first.set()
        release_second.set()
        caller.join(10)
    assert not caller.is_alive() and not errors
    assert results == [not deadline_expired]
    assert db.flush_token_counts(5)
    assert _totals(db) == (10, 10)
    assert not db.has_pending_token_counts()


def test_real_malformed_cost_rollback_is_retained_without_uncertainty(db, monkeypatch):
    _retain(db, monkeypatch, input_tokens=3, model="m", estimated_cost_usd="malformed")
    assert db.flush_token_counts(5) is False
    assert _totals(db) == (0, 0)
    assert not db._token_writer_uncertain
    assert len(db._token_queue) == 1
    assert db._token_queue[0][1]["estimated_cost_usd"] == "malformed"


def test_non_sqlite_rollback_retries_without_losing_or_duplicating_successor(db, monkeypatch):
    _retain(db, monkeypatch, input_tokens=3, model="first", api_call_count=1)
    _retain(db, monkeypatch, input_tokens=7, model="second", api_call_count=1)
    original = db._record_model_usage
    failed = False

    def fail_once(*args, **kwargs):
        nonlocal failed
        if not failed:
            failed = True
            raise ValueError("fixture attribution failure after session UPDATE")
        return original(*args, **kwargs)

    monkeypatch.setattr(db, "_record_model_usage", fail_once)
    assert db.flush_token_counts(5) is False
    assert _totals(db) == (0, 0)
    assert not db._token_writer_uncertain
    assert len(db._token_queue) == 2
    assert db.flush_token_counts(5) is True
    assert _totals(db) == (10, 10)
    assert db._read_one("SELECT api_call_count FROM sessions WHERE id='s'")[0] == 2


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_proven_rollback_keeps_interrupt_visible_and_delta_retryable(db, monkeypatch, interrupt):
    _retain(db, monkeypatch, input_tokens=3, model="m", api_call_count=1)
    original = db._record_model_usage

    def interrupted(*args, **kwargs):
        raise interrupt("fixture interruption inside active transaction")

    monkeypatch.setattr(db, "_record_model_usage", interrupted)
    with pytest.raises(interrupt):
        db.flush_token_counts(5)
    assert _totals(db) == (0, 0)
    assert not db._token_writer_uncertain
    assert len(db._token_queue) == 1
    monkeypatch.setattr(db, "_record_model_usage", original)
    assert db.flush_token_counts(5)
    assert _totals(db) == (3, 3)
