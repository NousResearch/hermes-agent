"""A claimed turn's writes are fenced by its claim, inside the write's own transaction.

While a turn's body runs (``agent.runtime_scope.bind_turn`` with the issuing store), every write to
that store checks the turn's fence (owner epoch + session runtime generation) against the session's
current lease. A turn that lost its lease writes nothing; everything outside the turn is untouched.
"""
import contextvars
import threading
from contextlib import closing

import pytest

import hermes_state_runtime as rt
from agent.runtime_scope import bind_profile_scope, bind_turn, claim_fence, owner_operation
from hermes_state import SessionDB
from hermes_state_errors import SessionTurnLeaseLostError, TurnFenceSupersededError
from tests.hermes_state.test_target_advance_fence import _local_session

BATCH = [{'role': 'user', 'content': 'one'}, {'role': 'assistant', 'content': 'two'},
         {'role': 'user', 'content': 'three'}]


def _claimed(db, tmp_path):
    epoch = rt.begin_runtime_epoch(db, instance_id='owner')
    sid = _local_session(db, epoch, tmp_path)
    rt.admit_session_input(db, epoch=epoch, principal_id='human', session_id=sid,
                           request_id='r1', payload={'text': 'hi'})
    return sid, rt.claim_session_input(db, epoch=epoch, session_id=sid)


def _snapshot(db, sid):
    session = db.get_session(sid)
    return [m['content'] for m in db.get_messages(sid)], session['message_count']


def _elsewhere(fn, *args):
    """Run *fn* as another writer would: in a fresh context, outside any turn."""
    return contextvars.Context().run(fn, *args)


def _move_generation(db, sid):
    """What a newer claim or an invalidating mutation does to the session's lease."""
    _elsewhere(db._write_sql, 'UPDATE sessions SET runtime_generation=runtime_generation+1 WHERE id=?', (sid,))


def _take_over(db, sid):
    """Another owner takes the profile over: a new owner epoch."""
    _elsewhere(lambda: rt.begin_runtime_epoch(db, instance_id='successor'))


@pytest.fixture
def store(tmp_path):
    with closing(SessionDB(tmp_path / 'state.db')) as db:
        yield db


def test_a_turn_holding_its_lease_writes_as_before(store, tmp_path):
    sid, row = _claimed(store, tmp_path)
    with bind_profile_scope(tmp_path), bind_turn(sid, row['admission_id'], claim_fence(row), store=store.db_path):
        store.append_messages_batch(sid, BATCH)
        store.append_message(sid, 'assistant', 'four')
    assert _snapshot(store, sid) == (['one', 'two', 'three', 'four'], 4)
    settled = rt.settle_session_input(store, epoch=row['owner_epoch'], admission_id=row['admission_id'],
                                      generation=row['generation'], outcome='completed', result={})
    assert settled['status'] == 'terminal'


@pytest.mark.parametrize('supersede', [_move_generation, _take_over], ids=['generation', 'owner_epoch'])
def test_a_superseded_turn_write_is_refused_and_leaves_nothing(store, tmp_path, supersede):
    sid, row = _claimed(store, tmp_path)
    with bind_profile_scope(tmp_path), bind_turn(sid, row['admission_id'], claim_fence(row), store=store.db_path):
        store.append_message(sid, 'user', 'before')
        before = _snapshot(store, sid)
        supersede(store, sid)
        with pytest.raises(TurnFenceSupersededError) as refused:
            store.append_messages_batch(sid, BATCH)
        with pytest.raises(SessionTurnLeaseLostError):  # every lost-lease handler catches it
            store.update_session_cwd(sid, '/elsewhere')
    assert _snapshot(store, sid) == before == (['before'], 1)
    assert store.get_session(sid)['cwd'] != '/elsewhere'
    from hermes_state_errors import classify_persistence_error
    assert classify_persistence_error(refused.value) == 'turn_lease'


def test_writes_outside_a_claimed_turn_are_unaffected(store, tmp_path):
    sid, row = _claimed(store, tmp_path)
    _move_generation(store, sid)
    with bind_profile_scope(tmp_path):  # the profile bound, no turn: maintenance, notes, CLI
        store.append_message(sid, 'user', 'outside')
    with bind_turn(sid, row['admission_id'], claim_fence(row), store=None):
        store.append_message(sid, 'user', 'unnamed store')  # no scope and no store: nothing to fence
    with bind_profile_scope(tmp_path), bind_turn(sid, row['admission_id'], claim_fence(row)):
        with owner_operation():  # the owner's own work in a turn's context
            store.append_message(sid, 'user', 'owner')
    assert _snapshot(store, sid) == (['outside', 'unnamed store', 'owner'], 3)


def test_a_turn_fences_only_the_store_that_issued_it(store, tmp_path):
    sid, row = _claimed(store, tmp_path)
    other_home = tmp_path / 'other'
    other_home.mkdir()
    with closing(SessionDB(other_home / 'state.db')) as other:
        other.create_session(sid, 'cli')
        _move_generation(store, sid)
        with bind_turn(sid, row['admission_id'], claim_fence(row), store=store.db_path):
            other.append_message(sid, 'user', 'another store')
            with pytest.raises(TurnFenceSupersededError):
                store.append_message(sid, 'user', 'issuing store')
        assert [m['content'] for m in other.get_messages(sid)] == ['another store']
    assert _snapshot(store, sid) == ([], 0)


def test_a_thread_the_turn_started_stays_fenced_after_the_turn_returns(store, tmp_path):
    sid, row = _claimed(store, tmp_path)
    go, done = threading.Event(), threading.Event()
    results = []

    def background():
        go.wait(10)
        try:
            store.append_message(sid, 'assistant', 'late')
            results.append('written')
        except TurnFenceSupersededError:
            results.append('refused')
        done.set()

    with bind_turn(sid, row['admission_id'], claim_fence(row), store=store.db_path):
        current = contextvars.copy_context()
        later = contextvars.copy_context()
    # The turn's body returned; the first thread still holds the lease and writes.
    threading.Thread(target=current.run, args=(background,)).start()
    go.set()
    assert done.wait(10)
    go.clear()
    done.clear()
    _move_generation(store, sid)
    threading.Thread(target=later.run, args=(background,)).start()
    go.set()
    assert done.wait(10)
    assert results == ['written', 'refused']
    assert _snapshot(store, sid) == (['late'], 1)


def test_a_transient_io_error_in_the_lease_check_is_retried_like_begin(store, tmp_path, monkeypatch):
    """The check runs before the callback, so its reads keep the pre-callback disk I/O retry."""
    import sqlite3

    import hermes_state
    sid, row = _claimed(store, tmp_path)
    real, calls = hermes_state.check_turn_lease, []

    def flaky(conn, mark):
        calls.append(mark)
        if len(calls) == 1:
            raise sqlite3.OperationalError('disk I/O error')
        return real(conn, mark)
    monkeypatch.setattr(hermes_state, 'check_turn_lease', flaky)
    with bind_profile_scope(tmp_path), bind_turn(sid, row['admission_id'], claim_fence(row), store=store.db_path):
        store.append_message(sid, 'user', 'kept')
    assert len(calls) == 2
    assert _snapshot(store, sid) == (['kept'], 1)
