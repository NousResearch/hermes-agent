"""A claimed turn's routed append keeps the durable turn lease guard.

The turn's rows go through the session storage port; the port's append carries the caller's
durable turn lease holder and TTL into the store's own lease-checked write, unchanged: a
lease replaced while the claimed turn still runs refuses the stale transcript, and an
expired lease the holder still owns is renewed by the write itself.
"""
import sqlite3
import time
from types import SimpleNamespace
from contextlib import closing

import pytest

from hermes_state import SessionDB
from hermes_state_errors import SessionTurnLeaseLostError
import hermes_state_runtime as rt
from agent.runtime_scope import bind_profile_scope, bind_turn, claim_fence
from agent.session_persistence import _db_flush_write


def _claimed_turn(db, session_id='s'):
    epoch = rt.begin_runtime_epoch(db, instance_id='one')
    rt.admit_session_input(db, epoch=epoch, principal_id='u', session_id=session_id,
                           request_id='r', payload={})
    row = rt.claim_session_input(db, epoch=epoch, session_id=session_id)
    assert isinstance(row, dict)
    return row


def _lease_expiry(db, holder):
    with closing(sqlite3.connect(str(db.db_path))) as conn:
        row = conn.execute("SELECT expires_at FROM session_turn_leases WHERE holder = ?",
                           (holder,)).fetchone()
        return None if row is None else float(row[0])


def test_a_replaced_durable_lease_refuses_the_claimed_turns_append(tmp_path):
    with closing(SessionDB(tmp_path / 'state.db')) as db:
        db.create_session('s', 'cli')
        row = _claimed_turn(db)
        assert db.acquire_session_turn_lease('s', 'old')
        agent = SimpleNamespace(_session_db=db, session_id='s', _active_session_turn_lease_holder='old')
        db.release_session_turn_lease('s', 'old')
        assert db.acquire_session_turn_lease('s', 'replacement')
        rows = [{'role': 'assistant', 'content': 'stale durable holder'}]
        with bind_profile_scope(tmp_path), bind_turn('s', row['admission_id'], claim_fence(row),
                                                     store=db.db_path):
            with pytest.raises(SessionTurnLeaseLostError):
                _db_flush_write(agent, rows, rows, rows)
        assert db.get_messages('s') == []


def test_an_expired_durable_lease_is_renewed_by_the_claimed_turns_append(tmp_path):
    with closing(SessionDB(tmp_path / 'state.db')) as db:
        db.create_session('s', 'cli')
        row = _claimed_turn(db)
        assert db.acquire_session_turn_lease('s', 'mine', ttl_seconds=30)
        # A starved refresher: the lease expired but the holder still owns the row.
        with closing(sqlite3.connect(str(db.db_path))) as conn:
            conn.execute("UPDATE session_turn_leases SET expires_at = ? WHERE holder = 'mine'",
                         (time.time() - 5,))
            conn.commit()
        agent = SimpleNamespace(_session_db=db, session_id='s',
                                _active_session_turn_lease_holder='mine',
                                _active_session_turn_lease_ttl_seconds=299.0)
        rows = [{'role': 'assistant', 'content': 'renewed holder'}]
        with bind_profile_scope(tmp_path), bind_turn('s', row['admission_id'], claim_fence(row),
                                                     store=db.db_path):
            _db_flush_write(agent, rows, rows, rows)
        assert [m['content'] for m in db.get_messages('s')] == ['renewed holder']
        after = _lease_expiry(db, 'mine')
        assert after is not None and after - time.time() > 200, \
            'the write must renew the expired lease of the holder that still owns it'
