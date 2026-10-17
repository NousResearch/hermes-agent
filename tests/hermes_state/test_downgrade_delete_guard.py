"""A release that predates the gateway runtime refuses, in words, to delete a ledgered session.

Older releases also stamp ``schema_version`` 31 and never refuse a newer schema, so the store
itself has to carry the refusal: their ``sessions delete`` / ``prune`` used to die on the bare
``FOREIGN KEY constraint failed`` of the ``ON DELETE RESTRICT`` ledgers (jquesnelle, #106742).
"""
import sqlite3
from contextlib import closing

import pytest

import hermes_state_runtime as rt
from hermes_state import SessionDB
from hermes_state_errors import classify_persistence_error


def _settled(db, epoch, sid):
    db.create_session(sid, source='cli')
    db.append_message(sid, role='user', content='kept')
    accepted = rt.admit_session_input(db, epoch=epoch, principal_id='human', session_id=sid,
                                      request_id='r', payload={'text': 'x'})
    claim = rt.claim_session_input(db, epoch=epoch, session_id=sid)
    rt.settle_session_input(db, epoch=epoch, admission_id=accepted['admission_id'],
                            generation=claim['generation'], outcome='completed')


def _older_release_delete(path, sid):
    """The statements v0.21.6's ``SessionDB.delete_session`` runs, in its one write transaction."""
    conn = sqlite3.connect(path, isolation_level=None)
    try:
        conn.execute('PRAGMA foreign_keys=ON')
        conn.execute('BEGIN IMMEDIATE')
        try:
            conn.execute('UPDATE sessions SET parent_session_id = NULL WHERE parent_session_id = ?', (sid,))
            conn.execute('DELETE FROM messages WHERE session_id = ?', (sid,))
            conn.execute('DELETE FROM sessions WHERE id = ?', (sid,))
            conn.execute('COMMIT')
        except BaseException:
            conn.execute('ROLLBACK')
            raise
    finally:
        conn.close()


def test_an_older_release_delete_of_a_ledgered_session_is_refused_with_the_way_back(tmp_path):
    path = tmp_path / 'state.db'
    with closing(SessionDB(path)) as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        _settled(db, epoch, 'ledgered')
        db.create_session('plain', source='cli')

    with pytest.raises(sqlite3.IntegrityError) as refused:
        _older_release_delete(path, 'ledgered')
    message = str(refused.value)
    assert 'newer Hermes' in message and 'Downgrading is unsupported' in message
    assert 'hermes update' in message and 'pre-update snapshot' in message
    # The older release buckets by phrase; this must not read as damage, contention or a full disk.
    assert classify_persistence_error(refused.value) == 'unknown'
    _older_release_delete(path, 'plain')  # a session without runtime history still deletes

    with closing(SessionDB(path)) as db:
        assert [m['content'] for m in db.get_messages('ledgered')] == ['kept']
        assert db.get_session('plain') is None
        # This release's own delete retires the ledger first and is never refused by the guard.
        assert db.delete_session('ledgered') and db.get_session('ledgered') is None


def test_an_upgraded_store_gains_the_guard_on_its_next_open(tmp_path):
    path = tmp_path / 'state.db'
    with closing(SessionDB(path)) as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        _settled(db, epoch, 's')
    with closing(sqlite3.connect(path)) as conn:
        conn.execute('DROP TRIGGER sessions_runtime_ledger_delete_guard')
        conn.commit()
    SessionDB(path).close()
    with pytest.raises(sqlite3.IntegrityError, match='newer Hermes'):
        _older_release_delete(path, 's')
