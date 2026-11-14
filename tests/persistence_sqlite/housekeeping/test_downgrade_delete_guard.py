"""A release that predates the gateway runtime refuses, in words, to delete a ledgered session of a SQLite backend store.

Older releases also stamp ``schema_version`` 31 and never refuse a newer schema, so the store
itself has to carry the refusal: their ``sessions delete`` / ``prune`` used to die on the bare
``FOREIGN KEY constraint failed`` of the ``ON DELETE RESTRICT`` ledgers (jquesnelle, #106742).
"""
import sqlite3
from contextlib import closing

import pytest

from persistence.sqlite._holder import StoreHolder
from hermes_state_errors import classify_persistence_error
from persistence.sqlite.authority._admission_store import (
    admit_session_input, claim_session_input, settle_session_input,
)
from persistence.sqlite.authority._runtime_ends import begin_runtime_epoch
from persistence.sqlite.conversation._store.messages import append_message, get_messages
from persistence.sqlite.conversation._store.sessions import create_session, delete_session, get_session
from tests.contracts._scopes import MEDIA_ROOT


def _settled(db, epoch, sid):
    create_session(db, sid, source='cli')
    append_message(db, sid, role='user', content='kept')
    accepted, _ = admit_session_input(db, epoch=epoch, principal_id='human', session_id=sid,
                                      request_id='r', payload={'text': 'x'})
    claim = claim_session_input(db, epoch=epoch, session_id=sid)
    settle_session_input(db, epoch=epoch, admission_id=accepted['admission_id'],
                         generation=claim['generation'], outcome='completed')


def _older_release_delete(path, sid):
    """The statements a v0.21.6 ``sessions delete`` runs, in its one write transaction."""
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
    with closing(StoreHolder(path)) as db:
        epoch = begin_runtime_epoch(db, instance_id='owner')
        _settled(db, epoch, 'ledgered')
        create_session(db, 'plain', source='cli')

    with pytest.raises(sqlite3.IntegrityError) as refused:
        _older_release_delete(path, 'ledgered')
    message = str(refused.value)
    assert 'newer Hermes' in message and 'Downgrading is unsupported' in message
    assert 'hermes update' in message and 'pre-update snapshot' in message
    # The older release buckets by phrase; this must not read as damage, contention or a full disk.
    assert classify_persistence_error(refused.value) == 'unknown'
    _older_release_delete(path, 'plain')  # a session without runtime history still deletes

    with closing(StoreHolder(path)) as db:
        assert [m['content'] for m in get_messages(db, 'ledgered')] == ['kept']
        assert get_session(db, 'plain') is None
        # This release's own delete retires the ledger first and is never refused by the guard.
        assert 'ledgered' in delete_session(db, 'ledgered', media_root=MEDIA_ROOT) and get_session(db, 'ledgered') is None


def test_an_upgraded_store_gains_the_guard_on_its_next_open(tmp_path):
    path = tmp_path / 'state.db'
    with closing(StoreHolder(path)) as db:
        epoch = begin_runtime_epoch(db, instance_id='owner')
        _settled(db, epoch, 's')
    with closing(sqlite3.connect(path)) as conn:
        conn.execute('DROP TRIGGER sessions_runtime_ledger_delete_guard')
        conn.commit()
    StoreHolder(path).close()
    with pytest.raises(sqlite3.IntegrityError, match='newer Hermes'):
        _older_release_delete(path, 's')
