"""The ``sessions.storage`` config key and its one resolver.

``agent.session_storage.session_storage`` is the only way production code reaches the session
storage ports: ``local`` (the default) wraps the profile's own ``SessionDB`` — today's store,
byte for byte — and any other name is a provider plugin that must prove it isolates before its
ports open. The local wrap serves its store wherever the caller runs: a caller under another
profile's scope gets the store's own profile bound around each operation, where today's unpinned
calls reached the store directly, and a hosted scope is refused by the store, never served by
it. A name that matches no provider is an error, never a fallback to the local store.
"""
from __future__ import annotations

import contextvars
from contextlib import closing
from pathlib import Path

import pytest

from agent.runtime_scope import bind_profile_scope, bind_turn, claim_fence
from agent.session_storage import session_storage
from hermes_scope import Principal, Scope
from hermes_state import SessionDB
from hermes_state_errors import TurnFenceSupersededError
from hermes_state_port import StoragePinRefused

BATCH = [{'role': 'user', 'content': 'one'}, {'role': 'assistant', 'content': 'two'}]


def _store(home: Path) -> SessionDB:
    db = SessionDB(home / 'state.db')
    db.create_session('s1', source='cli')
    return db


def _contents(db: SessionDB) -> list:
    return [m['content'] for m in db.get_messages_as_conversation('s1')]


def test_the_default_backend_is_the_profiles_own_store(tmp_path):
    """No config key: the resolver wraps the caller's SessionDB; the same rows land."""
    home = tmp_path / 'profile'
    home.mkdir()
    with closing(_store(home)) as db:
        storage = session_storage(db)
        assert storage is not None
        storage.sessions.append('s1', BATCH)
        assert _contents(db) == ['one', 'two']


def test_a_claimed_turns_append_through_the_port_is_fenced(tmp_path):
    """The turn-path append goes through the port, so the turn's fence applies: the write holds
    while the claim holds, and a superseded fence is refused whole."""
    home = tmp_path / 'profile'
    home.mkdir()
    with closing(_store(home)) as db:
        storage = session_storage(db)
        assert storage is not None
        admission = storage.admission
        with bind_profile_scope(home):
            epoch = admission.begin_epoch(instance_id='owner')
            admission.admit(epoch=epoch, principal_id='human', session_id='s1',
                            request_id='r1', payload={'text': 'hi'})
            row = admission.claim_next(epoch=epoch, session_id='s1')
            assert isinstance(row, dict)
            with bind_turn('s1', row['admission_id'], claim_fence(row), store=db.db_path):
                storage.sessions.append_turn('s1', BATCH)
            assert _contents(db) == ['one', 'two']
            # What a newer claim or an invalidating mutation does to the session's lease.
            contextvars.Context().run(
                db._write_sql, 'UPDATE sessions SET runtime_generation=runtime_generation+1 WHERE id=?',
                ('s1',))
            with bind_turn('s1', row['admission_id'], claim_fence(row), store=db.db_path):
                with pytest.raises(TurnFenceSupersededError):
                    storage.sessions.append_turn('s1', [{'role': 'user', 'content': 'three'}])
        assert _contents(db) == ['one', 'two']


def test_the_store_serves_its_own_profile_wherever_the_caller_runs(tmp_path):
    """A caller under another profile's scope (a multiplexed owner's drain) still reaches the
    store: the resolver binds the store's own profile around the operation, where today's
    calls had no pin at all."""
    other = tmp_path / 'launch'
    other.mkdir()
    home = tmp_path / 'served'
    home.mkdir()
    with closing(_store(home)) as db:
        storage = session_storage(db)
        assert storage is not None
        with bind_profile_scope(other):
            assert storage.admission.list('s1') == []
            assert storage.sessions.get('s1') is not None


def test_a_missing_named_backend_is_an_error_never_the_local_store(tmp_path):
    home = tmp_path / 'profile'
    home.mkdir()
    (home / 'config.yaml').write_text('sessions:\n  storage: no-such-backend\n')
    with closing(_store(home)) as db:
        with pytest.raises(LookupError, match='no-such-backend'):
            session_storage(db)


def test_a_hosted_scope_is_never_served_by_a_local_store(tmp_path):
    """The local store serves the local tenant only: a hosted scope is refused, never handed a
    local store as a fallback."""
    home = tmp_path / 'profile'
    home.mkdir()
    with closing(_store(home)) as db:
        storage = session_storage(db)
        assert storage is not None
        scope = Scope('tenant-a', 'agent-x', Principal('member'),
                      owner=Principal('owner'), can_administer=True)
        from agent.runtime_scope import bind_state
        with bind_state(scope):
            with pytest.raises(StoragePinRefused):
                storage.admission.list('s1')


def test_the_local_backend_without_a_store_is_nothing_to_persist_to(tmp_path):
    """No local store and no configured provider: the resolver hands back nothing, so the flush
    keeps today's skip."""
    assert session_storage(None) is None
