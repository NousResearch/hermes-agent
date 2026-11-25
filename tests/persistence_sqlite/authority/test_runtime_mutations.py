"""Revision, replay and business-rule invariants on real isolated SQLite."""
import sqlite3

import pytest

from persistence.conversation.sessions import CANONICAL_BOT_CHAT_TITLE
from persistence.core.errors import AuthorityRefusal
from persistence.sqlite._holder import StoreHolder
from persistence.sqlite._state.common import TITLE_SOURCE_USER
from persistence.sqlite.authority import _mutation
from persistence.sqlite.authority._runtime_ends import begin_runtime_epoch
from persistence.sqlite.conversation._store.sessions import create_session, end_session, get_session, set_session_hidden
from persistence.sqlite.conversation._store.titles import get_session_title, get_session_title_source, set_session_title


def mutate(holder, **args):
    """The mutation's receipt result."""
    return _mutation.mutate_runtime_session(holder, **args)[0]


@pytest.mark.parametrize('operation,payload', [('rename', {'title': ' New  name '}), ('rename', {'title': 'a' + chr(0xD800) + 'b'}), ('archive', {'archived': True})])
def test_mutation_receipt_fences_replay_revision_and_epoch(backend, store, operation, payload):
    holder = store.holder()
    create_session(holder, 's', 'test')
    epoch = begin_runtime_epoch(holder, instance_id='boot')
    assert callable(getattr(_mutation, 'mutate_runtime_session', None)), 'revision-fenced mutation operation missing'
    args = dict(epoch=epoch, principal_id='human', session_id='s', request_id='edit', expected_revision=0, operation=operation, payload=payload)
    if operation == 'rename':
        create_session(holder, 'direct', 'test')
        set_session_title(holder, 'direct', payload['title'])
        expected_title = get_session_title(holder, 'direct')
        set_session_title(holder, 'direct', '')
    result = mutate(holder, **args)
    if operation == 'rename':
        assert result['title'] == expected_title
        if payload['title'] != expected_title:
            with pytest.raises(AuthorityRefusal, match='admission_conflict'):
                mutate(holder, **(args | {'payload': {'title': expected_title}}))
    assert result['revision'] == 1
    assert mutate(holder, **args) == result
    assert get_session(holder, 's')['runtime_revision'] == 1
    with pytest.raises(AuthorityRefusal, match='admission_conflict'):
        mutate(holder, **{**args, 'expected_revision': 1})
    with pytest.raises(AuthorityRefusal, match='revision_conflict'):
        mutate(holder, **{**args, 'request_id': 'other'})
    peer = StoreHolder(store.store_path)
    try:
        with pytest.raises(AuthorityRefusal, match='revision_conflict'):
            mutate(peer, **{**args, 'request_id': 'peer'})
        later = mutate(peer, **{**args, 'request_id': 'peer', 'expected_revision': 1})
        assert later['revision'] == 2
        assert mutate(holder, **args) == result
        assert get_session(holder, 's')['runtime_revision'] == 2
    finally:
        peer.close()
    new_epoch = begin_runtime_epoch(holder, instance_id='replacement')
    with pytest.raises(AuthorityRefusal, match='stale_epoch'):
        mutate(holder, **args)
    assert mutate(holder, **{**args, 'epoch': new_epoch}) == result
    backend.reopen()
    holder = store.holder()
    assert mutate(holder, **{**args, 'epoch': new_epoch}) == result


@pytest.mark.parametrize('operation,payload', [('rename', {'title': ' New  name '}), ('archive', {'archived': True})])
def test_mutation_rules_and_receipt_failure_are_atomic(store, operation, payload):
    holder = store.holder()
    create_session(holder, 's', 'test')
    set_session_title(holder, 's', 'Original')
    epoch = begin_runtime_epoch(holder, instance_id='boot')
    assert callable(getattr(_mutation, 'mutate_runtime_session', None)), 'revision-fenced mutation operation missing'
    args = dict(epoch=epoch, principal_id='human', session_id='s', request_id='edit', expected_revision=0, operation=operation, payload=payload)
    before = get_session(holder, 's')
    holder._execute_write(lambda c: c.execute("CREATE TRIGGER refuse_receipt BEFORE INSERT ON state_meta WHEN NEW.key LIKE 'gateway.mutation.%' BEGIN SELECT RAISE(ABORT, 'receipt refused'); END"))
    with pytest.raises(sqlite3.IntegrityError, match='receipt refused'):
        mutate(holder, **args)
    assert get_session(holder, 's') == before
    holder._execute_write(lambda c: c.execute('DROP TRIGGER refuse_receipt'))
    if operation == 'rename':
        create_session(holder, 'bot', 'test')
        set_session_title(holder, 'bot', CANONICAL_BOT_CHAT_TITLE)
        set_session_hidden(holder, 'bot', True)
        with pytest.raises(ValueError, match='canonical Bot Chat'):
            mutate(holder, **{**args, 'session_id': 'bot'})
        create_session(holder, 'collision', 'test')
        set_session_title(holder, 'collision', 'New name')
        with pytest.raises(ValueError, match='already in use'):
            mutate(holder, **args)
        set_session_title(holder, 'collision', '')
        mutate(holder, **args)
        assert get_session_title(holder, 's') == 'New name'
        assert get_session_title_source(holder, 's') == TITLE_SOURCE_USER
        end_session(holder, 's', end_reason='compression')
        create_session(holder, 'tip', 'test', parent_session_id='s')
        mutate(holder, **{**args, 'session_id': 'tip', 'request_id': 'transfer'})
        assert get_session_title(holder, 's') is None
        assert get_session_title(holder, 'tip') == 'New name'
        assert get_session(holder, 's')['runtime_revision'] == 2
        assert get_session(holder, 'tip')['runtime_revision'] == 1
    else:
        create_session(holder, 'child', 'test', parent_session_id='s')
        end_session(holder, 's', end_reason='compression')
        mutate(holder, **args)
        assert get_session(holder, 's')['archived'] == get_session(holder, 'child')['archived'] == 1
        assert get_session(holder, 'child')['runtime_revision'] == 1
