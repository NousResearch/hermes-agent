"""Revision, replay and business-rule invariants on real isolated SQLite."""
import sqlite3

import pytest

from hermes_state import SessionDB
import hermes_state_runtime as rt


@pytest.mark.parametrize('operation,payload', [('rename', {'title': ' New  name '}), ('rename', {'title': 'a' + chr(0xD800) + 'b'}), ('archive', {'archived': True})])
def test_mutation_receipt_fences_replay_revision_and_epoch(tmp_path, operation, payload):
    db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        db.create_session('s', source='test')
        epoch = rt.begin_runtime_epoch(db, instance_id='boot')
        mutate = getattr(rt, 'mutate_runtime_session', None)
        assert callable(mutate), 'revision-fenced mutation operation missing'
        args = dict(epoch=epoch, principal_id='human', session_id='s', request_id='edit', expected_revision=0, operation=operation, payload=payload)
        if operation == 'rename':
            db.create_session('direct', source='test')
            db.set_session_title('direct', payload['title'])
            expected_title = db.get_session_title('direct')
            db.set_session_title('direct', '')
        result = mutate(db, **args)
        if operation == 'rename':
            assert result['title'] == expected_title
            if payload['title'] != expected_title:
                with pytest.raises(rt.RuntimeStoreError, match='admission_conflict'):
                    mutate(db, **(args | {'payload': {'title': expected_title}}))
        assert result['revision'] == 1
        assert mutate(db, **args) == result
        assert db.get_session('s')['runtime_revision'] == 1
        with pytest.raises(rt.RuntimeStoreError, match='admission_conflict'):
            mutate(db, **{**args, 'expected_revision': 1})
        with pytest.raises(rt.RuntimeStoreError, match='revision_conflict'):
            mutate(db, **{**args, 'request_id': 'other'})
        peer = SessionDB(db_path=tmp_path / 'state.db')
        try:
            with pytest.raises(rt.RuntimeStoreError, match='revision_conflict'):
                mutate(peer, **{**args, 'request_id': 'peer'})
            later = mutate(peer, **{**args, 'request_id': 'peer', 'expected_revision': 1})
            assert later['revision'] == 2
            assert mutate(db, **args) == result
            assert db.get_session('s')['runtime_revision'] == 2
        finally:
            peer.close()
        new_epoch = rt.begin_runtime_epoch(db, instance_id='replacement')
        with pytest.raises(rt.RuntimeStoreError, match='stale_epoch'):
            mutate(db, **args)
        assert mutate(db, **{**args, 'epoch': new_epoch}) == result
        db.close()
        db = SessionDB(db_path=tmp_path / 'state.db')
        assert mutate(db, **{**args, 'epoch': new_epoch}) == result
    finally:
        db.close()


@pytest.mark.parametrize('operation,payload', [('rename', {'title': ' New  name '}), ('archive', {'archived': True})])
def test_mutation_rules_and_receipt_failure_are_atomic(tmp_path, operation, payload):
    db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        db.create_session('s', source='test')
        db.set_session_title('s', 'Original')
        epoch = rt.begin_runtime_epoch(db, instance_id='boot')
        mutate = getattr(rt, 'mutate_runtime_session', None)
        assert callable(mutate), 'revision-fenced mutation operation missing'
        args = dict(epoch=epoch, principal_id='human', session_id='s', request_id='edit', expected_revision=0, operation=operation, payload=payload)
        before = db.get_session('s')
        db._execute_write(lambda c: c.execute("CREATE TRIGGER refuse_receipt BEFORE INSERT ON state_meta WHEN NEW.key LIKE 'gateway.mutation.%' BEGIN SELECT RAISE(ABORT, 'receipt refused'); END"))
        with pytest.raises(sqlite3.IntegrityError, match='receipt refused'):
            mutate(db, **args)
        assert db.get_session('s') == before
        db._execute_write(lambda c: c.execute('DROP TRIGGER refuse_receipt'))
        if operation == 'rename':
            db.create_session('bot', source='test')
            db.set_session_title('bot', db.CANONICAL_BOT_CHAT_TITLE)
            db.set_session_hidden('bot', True)
            with pytest.raises(ValueError, match='canonical Bot Chat'):
                mutate(db, **{**args, 'session_id': 'bot'})
            db.create_session('collision', source='test')
            db.set_session_title('collision', 'New name')
            with pytest.raises(ValueError, match='already in use'):
                mutate(db, **args)
            db.set_session_title('collision', '')
            mutate(db, **args)
            assert db.get_session_title('s') == 'New name'
            assert db.get_session_title_source('s') == db.TITLE_SOURCE_USER
            db.end_session('s', end_reason='compression')
            db.create_session('tip', source='test', parent_session_id='s')
            mutate(db, **{**args, 'session_id': 'tip', 'request_id': 'transfer'})
            assert db.get_session_title('s') is None
            assert db.get_session_title('tip') == 'New name'
            assert db.get_session('s')['runtime_revision'] == 2
            assert db.get_session('tip')['runtime_revision'] == 1
        else:
            db.create_session('child', source='test', parent_session_id='s')
            db.end_session('s', end_reason='compression')
            mutate(db, **args)
            assert db.get_session('s')['archived'] == db.get_session('child')['archived'] == 1
            assert db.get_session('child')['runtime_revision'] == 1
    finally:
        db.close()
