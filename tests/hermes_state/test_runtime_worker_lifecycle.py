"""Worker lifecycle parity on reserved rows; compose the documented parent hooks."""
import pytest

from agent.runtime_session_store import RuntimeSessionStore, WorkerPersistenceError
from hermes_state import SessionDB
from hermes_state_runtime import (
    RuntimeStoreError, begin_runtime_epoch, mutate_worker_execution, register_worker_execution,
)


def setup_store(tmp_path):
    db = SessionDB(tmp_path / 'state.db')
    db.create_session('owned', 'cli', cwd='/reserved', profile_name='fixture')
    db.create_session('foreign', 'cli', system_prompt='private')
    epoch = begin_runtime_epoch(db, instance_id='fixture')
    scope = dict(epoch=epoch, execution_id='worker', session_id='owned', generation=0)
    register_worker_execution(db, **scope, kind='compute', adoption_secret='private')
    store = RuntimeSessionStore(lambda method, **p: mutate_worker_execution(db, **p), scope, tmp_path / 'outbox')
    return db, store, scope


def test_reserved_constructor_backfill_and_end_are_receipted(tmp_path):
    db, store, scope = setup_store(tmp_path)
    try:
        assert store.create_session('owned', 'cli', model='requested', system_prompt='prefix',
                                    model_config={'max_iterations': 5}, cwd='/reserved',
                                    profile_name='fixture') == 'owned'
        assert db.get_session('owned')['model'] == 'requested'
        assert db.get_session('owned')['system_prompt'] == 'prefix'
        store.create_session('owned', 'cli', model='ignored', system_prompt='ignored',
                             model_config={'max_iterations': 10})
        assert db.get_session('owned')['model'] == 'requested'
        assert db.get_session_model_config_value('owned', 'max_iterations') == 5
        assert store.session_lifecycle_statuses(['owned']) == {'owned': 'empty'}
        with pytest.raises(WorkerPersistenceError, match='permission_denied'):
            store.create_session('foreign', 'cli')
        for payload in ({'source': 'cli', 'parent_session_id': 'foreign'},
                        {'source': 'cli', 'profile_name': 'foreign'},
                        {'source': 'cli', 'cwd': '/foreign'},
                        {'source': 'cli', 'model_config': {'_delegate_from': 'foreign'}}):
            with pytest.raises(RuntimeStoreError, match='permission_denied|invalid_params'):
                mutate_worker_execution(db, **scope, sequence=store.journal['next_sequence'],
                                        operation='session.create', payload=payload)
        store.append_messages_batch('owned', [{'role': 'assistant', 'content': 'done', 'finish_reason': 'stop'}])
        assert store.session_lifecycle_statuses(['owned']) == db.session_lifecycle_statuses(['owned'])
        original = store.rpc
        def lost_ack(method, **params):
            original(method, **params)
            raise TimeoutError('lost-ack')
        store.rpc = lost_ack
        with pytest.raises(TimeoutError, match='lost-ack'):
            store.end_session('owned', 'agent_close')
        ended = db.get_session('owned')
        store.rpc = original
        store.retry_pending()
        store.end_session('owned', 'late-reason')
        assert db.get_session('owned')['ended_at'] == ended['ended_at']
        assert db.get_session('owned')['end_reason'] == 'agent_close'
        assert store.finish()['status'] == 'terminal'
        assert db.get_session('foreign')['system_prompt'] == 'private'
    finally:
        store.close()
        db.close()


def test_late_finalizers_cannot_mutate_successor_or_foreign_rows(tmp_path):
    db, store, scope = setup_store(tmp_path)
    try:
        store.end_session('owned', 'agent_close')
        db.reopen_session('owned')
        db._execute_write(lambda conn: conn.execute(
            'UPDATE sessions SET runtime_generation=runtime_generation+1 WHERE id=?', ('owned',)))
        # Even replay of a formerly successful receipt must revalidate generation.
        for sequence in (1, 2):
            with pytest.raises(RuntimeStoreError, match='stale_generation'):
                mutate_worker_execution(db, **scope, sequence=sequence,
                                        operation='session.end', payload={'end_reason': 'agent_close'})
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            mutate_worker_execution(db, **dict(scope, session_id='foreign'), sequence=2,
                                    operation='session.end', payload={'end_reason': 'agent_close'})
        assert db.get_session('owned')['ended_at'] is None
        assert db.get_session('foreign')['ended_at'] is None
    finally:
        store.close()
        db.close()
