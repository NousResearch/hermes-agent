"""Worker lifecycle parity on reserved rows; compose the documented parent hooks."""
import pytest

from agent.runtime_session_store import WorkerPersistenceError
from persistence.core.errors import AuthorityRefusal
from persistence.sqlite.conversation._store.sessions import (
    get_session, get_session_model_config_value, reopen_session, session_lifecycle_statuses,
)
from persistence.sqlite.workers._receipt import mutate_worker_execution
from tests.persistence_sqlite.workers._runtime_worker import setup_store


def test_reserved_constructor_backfill_and_end_are_receipted(backend, scope_a, tmp_path):
    holder, store, scope = setup_store(backend.open(scope_a), tmp_path)
    try:
        assert store.create_session('owned', 'cli', model='requested', system_prompt='prefix',
                                    model_config={'max_iterations': 5}, cwd='/reserved',
                                    profile_name='fixture') == 'owned'
        assert get_session(holder, 'owned')['model'] == 'requested'
        assert get_session(holder, 'owned')['system_prompt'] == 'prefix'
        store.create_session('owned', 'cli', model='ignored', system_prompt='ignored',
                             model_config={'max_iterations': 10})
        assert get_session(holder, 'owned')['model'] == 'requested'
        assert get_session_model_config_value(holder, 'owned', 'max_iterations') == 5
        assert store.session_lifecycle_statuses(['owned']) == {'owned': 'empty'}
        with pytest.raises(WorkerPersistenceError, match='permission_denied'):
            store.create_session('foreign', 'cli')
        for payload in ({'source': 'cli', 'parent_session_id': 'foreign'},
                        {'source': 'cli', 'profile_name': 'foreign'},
                        {'source': 'cli', 'cwd': '/foreign'},
                        {'source': 'cli', 'model_config': {'_delegate_from': 'foreign'}}):
            with pytest.raises(AuthorityRefusal, match='permission_denied|invalid_params'):
                mutate_worker_execution(holder, **scope, sequence=store.journal['next_sequence'],
                                        operation='session.create', payload=payload)
        store.append_messages_batch('owned', [{'role': 'assistant', 'content': 'done', 'finish_reason': 'stop'}])
        assert store.session_lifecycle_statuses(['owned']) == session_lifecycle_statuses(holder, ['owned'])
        original = store.rpc
        def lost_ack(method, **params):
            original(method, **params)
            raise TimeoutError('lost-ack')
        store.rpc = lost_ack
        with pytest.raises(TimeoutError, match='lost-ack'):
            store.end_session('owned', 'agent_close')
        ended = get_session(holder, 'owned')
        store.rpc = original
        store.retry_pending()
        store.end_session('owned', 'late-reason')
        assert get_session(holder, 'owned')['ended_at'] == ended['ended_at']
        assert get_session(holder, 'owned')['end_reason'] == 'agent_close'
        assert store.finish()['status'] == 'terminal'
        assert get_session(holder, 'foreign')['system_prompt'] == 'private'
    finally:
        store.close()


def test_late_finalizers_cannot_mutate_successor_or_foreign_rows(backend, scope_a, tmp_path):
    holder, store, scope = setup_store(backend.open(scope_a), tmp_path)
    try:
        store.end_session('owned', 'agent_close')
        reopen_session(holder, 'owned')
        holder._execute_write(lambda conn: conn.execute(
            'UPDATE sessions SET runtime_generation=runtime_generation+1 WHERE id=?', ('owned',)))
        # Even replay of a formerly successful receipt must revalidate generation.
        for sequence in (1, 2):
            with pytest.raises(AuthorityRefusal, match='stale_generation'):
                mutate_worker_execution(holder, **scope, sequence=sequence,
                                        operation='session.end', payload={'end_reason': 'agent_close'})
        with pytest.raises(AuthorityRefusal, match='permission_denied'):
            mutate_worker_execution(holder, **dict(scope, session_id='foreign'), sequence=2,
                                    operation='session.end', payload={'end_reason': 'agent_close'})
        assert get_session(holder, 'owned')['ended_at'] is None
        assert get_session(holder, 'foreign')['ended_at'] is None
    finally:
        store.close()
