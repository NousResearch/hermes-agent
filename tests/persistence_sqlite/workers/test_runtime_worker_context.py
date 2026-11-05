"""Constructor context is assignment-scoped and shares durable mutation receipts."""
import pytest

from agent.runtime_session_store import WorkerPersistenceError
from persistence.sqlite.authority._runtime_ends import begin_runtime_epoch
from persistence.sqlite.conversation._store.compression import (
    get_compression_failure_cooldown_row, record_compression_failure_cooldown, set_compression_fallback_streak,
    set_compression_ineffective_count, set_compression_recovery_deadline,
)
from persistence.sqlite.conversation._store.sessions import create_session, get_session, get_session_model_config_value
from persistence.sqlite.conversation._store.titles import get_session_title, set_session_title
from persistence.sqlite.workers._executions import register_worker_execution
from persistence.sqlite.workers._receipt import mutate_worker_execution
from tests.persistence_sqlite.workers._runtime_worker import worker_store


def test_worker_context_preserves_authority_identity_and_prompt_receipts(backend, scope_a, tmp_path):
    holder = backend.open(scope_a).holder()
    create_session(holder, 'owned', 'subagent', parent_session_id=None, model='requested',
                   model_config={'_delegate_from': 'parent'}, system_prompt='prefix', profile_name='fixture')
    create_session(holder, 'foreign', 'cli', system_prompt='private')
    epoch = begin_runtime_epoch(holder, instance_id='fixture')
    scope = dict(epoch=epoch, execution_id='worker', session_id='owned', generation=0)
    register_worker_execution(holder, **scope, kind='child', adoption_secret='test-secret')
    store = worker_store(holder, scope, tmp_path / 'outbox')
    try:
        assert store.get_session('owned')['system_prompt'] == 'prefix'
        with pytest.raises(WorkerPersistenceError, match='permission_denied'):
            store.get_session('foreign')
        store.update_system_prompt('owned', 'stable-prefix')
        store.patch_session_model_config('owned', {'_usage_anchor': {'tokens': 23}})
        assert store.get_session_model_config_value('owned', '_usage_anchor') == {'tokens': 23}
        assert get_session_model_config_value(holder, 'owned', '_delegate_from') == 'parent'
        assert get_session(holder, 'foreign')['system_prompt'] == 'private'
        original = store.rpc
        def lost_ack(method, **params):
            original(method, **params)
            raise TimeoutError('lost-ack')
        store.rpc = lost_ack
        with pytest.raises(TimeoutError, match='lost-ack'):
            store.update_system_prompt('owned', 'durable-prefix')
        store.rpc = original
        assert len(store.retry_pending()) == 1
        assert get_session(holder, 'owned')['system_prompt'] == 'durable-prefix'
        with pytest.raises(Exception, match='invalid_params'):
            mutate_worker_execution(holder, **scope, sequence=store.journal['next_sequence'],
                                    operation='session.sidecars', payload={'patch': {'_delegate_from': 'foreign'}})
    finally:
        store.close()


def test_worker_constructor_restores_persisted_compressor_guards(backend, scope_a, tmp_path):
    import time
    from agent.context_compressor import ContextCompressor
    holder = backend.open(scope_a).holder()
    create_session(holder, 'owned', 'cli', model_config={'_proactive_prune_rearm_tokens': 1234})
    set_compression_fallback_streak(holder, 'owned', 4)
    set_compression_ineffective_count(holder, 'owned', 3)
    deadline = time.time() + 3600
    set_compression_recovery_deadline(holder, 'owned', deadline)
    record_compression_failure_cooldown(holder, 'owned', deadline, 'fixture-cooldown')
    epoch = begin_runtime_epoch(holder, instance_id='fixture')
    scope = dict(epoch=epoch, execution_id='worker', session_id='owned', generation=0)
    register_worker_execution(holder, **scope, kind='compute', adoption_secret='test-secret')
    store = worker_store(holder, scope, tmp_path / 'outbox')
    try:
        compressor = ContextCompressor(model='fixture', config_context_length=100000, quiet_mode=True)
        compressor.bind_session_state(store, 'owned')
        assert compressor._fallback_compression_streak == 4
        assert compressor._ineffective_compression_count == 3
        assert compressor._anti_thrash_recovery_deadline == deadline
        assert compressor._proactive_prune_rearm_tokens == 1234
        assert store.get_compression_failure_cooldown_row('owned') == get_compression_failure_cooldown_row(holder, 'owned')
        assert store.get_compression_failure_cooldown('owned')['cooldown_until'] == deadline
        set_session_title(holder, 'owned', 'retained-title')
        assert store.get_session_title('owned') == get_session_title(holder, 'owned')
    finally:
        store.close()
