"""Read receipts preserve replay markers and distinct cache/attribution roots."""
from types import SimpleNamespace

import pytest

from agent.prompt_cache_scope import resolve_prompt_cache_scope
from persistence.sqlite.authority._runtime_ends import begin_runtime_epoch
from persistence.sqlite.conversation._store.compression import get_compression_lineage, get_compression_tip
from persistence.sqlite.conversation._store.messages import (
    append_messages_batch, get_messages_as_conversation, latest_conversation_boundary, resolve_resume_session_id,
)
from persistence.sqlite.conversation._store.sessions import create_session, end_session
from persistence.sqlite.workers._executions import register_worker_execution
from persistence.sqlite.workers._receipt import mutate_worker_execution
from tests.persistence_sqlite.workers._runtime_worker import worker_store


def test_history_and_lineage_keep_cache_scope_distinct_from_attribution(backend, scope_a, tmp_path):
    holder = backend.open(scope_a).holder()
    create_session(holder, 'root', 'cli', system_prompt='root prefix')
    create_session(holder, 'child', 'cli', parent_session_id='root',
                   model_config={'_delegate_from': 'root'}, system_prompt='child prefix')
    end_session(holder, 'child', 'compression')
    create_session(holder, 'tip', 'cli', parent_session_id='child', model_config={'_delegate_from': 'root'},
                   system_prompt='child prefix')
    create_session(holder, 'foreign', 'cli', system_prompt='secret')
    append_messages_batch(holder, 'tip', [
        {'role': 'user', 'content': 'canonical', 'api_content': ' EXACT\nWIRE ', 'platform_message_id': 'delivery'},
        {'role': 'assistant', 'content': 'answer', 'reasoning': 'reason', 'codex_message_items': [{'type': 'message'}]}])
    epoch = begin_runtime_epoch(holder, instance_id='fixture')
    scope = dict(epoch=epoch, execution_id='worker', session_id='tip', generation=0)
    register_worker_execution(holder, **scope, kind='compute', adoption_secret='secret')
    store = worker_store(holder, scope, tmp_path / 'outbox')
    try:
        assert store.get_messages_as_conversation('tip', include_row_ids=True) == get_messages_as_conversation(holder, 'tip', include_row_ids=True)
        assert store.get_compression_lineage('tip') == get_compression_lineage(holder, 'tip') == ['child', 'tip']
        assert store.get_conversation_root('tip') == 'root'
        assert store.get_session('root')['system_prompt'] == 'root prefix'
        assert store.get_compression_tip('child') == get_compression_tip(holder, 'child')
        assert store.resolve_resume_session_id('child') == resolve_resume_session_id(holder, 'child')
        assert resolve_prompt_cache_scope(SimpleNamespace(session_id='tip', _session_db=store)) == 'child'
        assert store.declared_scope_identity('child') == (True, 'cli')
        seq = store.journal['next_sequence']
        with pytest.raises(Exception, match='permission_denied'):
            mutate_worker_execution(holder, **scope, sequence=seq, operation='compression.context', payload={'target': 'foreign'})
        assert holder._read_one('SELECT last_sequence FROM worker_executions')[0] == seq - 1
    finally:
        store.close()


def test_declared_cache_boundary_matches_local_database(backend, scope_a, tmp_path):
    persistence = backend.open(scope_a)
    holder = persistence.holder()
    create_session(holder, 'owned', 'cli', session_key='chat')
    end_session(holder, 'owned', 'session_reset')
    create_session(holder, 'next', 'cli', session_key='chat')
    epoch = begin_runtime_epoch(holder, instance_id='fixture')
    scope = dict(epoch=epoch, execution_id='worker', session_id='next', generation=0)
    register_worker_execution(holder, **scope, kind='compute', adoption_secret='secret')
    store = worker_store(holder, scope, tmp_path / 'outbox')
    try:
        assert store.latest_conversation_boundary('chat', 'cli') == latest_conversation_boundary(holder, 'chat', 'cli')
        def agent(handle, interface=None):
            return SimpleNamespace(session_id='next', _session_db=handle, _gateway_session_key='chat', persistence=interface)
        # The local writer reads through its storage interface handle.
        assert resolve_prompt_cache_scope(agent(store)) == resolve_prompt_cache_scope(agent(holder, persistence))
        with pytest.raises(Exception, match='permission_denied'):
            store.latest_conversation_boundary('foreign-chat', 'cli')
    finally:
        store.failure = None
        store.journal['pending'] = []
        store.close()


def test_terminal_worker_keeps_no_full_history_receipt(backend, scope_a):
    """Every managed turn reads compression.history; once execution.finish (or settlement)
    makes the execution terminal, that full-transcript copy must not stay in worker_receipts,
    while its digest still makes a late duplicate with other content conflict."""
    from persistence.core.errors import AuthorityRefusal
    holder = backend.open(scope_a).holder()
    create_session(holder, 's', 'cli')
    append_messages_batch(holder, 's', [{'role': 'user', 'content': 'x' * 2000}])
    epoch = begin_runtime_epoch(holder, instance_id='fixture')
    scope = dict(epoch=epoch, execution_id='worker', session_id='s', generation=0)
    register_worker_execution(holder, **scope, kind='compute', adoption_secret='secret')
    read = dict(target='s', include_ancestors=False, include_inactive=False, repair_alternation=False,
                include_row_ids=False, include_compacted=False)
    history = mutate_worker_execution(holder, **scope, sequence=1, operation='compression.history', payload=read)
    assert history['messages'][0]['content'] == 'x' * 2000
    mutate_worker_execution(holder, **scope, sequence=2, operation='execution.finish', payload={})
    stored = holder._read_all('SELECT sequence, LENGTH(result_json) FROM worker_receipts ORDER BY sequence')
    assert stored[0][1] < 200, 'a terminal execution kept a full-history receipt'
    with pytest.raises(AuthorityRefusal, match='admission_conflict'):
        mutate_worker_execution(holder, **scope, sequence=1, operation='compression.history',
                                payload=dict(read, include_row_ids=True))
    with pytest.raises(AuthorityRefusal, match='stale_generation'):
        mutate_worker_execution(holder, **scope, sequence=1, operation='compression.history', payload=read)
    assert mutate_worker_execution(holder, **scope, sequence=2, operation='execution.finish',
                                   payload={}) == {'status': 'terminal'}
