"""Read receipts preserve replay markers and distinct cache/attribution roots."""
from types import SimpleNamespace

import pytest

from agent.prompt_cache_scope import resolve_prompt_cache_scope
from agent.runtime_session_store import RuntimeSessionStore
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch, mutate_worker_execution, register_worker_execution


def test_history_and_lineage_keep_cache_scope_distinct_from_attribution(tmp_path):
    db = SessionDB(tmp_path / 'state.db')
    try:
        db.create_session('root', 'cli', system_prompt='root prefix')
        db.create_session('child', 'cli', parent_session_id='root',
                          model_config={'_delegate_from': 'root'}, system_prompt='child prefix')
        db.end_session('child', 'compression')
        db.create_session('tip', 'cli', parent_session_id='child', model_config={'_delegate_from': 'root'},
                          system_prompt='child prefix')
        db.create_session('foreign', 'cli', system_prompt='secret')
        db.append_messages_batch('tip', [
            {'role': 'user', 'content': 'canonical', 'api_content': ' EXACT\nWIRE ', 'platform_message_id': 'delivery'},
            {'role': 'assistant', 'content': 'answer', 'reasoning': 'reason', 'codex_message_items': [{'type': 'message'}]}])
        epoch = begin_runtime_epoch(db, instance_id='fixture')
        scope = dict(epoch=epoch, execution_id='worker', session_id='tip', generation=0)
        register_worker_execution(db, **scope, kind='compute', adoption_secret='secret')
        store = RuntimeSessionStore(lambda method, **p: mutate_worker_execution(db, **p), scope, tmp_path / 'outbox')
        try:
            assert store.get_messages_as_conversation('tip', include_row_ids=True) == db.get_messages_as_conversation('tip', include_row_ids=True)
            assert store.get_compression_lineage('tip') == db.get_compression_lineage('tip') == ['child', 'tip']
            assert store.get_conversation_root('tip') == 'root'
            assert store.get_session('root')['system_prompt'] == 'root prefix'
            assert store.get_compression_tip('child') == db.get_compression_tip('child')
            assert store.resolve_resume_session_id('child') == db.resolve_resume_session_id('child')
            assert resolve_prompt_cache_scope(SimpleNamespace(session_id='tip', _session_db=store)) == 'child'
            assert store.declared_scope_identity('child') == (True, 'cli')
            seq = store.journal['next_sequence']
            with pytest.raises(Exception, match='permission_denied'):
                mutate_worker_execution(db, **scope, sequence=seq, operation='compression.context', payload={'target': 'foreign'})
            assert db._read_one('SELECT last_sequence FROM worker_executions')[0] == seq - 1
        finally:
            store.close()
    finally:
        db.close()


def test_declared_cache_boundary_matches_local_database(tmp_path):
    db = SessionDB(tmp_path / 'state.db')
    try:
        db.create_session('owned', 'cli', session_key='chat')
        db.end_session('owned', 'session_reset')
        db.create_session('next', 'cli', session_key='chat')
        epoch = begin_runtime_epoch(db, instance_id='fixture')
        scope = dict(epoch=epoch, execution_id='worker', session_id='next', generation=0)
        register_worker_execution(db, **scope, kind='compute', adoption_secret='secret')
        store = RuntimeSessionStore(lambda method, **p: mutate_worker_execution(db, **p), scope, tmp_path / 'outbox')
        try:
            assert store.latest_conversation_boundary('chat', 'cli') == db.latest_conversation_boundary('chat', 'cli')
            def agent(handle):
                return SimpleNamespace(session_id='next', _session_db=handle, _gateway_session_key='chat')
            assert resolve_prompt_cache_scope(agent(store)) == resolve_prompt_cache_scope(agent(db))
            with pytest.raises(Exception, match='permission_denied'):
                store.latest_conversation_boundary('foreign-chat', 'cli')
        finally:
            store.failure = None
            store.journal['pending'] = []
            store.close()
    finally:
        db.close()


def test_terminal_worker_keeps_no_full_history_receipt(tmp_path):
    """N19: every managed turn reads compression.history; once execution.finish (or settlement)
    makes the execution terminal, that full-transcript copy must not stay in worker_receipts,
    while its digest still makes a late duplicate with other content conflict."""
    from hermes_state_runtime import RuntimeStoreError
    db = SessionDB(tmp_path / 'state.db')
    try:
        db.create_session('s', 'cli')
        db.append_messages_batch('s', [{'role': 'user', 'content': 'x' * 2000}])
        epoch = begin_runtime_epoch(db, instance_id='fixture')
        scope = dict(epoch=epoch, execution_id='worker', session_id='s', generation=0)
        register_worker_execution(db, **scope, kind='compute', adoption_secret='secret')
        read = dict(target='s', include_ancestors=False, include_inactive=False, repair_alternation=False,
                    include_row_ids=False, include_compacted=False)
        history = mutate_worker_execution(db, **scope, sequence=1, operation='compression.history', payload=read)
        assert history['messages'][0]['content'] == 'x' * 2000
        mutate_worker_execution(db, **scope, sequence=2, operation='execution.finish', payload={})
        stored = db._read_all('SELECT sequence, LENGTH(result_json) FROM worker_receipts ORDER BY sequence')
        assert stored[0][1] < 200, 'a terminal execution kept a full-history receipt'
        with pytest.raises(RuntimeStoreError, match='admission_conflict'):
            mutate_worker_execution(db, **scope, sequence=1, operation='compression.history',
                                    payload=dict(read, include_row_ids=True))
        with pytest.raises(RuntimeStoreError, match='stale_generation'):
            mutate_worker_execution(db, **scope, sequence=1, operation='compression.history', payload=read)
        assert mutate_worker_execution(db, **scope, sequence=2, operation='execution.finish',
                                       payload={}) == {'status': 'terminal'}
    finally:
        db.close()
