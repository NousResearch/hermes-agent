"""Ranked labels and exact provider-side data share assignment receipts."""
import time

import pytest

from hermes_state_runtime import RuntimeStoreError, mutate_worker_execution
from tests.state.test_runtime_worker_lifecycle import setup_store


def test_worker_labels_keep_rank_and_reject_late_generations(tmp_path):
    db, store, scope = setup_store(tmp_path)
    try:
        assert store.set_auto_title('owned', 'derived', source='derived')
        assert store.set_auto_title('owned', 'generated', source='llm')
        assert not store.set_auto_title('owned', 'late-derived', source='derived')
        assert store.set_session_title('owned', 'manual')
        assert not store.set_auto_title('owned', 'late-llm', source='llm')
        assert store.get_session_title_source('owned') == 'user'
        assert store.get_next_title_in_lineage('manual') == db.get_next_title_in_lineage('manual')
        assert store.set_session_title_source('owned', 'llm')
        now = time.time() + 10
        store.touch_session_activity('owned', now, description='  active   work ', provenance='unknown')
        expected = db.get_session('owned')
        store.touch_session_activity('owned', now - 1, description='older')
        assert db.get_session('owned')['last_activity_description'] == expected['last_activity_description']
        store.clear_session_activity_labels('owned')
        assert db.get_session('owned')['last_activity_at'] == now
        assert db.get_session('owned')['last_activity_description'] == ''
        store.touch_session_activity('owned', now + 1, description='successor')
        db._execute_write(lambda conn: conn.execute(
            'UPDATE sessions SET runtime_generation=runtime_generation+1 WHERE id=?', ('owned',)))
        for operation, payload in [('session.title', {'title': 'late', 'source': 'user'}),
                                   ('session.title_source', {'source': 'derived'}),
                                   ('session.activity_clear', {}),
                                   ('session.activity', {'ts': now + 2, 'description': 'late', 'provenance': None})]:
            with pytest.raises(RuntimeStoreError, match='stale_generation'):
                mutate_worker_execution(db, **scope, sequence=store.journal['next_sequence'],
                                        operation=operation, payload=payload)
        assert db.get_session('owned')['title'] == 'manual'
        assert db.get_session('owned')['last_activity_description'] == 'successor'
    finally:
        store.close()
        db.close()


def test_worker_route_and_api_content_preserve_exact_usage_and_target_fences(tmp_path):
    db, store, scope = setup_store(tmp_path)
    try:
        store.update_system_prompt('owned', 'stable-prefix')
        store.queue_token_counts('owned', input_tokens=11, output_tokens=7, model='first',
                                 billing_provider='provider-one', billing_base_url='https://first.invalid', api_call_count=1)
        store.update_session_billing_route('owned', provider='provider-two', base_url='https://second.invalid',
                                           billing_mode='api')
        store.queue_token_counts('owned', input_tokens=13, output_tokens=3, model='second',
                                 billing_provider='provider-two', billing_base_url='https://second.invalid', api_call_count=1)
        row = db.get_session('owned')
        assert (row['input_tokens'], row['output_tokens']) == (24, 10)
        assert row['billing_provider'] == 'provider-two'
        assert row['system_prompt'] is None
        store.append_messages_batch('owned', [{'role': 'user', 'content': [{'type': 'text', 'text': 'visible'}]}])
        assert store.set_latest_user_api_content('owned', 'wrong', 'must-not-land') == 0
        assert store.set_latest_user_api_content('owned', [{'type': 'text', 'text': 'visible'}], 'wire-only') == 1
        with db._read_ctx() as conn:
            assert conn.execute('SELECT api_content FROM messages WHERE session_id=?', ('owned',)).fetchone()[0] == 'wire-only'
            usage = [tuple(r) for r in conn.execute(
                'SELECT model,billing_provider,input_tokens,output_tokens FROM session_model_usage WHERE session_id=? ORDER BY model', ('owned',))]
        assert usage == [('first', 'provider-one', 11, 7), ('second', 'provider-two', 13, 3)]
        original = store.rpc
        def lost_ack(method, **params):
            original(method, **params)
            raise TimeoutError('lost-ack')
        store.rpc = lost_ack
        with pytest.raises(TimeoutError, match='lost-ack'):
            store.update_session_billing_route('owned', provider='third', base_url='https://third.invalid')
        store.rpc = original
        store.retry_pending()
        assert db.get_session('owned')['billing_provider'] == 'third'
        for operation, payload in [('session.billing_route', {'provider': 'evil', 'base_url': 'evil', 'billing_mode': None}),
                                   ('session.api_content', {'content': 'private', 'api_content': 'evil'}),
                                   ('session.title', {'title': 'evil', 'source': 'user'}),
                                   ('session.activity_clear', {})]:
            with pytest.raises(RuntimeStoreError, match='permission_denied'):
                mutate_worker_execution(db, **dict(scope, session_id='foreign'),
                                        sequence=store.journal['next_sequence'], operation=operation, payload=payload)
        assert db.get_session('foreign')['system_prompt'] == 'private'
    finally:
        store.close()
        db.close()
