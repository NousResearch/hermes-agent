"""Ranked labels and exact provider-side data share assignment receipts."""
import time

import pytest

from persistence.core.errors import AuthorityRefusal
from persistence.sqlite.conversation._store.messages import append_message, append_messages_batch, archive_and_compact
from persistence.sqlite.conversation._store.sessions import create_session, get_session
from persistence.sqlite.conversation._store.titles import get_next_title_in_lineage
from persistence.sqlite.workers._receipt import mutate_worker_execution
from tests.persistence_sqlite.workers._runtime_worker import setup_store
from tests.persistence_sqlite.workers._runtime_worker import worker as worker  # noqa: PLC0414 -- the self-alias is load-bearing: it suppresses F811 for the pytest fixture parameter shadowing this import


def test_worker_labels_keep_rank_and_reject_late_generations(backend, scope_a, tmp_path):
    holder, store, scope = setup_store(backend.open(scope_a), tmp_path)
    try:
        assert store.set_auto_title('owned', 'derived', source='derived')
        assert store.set_auto_title('owned', 'generated', source='llm')
        assert not store.set_auto_title('owned', 'late-derived', source='derived')
        assert store.set_session_title('owned', 'manual')
        assert not store.set_auto_title('owned', 'late-llm', source='llm')
        assert store.get_session_title_source('owned') == 'user'
        assert store.get_next_title_in_lineage('manual') == get_next_title_in_lineage(holder, 'manual')
        assert store.set_session_title_source('owned', 'llm')
        now = time.time() + 10
        store.touch_session_activity('owned', now, description='  active   work ', provenance='unknown')
        expected = get_session(holder, 'owned')
        store.touch_session_activity('owned', now - 1, description='older')
        assert get_session(holder, 'owned')['last_activity_description'] == expected['last_activity_description']
        store.clear_session_activity_labels('owned')
        assert get_session(holder, 'owned')['last_activity_at'] == now
        assert get_session(holder, 'owned')['last_activity_description'] == ''
        store.touch_session_activity('owned', now + 1, description='successor')
        holder._execute_write(lambda conn: conn.execute(
            'UPDATE sessions SET runtime_generation=runtime_generation+1 WHERE id=?', ('owned',)))
        for operation, payload in [('session.title', {'title': 'late', 'source': 'user'}),
                                   ('session.title_source', {'source': 'derived'}),
                                   ('session.activity_clear', {}),
                                   ('session.activity', {'ts': now + 2, 'description': 'late', 'provenance': None})]:
            with pytest.raises(AuthorityRefusal, match='stale_generation'):
                mutate_worker_execution(holder, **scope, sequence=store.journal['next_sequence'],
                                        operation=operation, payload=payload)
        assert get_session(holder, 'owned')['title'] == 'manual'
        assert get_session(holder, 'owned')['last_activity_description'] == 'successor'
    finally:
        store.close()


def test_worker_route_and_api_content_preserve_exact_usage_and_target_fences(backend, scope_a, tmp_path):
    holder, store, scope = setup_store(backend.open(scope_a), tmp_path)
    try:
        store.update_system_prompt('owned', 'stable-prefix')
        store.queue_token_counts('owned', input_tokens=11, output_tokens=7, model='first',
                                 billing_provider='provider-one', billing_base_url='https://first.invalid', api_call_count=1)
        store.update_session_billing_route('owned', provider='provider-two', base_url='https://second.invalid',
                                           billing_mode='api')
        store.queue_token_counts('owned', input_tokens=13, output_tokens=3, model='second',
                                 billing_provider='provider-two', billing_base_url='https://second.invalid', api_call_count=1)
        row = get_session(holder, 'owned')
        assert (row['input_tokens'], row['output_tokens']) == (24, 10)
        assert row['billing_provider'] == 'provider-two'
        # Direct-writer parity: a route write never touches the frozen prompt snapshot.
        assert row['system_prompt'] == 'stable-prefix'
        with holder._read_ctx() as conn:
            assert conn.execute('SELECT COUNT(*) FROM system_prompts WHERE prompt=?', ('stable-prefix',)).fetchone()[0] == 1
        store.append_messages_batch('owned', [{'role': 'user', 'content': [{'type': 'text', 'text': 'visible'}]}])
        assert store.set_latest_user_api_content('owned', 'wrong', 'must-not-land') == 0
        assert store.set_latest_user_api_content('owned', [{'type': 'text', 'text': 'visible'}], 'wire-only') == 1
        with holder._read_ctx() as conn:
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
        assert get_session(holder, 'owned')['billing_provider'] == 'third'
        for operation, payload in [('session.billing_route', {'provider': 'evil', 'base_url': 'evil', 'billing_mode': None}),
                                   ('session.api_content', {'content': 'private', 'api_content': 'evil'}),
                                   ('session.title', {'title': 'evil', 'source': 'user'}),
                                   ('session.activity_clear', {})]:
            with pytest.raises(AuthorityRefusal, match='permission_denied'):
                mutate_worker_execution(holder, **dict(scope, session_id='foreign'),
                                        sequence=store.journal['next_sequence'], operation=operation, payload=payload)
        assert get_session(holder, 'foreign')['system_prompt'] == 'private'
    finally:
        store.close()


def test_turn_prologue_row_backfills_land_through_the_worker_like_direct(worker):
    """The real prologue (agent/turn_context.py) updates an already-materialized user row by ``_row_id``:
    the api_content sidecar after in-place preflight compaction, and the multimodal context part. The
    managed facade must implement both row-addressed ops, fenced to its own session's active user rows,
    or the durable row silently diverges from what the model saw (helix4u #4)."""
    import threading
    import types
    from agent.turn_context import _append_multimodal_context, _stamp_api_content_sidecar
    persistence, store = worker
    holder = persistence.holder()
    create_session(holder, 'direct', 'cli')

    def newest_user(sid):
        with holder._read_ctx() as conn:
            return tuple(conn.execute("SELECT content, api_content FROM messages WHERE session_id=? AND role='user' "
                                      "ORDER BY id DESC LIMIT 1", (sid,)).fetchone())

    durable = {}
    # The direct writer's backfills go through its storage interface handle; the worker's through its owner.
    for sid, owner, interface, compact, append in (
            ('direct', holder, persistence, lambda sid, rows, **kw: archive_and_compact(holder, sid, rows, **kw),
             lambda sid, rows: append_messages_batch(holder, sid, rows)),
            ('owned', store, None, store.archive_and_compact, store.append_messages_batch)):
        agent = types.SimpleNamespace(_session_db=owner, persistence=interface, session_id=sid,
                                      _session_persist_lock=threading.Lock(),
                                      _persist_user_message_override=None, _last_compaction_in_place=True)
        text = [{'role': 'user', 'content': 'question'}]
        compact(sid, text, tail_count=0)
        _stamp_api_content_sidecar(agent, text, 0, '', 'PLUGIN CONTEXT', preflight_compressed=True)
        sidecar = newest_user(sid)
        append(sid, [{'role': 'assistant', 'content': 'ok'}])
        parts = [{'role': 'user', 'content': [{'type': 'text', 'text': 'look'},
                                               {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}}]}]
        append(sid, parts)
        _append_multimodal_context(agent, parts[0], '', 'PLUGIN CONTEXT', preflight_compressed=False)
        durable[sid] = (sidecar, newest_user(sid)[0])
    assert durable['owned'] == durable['direct'] == (('question', 'question\n\nPLUGIN CONTEXT'),
                                                     'look\n[screenshot]\nPLUGIN CONTEXT')
    foreign = append_message(holder, 'foreign', 'user', 'private')
    assert store.set_user_message_content('owned', foreign, 'overwritten') == 0
    assert store.set_message_api_content('owned', foreign, 'private', 'injected') == 0
    assert newest_user('foreign') == ('private', None)
