"""Watermarked compaction and rotation share the worker receipt transaction."""
import pytest

from tests.state.test_runtime_worker_compression import worker  # noqa: F401


def test_watermarked_compaction_preserves_foreign_tail_and_rejects_stale_holder(worker):
    db, store = worker
    store.append_messages_batch('owned', [{'role': 'user', 'content': 'old'}, {'role': 'assistant', 'content': 'carry'}])
    assert store.try_acquire_compression_lock('owned', 'holder')
    watermark = store.get_active_message_watermark('owned')
    db.append_messages_batch('owned', [{'role': 'user', 'content': 'foreign tail', 'api_content': ' EXACT ',
                                       'display_metadata': {'_accepted_input_id': 'input-tail'}}])
    seq = store.journal['next_sequence']
    from hermes_state_runtime import mutate_worker_execution
    with pytest.raises(Exception, match='lease'):
        mutate_worker_execution(db, **store.scope, sequence=seq, operation='compression.archive', payload={
            'messages': [{'role': 'assistant', 'content': 'bad'}], 'model_config_patch': None,
            'watermark': watermark, 'lock_holder': 'old', 'tail_count': 0})
    assert store.archive_and_compact('owned', [{'role': 'assistant', 'content': 'summary'},
        {'role': 'assistant', 'content': 'carry'}], watermark=watermark, lock_holder='holder', tail_count=1,
        model_config_patch={'_proactive_prune_rearm_tokens': 42}) == 3
    rows = store.get_messages_as_conversation('owned', include_row_ids=True)
    assert rows[-1]['content'] == 'foreign tail' and rows[-1]['api_content'] == ' EXACT '
    assert rows[-1]['display_metadata'] == {'_accepted_input_id': 'input-tail'}
    assert db._read_one("SELECT COUNT(*) FROM messages WHERE session_id='owned' AND compacted=1")[0] == 1
    assert store.get_session_model_config_value('owned', '_proactive_prune_rearm_tokens') == 42


def test_publication_replay_advances_assignment_once_and_keeps_markers(worker):
    db, store = worker
    store.append_messages_batch('owned', [{'role': 'user', 'content': 'before'}])
    assert store.try_acquire_compression_lock('owned', 'holder')
    watermark = store.get_active_message_watermark('owned')
    db.append_messages_batch('owned', [{'role': 'user', 'content': 'foreign', 'api_content': ' exact tail ',
                                       'display_metadata': {'_accepted_input_id': 'delivery'}}])
    original = store.rpc
    def lost_ack(method, **params):
        original(method, **params)
        raise TimeoutError('lost-ack')
    store.rpc = lost_ack
    with pytest.raises(TimeoutError, match='lost-ack'):
        store.publish_compression_child(parent_session_id='owned', child_session_id='child', source='cli',
            messages=[{'role': 'assistant', 'content': 'summary'}], system_prompt='prefix',
            compression_lock_holder='holder', watermark=watermark)
    assert db._read_one('SELECT session_id FROM worker_executions')[0] == 'child'
    assert db.get_session('owned')['end_reason'] == 'compression'
    store.rpc = original
    store.retry_pending()
    assert store.scope['session_id'] == store.journal['scope']['session_id'] == 'child'
    store.append_messages_batch('child', [{'role': 'assistant', 'content': 'after'}])
    rows = store.get_messages_as_conversation('child')
    assert [r['content'] for r in rows] == ['summary', 'foreign', 'after']
    assert rows[1]['api_content'] == ' exact tail '
    assert rows[1]['display_metadata'] == {'_accepted_input_id': 'delivery'}
    assert store.get_compression_lineage('child') == ['owned', 'child']
    assert db.get_session('child')['system_prompt'] == 'prefix'
