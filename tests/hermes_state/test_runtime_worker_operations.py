"""Worker receipts cover the real transcript serializer, not a text-only substitute."""
import pytest

from hermes_state import SessionDB
from hermes_state_runtime import RuntimeStoreError, begin_runtime_epoch, register_worker_execution


def test_structured_worker_receipts_are_atomic_and_scoped(tmp_path):
    import hermes_state_runtime as runtime
    assert callable(getattr(runtime, 'mutate_worker_execution', None))
    db = SessionDB(tmp_path / 'state.db')
    try:
        db.create_session('assigned', source='cli')
        db.create_session('foreign', source='cli')
        epoch = begin_runtime_epoch(db, instance_id='owner')
        register_worker_execution(db, epoch=epoch, execution_id='worker', session_id='assigned',
                                  generation=0, kind='compute', adoption_secret='private')
        scope = dict(epoch=epoch, execution_id='worker', session_id='assigned', generation=0)
        def apply(seq, op, payload, **overrides):
            return runtime.mutate_worker_execution(db, **(scope | overrides), sequence=seq,
                                                   operation=op, payload=payload)
        assert apply(1, 'turn.acquire', {'holder': 'worker-lease', 'ttl_seconds': 300})['value']
        messages = [
            {'role': 'user', 'content': [{'type': 'text', 'text': 'hello\ud800'}], 'api_content': 'wire'},
            {'role': 'assistant', 'content': None, 'reasoning': 'private reason',
             'tool_calls': [{'id': 'call1', 'type': 'function', 'function': {'name': 'terminal', 'arguments': '{}'}}]},
            {'role': 'tool', 'content': 'tool-marker', 'tool_call_id': 'call1', 'tool_name': 'terminal'},
        ]
        payload = {'messages': messages, 'turn_lease_holder': 'worker-lease'}
        receipt = apply(2, 'transcript.append', payload)
        assert receipt['count'] == 3
        assert all(m['_row_id'] > 0 for m in receipt['annotations'])
        assert apply(2, 'transcript.append', payload) == receipt
        with pytest.raises(RuntimeStoreError, match='admission_conflict'):
            apply(2, 'transcript.append', {'messages': [{'role': 'user', 'content': 'different'}]})
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            apply(3, 'transcript.append', payload, session_id='foreign')
        with pytest.raises(RuntimeStoreError, match='stale_generation'):
            apply(3, 'transcript.append', payload, generation=1)
        assert len(db.get_messages('assigned')) == 3
        stored = db.get_messages('assigned')
        assert stored[1]['reasoning'] == 'private reason'
        assert stored[2]['tool_call_id'] == 'call1'
        assert stored[2]['content'] == 'tool-marker'
        with pytest.raises(RuntimeStoreError, match='invalid_params'):
            apply(3, 'SQL', {'sql': 'DELETE FROM messages'})
        assert apply(3, 'turn.renew', {'holder': 'worker-lease', 'ttl_seconds': 300})['value']
        assert apply(4, 'turn.release', {'holder': 'worker-lease'}) == {'value': None}
        assistant = {'role': 'assistant', 'content': [{'type': 'text', 'text': 'canonical winner\ud800'}]}
        winner = apply(5, 'transcript.append', {'messages': [assistant]})
        repaired = apply(6, 'transcript.append', {'messages': [{'role': 'assistant',
            'content': 'loser', '_row_id': winner['annotations'][0]['_row_id']}]})
        assert repaired['count'] == 0
        assert repaired['annotations'][0]['_canonical_content'] == assistant['content']
        terminal = apply(7, 'execution.finish', {})
        assert terminal['status'] == 'terminal'
        assert apply(7, 'execution.finish', {}) == terminal
        with pytest.raises(RuntimeStoreError, match='stale_generation'):
            apply(8, 'usage.main', {'input_tokens': 1})
    finally:
        db.close()
