"""Worker receipts cover the real transcript serializer, not a text-only substitute."""
import pytest

from persistence.core.errors import AuthorityRefusal
from persistence.sqlite.authority._runtime_ends import begin_runtime_epoch
from persistence.sqlite.conversation._store.messages import get_messages
from persistence.sqlite.conversation._store.sessions import create_session
from persistence.sqlite.workers._executions import register_worker_execution
from agent.message_metadata import CANONICAL_ROW, DB_ROW_SNAPSHOT


def test_structured_worker_receipts_are_atomic_and_scoped(backend, scope_a):
    from persistence.sqlite.workers import _receipt as runtime
    assert callable(getattr(runtime, 'mutate_worker_execution', None))
    holder = backend.open(scope_a).holder()
    create_session(holder, 'assigned', 'cli')
    create_session(holder, 'foreign', 'cli')
    epoch = begin_runtime_epoch(holder, instance_id='owner')
    register_worker_execution(holder, epoch=epoch, execution_id='worker', session_id='assigned',
                              generation=0, kind='compute', adoption_secret='private')
    scope = dict(epoch=epoch, execution_id='worker', session_id='assigned', generation=0)
    def apply(seq, op, payload, **overrides):
        return runtime.mutate_worker_execution(holder, **(scope | overrides), sequence=seq,
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
    with pytest.raises(AuthorityRefusal, match='admission_conflict'):
        apply(2, 'transcript.append', {'messages': [{'role': 'user', 'content': 'different'}]})
    with pytest.raises(AuthorityRefusal, match='permission_denied'):
        apply(3, 'transcript.append', payload, session_id='foreign')
    with pytest.raises(AuthorityRefusal, match='stale_generation'):
        apply(3, 'transcript.append', payload, generation=1)
    assert len(get_messages(holder, 'assigned')) == 3
    stored = get_messages(holder, 'assigned')
    assert stored[1]['reasoning'] == 'private reason'
    assert stored[2]['tool_call_id'] == 'call1'
    assert stored[2]['content'] == 'tool-marker'
    with pytest.raises(AuthorityRefusal, match='invalid_params'):
        apply(3, 'SQL', {'sql': 'DELETE FROM messages'})
    assert apply(3, 'turn.renew', {'holder': 'worker-lease', 'ttl_seconds': 300})['value']
    # The worker releases its turn lease through ``turn.cleanup`` on its own session.
    assert apply(4, 'turn.cleanup', {'target': 'assigned', 'holder': 'worker-lease'}) == {'released': True}
    assistant = {'role': 'assistant', 'content': [{'type': 'text', 'text': 'canonical winner\ud800'}]}
    winner = apply(5, 'transcript.append', {'messages': [assistant]})
    repaired = apply(6, 'transcript.append', {'messages': [{'role': 'assistant',
        'content': 'loser', '_row_id': winner['annotations'][0]['_row_id']}]})
    assert repaired['count'] == 0
    adopted = repaired['annotations'][0]
    assert adopted[CANONICAL_ROW]['content'] == assistant['content']
    assert adopted[DB_ROW_SNAPSHOT] and adopted['_row_id'] == winner['annotations'][0]['_row_id']
    terminal = apply(7, 'execution.finish', {})
    assert terminal['status'] == 'terminal'
    assert apply(7, 'execution.finish', {}) == terminal
    with pytest.raises(AuthorityRefusal, match='stale_generation'):
        apply(8, 'usage.main', {'input_tokens': 1})
