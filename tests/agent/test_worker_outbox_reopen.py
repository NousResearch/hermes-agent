"""A worker reopening an acknowledged-but-unjournaled write reconciles before its next write."""
import pytest

from agent.runtime_session_store import RuntimeSessionStore


def test_reopen_reconciles_pending_receipt_before_new_write(tmp_path):
    calls = []
    def lost_reply(method, **params):
        calls.append(params)
        raise TimeoutError('lost reply')
    scope = {'session_id': 's', 'execution_id': 'w', 'epoch': 1}
    store = RuntimeSessionStore(lost_reply, scope, tmp_path)
    with pytest.raises(TimeoutError):
        store.queue_token_counts('s', input_tokens=10)
    store._outbox_owner.close()  # process exit left its durable pending journal
    restored_calls = []
    def owner(method, **params):
        restored_calls.append(params)
        return {'ok': True}
    reopened = RuntimeSessionStore(owner, scope, tmp_path)
    try:
        reopened.queue_token_counts('s', input_tokens=20)
        assert restored_calls[0] == calls[0]
        assert [row['sequence'] for row in restored_calls] == [1, 2]
        assert [row['payload']['input_tokens'] for row in restored_calls] == [10, 20]
    finally:
        reopened.close()
