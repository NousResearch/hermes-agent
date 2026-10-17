"""Against the real owner, a resend after a lost reply returns the stored receipt and applies once."""
import time


def test_lost_reply_is_resent_and_applied_once(worker, monkeypatch):
    db, store = worker
    monkeypatch.setattr(time, 'sleep', lambda s: None)
    owner, lost = store.rpc, [True]

    def lose_first_reply(method, **params):
        result = owner(method, **params)  # the owner commits; only its reply is lost
        if lost:
            lost.pop()
            raise TimeoutError('worker_receipt_timeout')
        return result

    store.rpc = lose_first_reply
    store.append_messages_batch('owned', [{'role': 'user', 'content': 'once'}])

    assert store.failure is None and store.journal['pending'] == []
    assert db._read_one("SELECT COUNT(*) FROM messages WHERE session_id='owned'")[0] == 1
    assert db._read_one("SELECT COUNT(*) FROM worker_receipts WHERE execution_id='worker'")[0] == 1
    # The next write takes the next sequence: the resend did not consume one of its own.
    store.append_messages_batch('owned', [{'role': 'assistant', 'content': 'after'}])
    assert db._read_one("SELECT COUNT(*) FROM messages WHERE session_id='owned'")[0] == 2
