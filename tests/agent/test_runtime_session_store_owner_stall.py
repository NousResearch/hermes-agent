"""A stalled owner (no answer in time) is retried with the same journal entry, not made sticky."""
import time

import pytest

from agent.runtime_session_store import RuntimeSessionStore, WorkerPersistenceError


def _store(tmp_path, answers):
    sent = []

    def rpc(method, **params):
        sent.append((params['sequence'], params['operation']))
        answer = answers.pop(0) if answers else {'value': True}
        if isinstance(answer, Exception):
            raise answer
        return answer

    return RuntimeSessionStore(rpc, {'session_id': 's'}, tmp_path / 'private'), sent


def test_owner_stall_is_retried_with_the_same_sequence(tmp_path, monkeypatch):
    slept = []
    monkeypatch.setattr(time, 'sleep', slept.append)
    store, sent = _store(tmp_path, [WorkerPersistenceError('owner_unavailable'), TimeoutError('timed out'),
                                    ConnectionResetError()])
    try:
        assert store.refresh_session_turn_lease('s', 'holder') is True
        assert sent == [(1, 'turn.renew')] * 4  # one entry, resent until the owner answers
        assert slept == [1.0, 2.0, 4.0]
        assert store.failure is None and store.journal['pending'] == []
        assert store.flush_token_counts() is True
    finally:
        store.close()


def test_backoff_is_capped_at_eight_seconds(tmp_path, monkeypatch):
    slept = []
    monkeypatch.setattr(time, 'sleep', slept.append)
    store, sent = _store(tmp_path, [TimeoutError()] * 6)
    try:
        assert store.refresh_session_turn_lease('s', 'holder') is True
        assert slept == [1.0, 2.0, 4.0, 8.0, 8.0, 8.0] and len(sent) == 7
    finally:
        store.close()


@pytest.mark.parametrize('answers, patience, tries, error', [
    ([WorkerPersistenceError('stale_epoch')], 60.0, 1, 'stale_epoch'),  # the owner's own refusal
    ([TimeoutError(), WorkerPersistenceError('stale_epoch')], 60.0, 2, 'stale_epoch'),  # stall, then refusal
    ([WorkerPersistenceError('owner_unavailable')] * 2, 0, 1, 'owner_unavailable'),  # window spent
    ([TimeoutError()], 0, 1, 'TimeoutError'),  # a bare timeout still leaves a sticky reason
])
def test_refusal_or_exhausted_patience_stays_sticky(tmp_path, monkeypatch, answers, patience, tries, error):
    monkeypatch.setattr(time, 'sleep', lambda s: None)
    store, sent = _store(tmp_path, answers)
    store.owner_patience_s = patience
    with pytest.raises((WorkerPersistenceError, TimeoutError)):
        store.refresh_session_turn_lease('s', 'holder')
    assert sent == [(1, 'turn.renew')] * tries and store.failure == error
    with pytest.raises(WorkerPersistenceError, match=error):
        store.close()  # close() still reports the sticky failure, as before
