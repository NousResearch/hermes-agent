"""Queued prior ACK must not satisfy a later read-only request on the same socket."""
import queue
import threading
from types import SimpleNamespace

import pytest

from agent.runtime_session_store import WorkerPersistenceError
from tests.gateway.test_worker_receipt_reconciliation import owner, committed
from tests.waiting import wait_until
import hermes_state_runtime as rt


def test_stale_queued_responses_share_one_receive_deadline(monkeypatch):
    from agent.runtime_session_store import _OwnerLink
    import agent.runtime_session_store as runtime
    now, timeouts = [0.0], []
    class Replies:
        def get(self, timeout):
            timeouts.append(timeout)
            now[0] += .4
            return {'id': 1, 'result': 'old'}
    link = object.__new__(_OwnerLink)
    link.awaiting, link.replies = 2, Replies()
    monkeypatch.setattr(runtime.time, 'monotonic', lambda: now[0])
    with pytest.raises(TimeoutError):
        link.receive(1)
    assert timeouts == pytest.approx([1, .6, .2])


@pytest.mark.parametrize('current_error', [False, True])
def test_receive_discards_prior_success_and_error_but_preserves_current(current_error):
    from agent.runtime_session_store import _OwnerLink
    link = object.__new__(_OwnerLink)
    link.awaiting, link.replies = 2, queue.SimpleQueue()
    link.replies.put({'id': 1, 'result': 'old'})
    link.replies.put({'id': 1, 'error': {'message': 'old failure'}})
    current = {'id': 2, 'error': {'message': 'current rejection'}} if current_error else {'id': 2, 'result': 'current'}
    link.replies.put(current)
    assert link.receive(1) is current


def test_receive_preserves_transport_error_identity_after_stale_response():
    from agent.runtime_session_store import _OwnerLink
    link = object.__new__(_OwnerLink)
    link.awaiting, link.replies = 2, queue.SimpleQueue()
    link.replies.put({'id': 1, 'result': 'old'})
    error = OSError('synthetic disconnect')
    link.replies.put(error)
    with pytest.raises(OSError) as caught:
        link.receive(1)
    assert caught.value is error


@pytest.mark.parametrize('change,reason', [
    ('none', None), ('epoch', 'stale_epoch'), ('generation', 'stale_generation'),
    ('digest', 'admission_conflict'),
])
def test_native_queued_ack_cannot_replace_current_receipt_reply(owner, monkeypatch, change, reason):
    # Scale only this fixture's observation window; actual socket send and SQLite commit remain real.
    owner.rpc.budget = SimpleNamespace(rpc_timeout_seconds=3, recovery_timeout_seconds=2,
                                      close_timeout_seconds=.05)
    release = threading.Event()
    injected = []
    real_connection = owner.rpc._connection

    # Hold the physical BoundedIO call AFTER native WebSocket send returns.
    # Holding sendall itself holds the supplier protocol lock and prevents reply
    # enqueue, so it cannot schedule this queued-ACK race.
    def connection():
        link = real_connection()
        if not injected:
            injected.append(True)
            native_send = link.send
            first = [True]
            def held_send(request_id, message):
                native_send(request_id, message)
                if first.pop() if first else False:
                    assert release.wait(10), 'owned send observer not released'
            monkeypatch.setattr(link, 'send', held_send)
        return link

    monkeypatch.setattr(owner.rpc, '_connection', connection)
    params = dict(owner.scope, sequence=1, operation='session.activity',
                  payload={'ts':1234567890.0, 'description':None, 'provenance':None})
    flight = None
    try:
        with pytest.raises(TimeoutError):
            owner.rpc('worker.persist', **params)
        flight = owner.rpc._inflight_io
        assert flight is not None and not flight.done.is_set()
        link = owner.rpc.link
        assert link is not None
        wait_until(lambda: not link.replies.empty(), timeout=2)
        assert len(committed(owner)[1]) == 1
        if change == 'epoch':
            rt.begin_runtime_epoch(owner.db, instance_id='new-owner')
        elif change == 'generation':
            owner.db._execute_write(lambda c: c.execute('UPDATE sessions SET runtime_generation=1'))
        elif change == 'digest':
            params['payload']['ts'] += 1
        before = committed(owner)
        release.set()
        assert flight.done.wait(2)
        if reason:
            with pytest.raises(WorkerPersistenceError, match=reason):
                owner.rpc.reconcile_receipt(**params)
        else:
            assert owner.rpc.reconcile_receipt(**params) == {'value': None}
        assert owner.rpc.link is link, 'this control must exercise same-link reuse'
        assert len(owner.requests) == 2
        assert owner.requests[1]['receipt_only'] is True
        assert owner.requests[1]['sequence'] == owner.requests[0]['sequence'] == 1
        assert committed(owner) == before
        assert len(committed(owner)[1]) == 1
    finally:
        release.set()
        if flight is not None:
            assert flight.done.wait(5)
            flight.thread.join(2)
            assert not flight.thread.is_alive()
