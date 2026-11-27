"""Actual native waiter races, overflow reentry, and terminal identity controls."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading

import pytest

from gateway.session_events import SessionEvents
from gateway import session_server_requests as requests
from tests.gateway.fixtures.canonical_batch_owner import Peer, QUESTIONS


def owner_for(events):
    return SimpleNamespace(profile_id='p', epoch=1, sessions={'s': SimpleNamespace(event_stream=events)},
                           executing={'s': 1}, _handle=lambda ref: SimpleNamespace(execution_generation=1))


def test_overflow_callback_can_snapshot_and_withdraw():
    events = SessionEvents()
    peer = Peer()
    peer.canonical_clarify_projection = True
    events.fanout.attach(peer)
    events.fanout._MAX_PENDING_FRAMES = 0
    observed = threading.Event()
    def overflow(transport):
        assert len(requests.canonical_batches(events, 's')) == 1
        assert requests.withdraw(events, 's', 'clarify') == 1
        observed.set()
    events.on_overflow = overflow
    owner = owner_for(events)
    # A daemon worker ensures a broken lock boundary fails rather than hanging pytest exit.
    out, failures = [], []
    def ask():
        try:
            out.append(requests.ask_viewer_result(events, 's', 'clarify', {'questions': QUESTIONS},
                                                  qids=['q0', 'q1'], batch_owner=(owner, 1), timeout=1))
        except BaseException as exc:
            failures.append(exc)
    worker = threading.Thread(target=ask, daemon=True)
    worker.start()
    assert observed.wait(3)
    worker.join(3)
    assert not worker.is_alive()
    assert failures == []
    assert out == [{'answers': {}, 'outcome': 'cancelled'}]


@pytest.mark.parametrize('iteration', range(8))
def test_timeout_full_result_last_lock_and_withdraw_have_one_immutable_winner(iteration):
    events = SessionEvents()
    peer = Peer()
    peer.canonical_clarify_projection = True
    events.fanout.attach(peer)
    opened, timed_wait = threading.Event(), threading.Event()
    projections = []
    def observe(frame):
        if frame['params']['type'] == 'clarify.batch.request':
            projections.append(frame['params']['payload'])
            opened.set()
    events.observers.add(observe)
    owner = owner_for(events)
    out, failures = [], []
    def polled():
        timed_wait.set()
        return False
    def ask():
        try:
            out.append(requests.ask_viewer_result(events, 's', 'clarify', {'questions': QUESTIONS},
                                                  qids=['q0', 'q1'], batch_owner=(owner, 1),
                                                  timeout=.25, cancelled=polled))
        except BaseException as exc:
            failures.append(exc)
    worker = threading.Thread(target=ask, daemon=True)
    worker.start()
    assert opened.wait(3)
    projection = projections[0]
    rid, identity = projection['request_id'], projection['identity']
    assert requests.lock_answer(rid, 'q0', None, session_ok=lambda *args: True, identity=identity) == ['q1']
    ready = [threading.Event() for _ in range(3)]
    def settle(index):
        ready[index].set()
        if index == 0:
            return requests.resolve({'id': rid, 'result': {'answers': {'q1': ''}, 'outcome': 'submitted'}}, identity=identity)
        if index == 1:
            return requests.lock_answer(rid, 'q1', 'locked', session_ok=lambda *args: True, identity=identity)
        return requests.withdraw(events, 's', 'clarify')
    with ThreadPoolExecutor(max_workers=3) as pool:
        with events.lock:
            contenders = [pool.submit(settle, i) for i in range(3)]
            assert all(flag.wait(3) for flag in ready)
            assert timed_wait.wait(3), 'actual waiter never reached its timeout boundary'
        results = [future.result(timeout=3) for future in contenders]
    worker.join(3)
    assert not worker.is_alive()
    assert not failures
    assert len(out) == 1
    assert out[0]['answers']['q0'] is None
    assert out[0]['outcome'] in ('submitted', 'cancelled', 'timed_out')
    assert sum((results[0] is True, results[1] == [], results[2] == 1)) <= 1
    frozen = dict(out[0]['answers'])
    assert not requests.resolve({'id': rid, 'result': {'answers': {'q0': 'late'}}}, identity=identity)
    assert requests.lock_answer(rid, 'q0', 'late', session_ok=lambda *args: True, identity=identity) is None
    assert out[0]['answers'] == frozen
    assert requests.canonical_batches(events, 's') == []
    assert requests.open_requests(events, 's') == []
    replay = events.since(events.epoch, 0)
    assert not any(f['type'] == 'clarify.batch.request' for f in replay['events'])
    withdrawals = [f['payload'] for f in replay['events'] if f['type'] == 'request.cancel']
    assert withdrawals == [{'id': rid, 'method': 'clarify', 'reason': 'timeout' if out[0]['outcome'] == 'timed_out' else 'resolved',
                             'identity': identity}]
    assert all('answers' not in f for f in withdrawals)
