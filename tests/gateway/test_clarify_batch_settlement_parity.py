"""One settlement rule for a batch clarify, whichever registry holds it.

Main's sidecar (``tui_gateway/server_requests.py``) and the session owner (``gateway/session_server_requests.py``)
each keep open server→client requests, and each used to compute a batch's terminal ``{answers, outcome}``
in its own code, which drifted twice (#140472 review). Both now call ``tools.clarify_batch.settle_batch``.
These cases drive each real registry through the same ending, one ``clarify.lock`` on ``q0`` first, and
require the same result.
"""
import threading
import time

import pytest

from tools.clarify_batch import settle_batch

QUESTIONS = [{'qid': 'q0', 'question': 'colour?', 'choices': ['red', 'blue'], 'multi_select': False},
             {'qid': 'q1', 'question': 'size?', 'choices': None, 'multi_select': False}]
QIDS = ['q0', 'q1']

# ending -> (how the wait ends after q0 is locked, expected result)
CASES = {
    'cancel_all_frame': ({'result': {}}, {'answers': {'q0': 'blue'}, 'outcome': 'cancelled'}),
    'closing_frame_with_tail': ({'result': {'answers': {'q1': 'large'}}},
                                {'answers': {'q0': 'blue', 'q1': 'large'}, 'outcome': 'submitted'}),
    'frame_names_its_outcome': ({'result': {'answers': {'q1': 'large'}, 'outcome': 'timed_out'}},
                                {'answers': {'q0': 'blue', 'q1': 'large'}, 'outcome': 'timed_out'}),
    'frame_overrides_a_lock': ({'result': {'answers': {'q0': 'red', 'q1': 'large'}}},
                               {'answers': {'q0': 'red', 'q1': 'large'}, 'outcome': 'submitted'}),
    'last_lock': ('lock', {'answers': {'q0': 'blue', 'q1': 'large'}, 'outcome': 'submitted'}),
    'timeout': ('timeout', {'answers': {'q0': 'blue'}, 'outcome': 'timed_out'}),
}


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.01)


def _sidecar(monkeypatch, ending):
    from tui_gateway import server_requests
    frames = []
    monkeypatch.setattr(server_requests, '_write', frames.append)
    monkeypatch.setattr(server_requests, '_emit', lambda *a: None)
    monkeypatch.setattr(server_requests, '_answerable', lambda sid: True)
    server_requests.reset_for_tests()
    out = {}
    wire = {'questions': QUESTIONS}
    timeout = 0.5 if ending == 'timeout' else 10
    worker = threading.Thread(target=lambda: out.update(
        result=server_requests.send('clarify', 's1', wire, timeout=timeout, qids=QIDS)))
    worker.start()
    try:
        _wait_for(lambda: frames)
        rid = frames[0]['id']
        assert server_requests.lock_answer(rid, 'q0', 'blue') == ['q1']
        if ending == 'lock':
            assert server_requests.lock_answer(rid, 'q1', 'large') == []
        elif ending != 'timeout':
            assert server_requests.resolve_response({'jsonrpc': '2.0', 'id': rid, **ending})
        worker.join(5)
        return out['result']
    finally:
        server_requests.reset_for_tests()


def _owner(ending):
    from gateway.session_events import SessionEvents
    from gateway.session_server_requests import ask_viewer_result, lock_answer, resolve

    class Peer:
        frames = []

        def write(self, frame):
            self.frames.append(frame)
            return True

        def close(self):
            pass
    events, peer = SessionEvents(), Peer()
    peer.frames = []
    events.fanout.attach(peer)
    out = {}
    timeout = 0.5 if ending == 'timeout' else 10
    worker = threading.Thread(target=lambda: out.update(result=ask_viewer_result(
        events, 's1', 'clarify', {'questions': QUESTIONS}, timeout=timeout, qids=QIDS)))
    worker.start()
    _wait_for(lambda: any(f.get('method') == 'clarify' for f in peer.frames))
    rid = next(f['id'] for f in peer.frames if f.get('method') == 'clarify')
    allow = lambda *a: True  # entitlement is covered in test_clarify_lock_owner
    assert lock_answer(rid, 'q0', 'blue', session_ok=allow) == ['q1']
    if ending == 'lock':
        assert lock_answer(rid, 'q1', 'large', session_ok=allow) == []
    elif ending != 'timeout':
        assert resolve({'jsonrpc': '2.0', 'id': rid, **ending})
    worker.join(5)
    return out['result']


@pytest.mark.parametrize('case', sorted(CASES))
def test_both_registries_settle_a_locked_batch_the_same_way(case, monkeypatch):
    ending, expected = CASES[case]
    assert _sidecar(monkeypatch, ending) == expected, 'sidecar'
    assert _owner(ending) == expected, 'owner'


def test_the_rule_itself():
    locked = {'q0': 'blue'}
    assert settle_batch(locked, ended='cancelled') == {'answers': {'q0': 'blue'}, 'outcome': 'cancelled'}
    assert settle_batch(locked, {}) == {'answers': {'q0': 'blue'}, 'outcome': 'cancelled'}
    assert settle_batch(locked, None) == {'answers': {'q0': 'blue'}, 'outcome': 'cancelled'}  # error frame
    assert settle_batch(locked, {'answers': {}, 'outcome': 'bogus'}) == {'answers': locked, 'outcome': 'submitted'}
    # The caller's dict is never aliased into the result.
    assert settle_batch(locked, ended='timed_out')['answers'] is not locked
