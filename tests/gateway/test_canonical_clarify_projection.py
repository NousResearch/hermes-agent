"""The negotiated batch projection shares the native waiter, never a scalar prompt."""
import threading
from concurrent.futures import Future
from types import SimpleNamespace

from gateway.session_events import SessionEvents
from gateway import session_server_requests as requests


class Peer:
    def __init__(self, canonical=False):
        self.canonical_clarify_projection = canonical
        self.frames = []

    def write(self, frame):
        self.frames.append(frame)
        return True


def test_canonical_projection_and_legacy_request_share_one_waiter():
    events = SessionEvents()
    canonical, legacy = Peer(True), Peer()
    events.fanout.attach(canonical)
    events.fanout.attach(legacy)
    authority = SimpleNamespace(profile_id='owner', epoch=1, sessions={'s': SimpleNamespace(event_stream=events)},
                                executing={'s': 7},
                                _handle=lambda ref: SimpleNamespace(execution_generation=7))
    output = Future()
    def ask():
        try:
            output.set_result(requests.ask_viewer_result(
                events, 's', 'clarify', {'questions': [{'qid': 'q0', 'question': 'First?'}, {'qid': 'q1', 'question': 'Second?'}]},
                qids=['q0', 'q1'], timeout=2, batch_owner=(authority, 7)))
        except BaseException as exc:
            output.set_exception(exc)
    worker = threading.Thread(target=ask, daemon=True)
    worker.start()
    # Synchronize with actual fanout delivery and propagate a worker failure immediately.
    from tests.waiting import wait_until
    def delivered():
        if output.done():
            output.result()
        return canonical.frames and legacy.frames
    wait_until(delivered, timeout=3)
    opening = canonical.frames[0]
    assert opening['method'] == 'event'
    assert opening['params']['type'] == 'clarify.batch.request'
    projection = opening['params']['payload']
    rid = projection['request_id']
    assert legacy.frames[0]['id'] == rid
    assert legacy.frames[0]['method'] == 'clarify'
    assert not any(f.get('method') == 'clarify' for f in canonical.frames)
    assert projection['identity'] == {'profile_id': 'owner', 'authority_epoch': 1,
                                     'stream_id': events._key, 'execution_generation': 7}
    assert requests.canonical_batches(events, 's') == [projection]
    assert requests.open_requests(events, 's', canonical=True) == []
    assert len(requests.open_requests(events, 's')) == 1
    assert requests.resolve({'id': rid, 'result': {'answers': {'q0': '', 'q1': None}, 'outcome': 'submitted'}})
    worker.join(2)
    assert not worker.is_alive()
    assert output.result(timeout=2) == {'answers': {'q0': '', 'q1': None}, 'outcome': 'submitted'}
    assert requests.canonical_batches(events, 's') == []
