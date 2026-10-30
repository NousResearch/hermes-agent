"""A refused turn still ends with message.complete, so the viewer never hangs."""
from types import SimpleNamespace

from gateway.session_events import SessionEvents
from gateway.session_settlement_recovery import _publish_completion
from hermes_cli.active_sessions import MAX_CONCURRENT_SESSIONS, ActiveSessionRefusal


def test_refused_turn_publishes_completion_with_its_reason():
    stream = SessionEvents()
    received = []
    stream.observers.add(received.append)
    live = SimpleNamespace(controls=SimpleNamespace(snapshot=lambda *a: None), event_stream=stream)
    refusal = ActiveSessionRefusal('Too many active sessions.', MAX_CONCURRENT_SESSIONS)

    _publish_completion(SimpleNamespace(runner=None), live, SimpleNamespace(session_id='s'),
                        {'admission_id': 'a1', 'request_id': 'r1'}, {'outcome': 'failed'}, refusal, None)

    [frame] = received
    payload = frame['params']['payload']
    assert frame['params']['type'] == 'message.complete'
    assert payload['text'] == 'Too many active sessions.'
    assert payload['text'].reason == MAX_CONCURRENT_SESSIONS
