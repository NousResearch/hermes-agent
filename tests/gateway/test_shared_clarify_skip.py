"""An empty ``clarify.respond`` answer is a skip, never a timeout (the method's documented contract)."""
from concurrent.futures import Future
import threading
from types import SimpleNamespace


def _wait_while_viewer_answers(answer):
    from gateway.run_turn_runner_clarify_delivery import _clarify_send_then_wait
    from gateway.session_events import SessionEvents
    from gateway.session_pending_controls import PendingControls
    from tools import clarify_gateway

    controls = PendingControls(SessionEvents())
    entry = clarify_gateway.register('skip-question', 'skip-route', 'Pick one', ['a', 'b'])
    controls.register_clarify('skip-session', 1, entry)
    sent = Future()
    sent.set_result(SimpleNamespace(success=True))
    answered = threading.Thread(target=controls.respond, args=('skip-session', 1, entry.clarify_id, {'answer': answer}),
                                kwargs={'kind': 'clarify'})
    answered.start()
    try:
        return _clarify_send_then_wait(sent, clarify_id=entry.clarify_id, session_key=entry.session_key,
                                       clarify_mod=clarify_gateway), clarify_gateway.SKIPPED
    finally:
        answered.join(5)
        clarify_gateway.clear_session(entry.session_key)


def test_empty_answer_is_a_skip_the_batch_continues_past():
    (response, answered), skipped = _wait_while_viewer_answers('')
    assert (response, answered) == (skipped, True)


def test_real_answer_still_reaches_the_tool():
    (response, answered), _ = _wait_while_viewer_answers('a')
    assert (response, answered) == ('a', True)
