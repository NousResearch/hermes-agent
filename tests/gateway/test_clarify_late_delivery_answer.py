"""Keep a committed native answer when a send reports failure late.
The real delivery watcher and waiter run without a wall-clock delay.
"""
from concurrent.futures import Future, TimeoutError

from gateway.platforms.base import SendResult
from gateway.run_turn_runner_clarify_delivery import _clarify_send_then_wait
from tools import clarify_gateway as cm


def test_committed_answer_wins_late_send_failure(monkeypatch):
    class DelayedSend(Future):
        def result(self, timeout=None):
            if not self.done():
                raise TimeoutError()
            return super().result(timeout=timeout)

    future = DelayedSend()
    entry = cm.register('late-send-answer', 'late-send-route', 'Choose', ['a', 'b'])

    def wait(timeout):
        assert cm.resolve_gateway_clarify(entry.clarify_id, 'b')
        future.set_result(SendResult(success=False))
        return True

    monkeypatch.setattr(entry.event, 'wait', wait)
    try:
        assert _clarify_send_then_wait(future, clarify_id=entry.clarify_id,
            session_key=entry.session_key, clarify_mod=cm) == ('b', True)
    finally:
        cm.clear_session(entry.session_key)
