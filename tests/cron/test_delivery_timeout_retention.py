"""Timeout confirmation must honor an outcome moved to retained metadata."""

import pytest


@pytest.mark.parametrize("error", [None, "transport unavailable"])
def test_timeout_keeps_terminal_outcome_after_retention(tmp_path, monkeypatch, error):
    from cron import delivery_queue as queue

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(queue, "MAX_TERMINAL_DELIVERIES", 0)
    original_enqueue = queue.enqueue
    sends = []

    def enqueue_then_deliver(*args, **kwargs):
        pending = original_enqueue(*args, **kwargs)

        def send(*delivery_args):
            sends.append(delivery_args)
            return error

        assert queue.drain(send) == 1
        return pending

    # The gateway completes delivery and retention after the worker enqueues,
    # just as the worker reaches its wait deadline. Real queue APIs and SQL.
    monkeypatch.setattr(queue, "enqueue", enqueue_then_deliver)
    result = queue.enqueue_and_wait("execution", {"id": "job"}, "content", timeout=0)
    status = queue.get_status("execution")
    assert status["status"] == ("failed" if error else "delivered")
    assert result == ("delivery failed" if error else None)
    assert len(sends) == 1
    assert queue.drain(lambda *args: sends.append(args)) == 0
