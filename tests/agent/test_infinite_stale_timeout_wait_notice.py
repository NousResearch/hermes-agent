"""Regression coverage for unbounded local/MoA API-call timeouts."""

from types import SimpleNamespace


def test_moa_wait_notice_formats_infinite_deadline_without_overflow(monkeypatch):
    """The 30-second heartbeat must not cast MoA's ``float('inf')`` to int."""
    from agent import chat_completion_helpers as helpers

    notices: list[str] = []
    agent = SimpleNamespace(
        platform="desktop",
        api_mode="chat_completions",
        provider="moa",
        _consecutive_stale_streams=0,
        _interrupt_requested=False,
        _compute_non_stream_stale_timeout=lambda _kwargs: float("inf"),
        _touch_activity=lambda _message: None,
        _emit_wait_notice=lambda message: notices.append(message),
    )

    class HeartbeatThread:
        """Keep the synthetic worker alive for exactly one 100-poll heartbeat."""

        def __init__(self, *, target, daemon):
            self._polls = 0

        def start(self):
            pass

        def join(self, timeout=None):
            pass

        def is_alive(self):
            self._polls += 1
            return self._polls <= 100

    monkeypatch.setattr(helpers.threading, "Thread", HeartbeatThread)

    response = helpers.interruptible_api_call(agent, {"model": "default"})

    assert response is None
    assert len(notices) == 1
    assert "waiting on default" in notices[0]
    assert "auto-reconnect at never" in notices[0]
