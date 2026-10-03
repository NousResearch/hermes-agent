"""Bedrock silence deadlines must not count a wall-clock jump as provider silence."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent import chat_completion_helpers as h


@pytest.fixture
def clock(monkeypatch):
    clock = SimpleNamespace(wall=1000000.0, steady=1000.0)
    monkeypatch.setattr(h, "time", SimpleNamespace(
        time=lambda: clock.wall, monotonic=lambda: clock.steady))
    return clock


def make_call(monkeypatch):
    agent = MagicMock()
    agent._interrupt_requested = False
    agent._consecutive_stale_streams = 0
    monkeypatch.setattr(h, "_derive_stream_stale_timeout", lambda *args: 600.0)
    return h._BedrockStream(agent, {"modelId": "fixture-model"}, None)


def drive_poll(monkeypatch, call, ticks):
    """Control worker scheduling, not the production deadline or error handling."""
    remaining = iter(ticks)
    class Worker:
        alive = True
        def __init__(self, **kwargs):
            pass
        def start(self):
            pass
        def is_alive(self):
            return self.alive
        def join(self, timeout):
            tick = next(remaining, None)
            if tick is None:
                self.alive = False
                call.result["response"] = "completed"
            else:
                tick()
    monkeypatch.setattr(h, "threading", SimpleNamespace(Thread=Worker))
    return call.run()


@pytest.mark.parametrize("wall_jump", [23000, -23000])
def test_wall_jump_does_not_change_remaining_silence_budget(monkeypatch, clock, wall_jump):
    call = make_call(monkeypatch)
    def resume():
        clock.wall += wall_jump
        clock.steady += 456
    assert drive_poll(monkeypatch, call, [resume]) == "completed"
    assert call.agent._consecutive_stale_streams == 0
    call.agent._buffer_diagnostic_status.assert_not_called()


def test_exact_deadline_does_not_expire(monkeypatch, clock):
    call = make_call(monkeypatch)
    def deadline():
        clock.wall += 600
        clock.steady += 600
    assert drive_poll(monkeypatch, call, [deadline]) == "completed"


def test_worker_error_is_not_replaced_by_a_wall_jump(monkeypatch, clock):
    call = make_call(monkeypatch)
    error = ValueError("provider rejected fixture")
    def failed():
        clock.wall += 23000
        clock.steady += 1
        call.result["error"] = error
    with pytest.raises(ValueError) as caught:
        drive_poll(monkeypatch, call, [failed])
    assert caught.value is error


def test_derived_caller_timeout_is_preserved(monkeypatch, clock):
    call = make_call(monkeypatch)
    monkeypatch.setattr(h, "_derive_stream_stale_timeout", lambda *args: 12.0)
    call = h._BedrockStream(call.agent, {"modelId": "fixture-model"}, None)
    def stalled():
        clock.steady += 13
    with pytest.raises(TimeoutError, match="threshold 12s"):
        drive_poll(monkeypatch, call, [stalled])


def test_real_silence_still_times_out_after_backward_wall_jump(monkeypatch, clock):
    from agent import bedrock_adapter
    evicted = []
    monkeypatch.setattr(bedrock_adapter, "invalidate_runtime_client", evicted.append)
    call = make_call(monkeypatch)
    def stalled():
        clock.wall -= 23000
        clock.steady += 601
    with pytest.raises(TimeoutError, match="no events for 601s"):
        drive_poll(monkeypatch, call, [stalled])
    assert call.agent._consecutive_stale_streams == 1
    assert evicted == ["us-east-1"]


def test_worker_events_refresh_the_same_deadline_clock(monkeypatch, clock):
    from agent import relay_llm
    call = make_call(monkeypatch)
    call.agent.reasoning_callback = None
    call.agent.stream_delta_callback = None
    call.agent._has_stream_consumers.return_value = False
    class Stream:
        final_response = None
        def __iter__(self):
            clock.wall += 23000
            clock.steady += 20
            yield {"messageStop": {"stopReason": "end_turn"}}
        def close(self):
            pass
    monkeypatch.setattr(relay_llm, "stream", lambda *a, **kw: Stream())
    call._worker()
    assert call.result["error"] is None
    assert call.result["response"] is not None
    # Advancing beyond the refreshed budget must expire even if wall freezes.
    def stalled():
        clock.steady += 601
    with pytest.raises(TimeoutError, match="no events for 601s"):
        drive_poll(monkeypatch, call, [stalled])
