"""Partial delivery must not erase a watchdog-counted stale attempt."""

import threading
from types import SimpleNamespace

import httpx
import pytest

from agent import chat_completion_helpers as helpers


def _partial_agent():
    def noop(*args, **kwargs):
        pass

    agent = SimpleNamespace(
        provider="synthetic",
        model="synthetic",
        base_url="https://synthetic.invalid/v1",
        api_mode="chat_completions",
        platform="cli",
        _interrupt_requested=False,
        _consecutive_stale_streams=0,
        _current_streamed_assistant_text="",
        _stream_options_unsupported=True,
        _buffer_diagnostic_status=noop,
        _emit_diagnostic_wait=noop,
        _touch_activity=noop,
        _is_provider_stream_parse_error=lambda error: False,
    )

    def record_text(text):
        agent._current_streamed_assistant_text += text

    agent._fire_stream_delta = record_text
    return agent


def test_partial_stale_streams_trip_breaker_before_another_request(monkeypatch):
    monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "0.1")
    monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "2")
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
    opened = []

    def stalled_wire(call, attempt_id):
        opened.append(attempt_id)
        released = threading.Event()
        call.clients.set_client(SimpleNamespace(close=released.set), kind="stream")
        call._emit_text("Visible partial.")
        assert released.wait(5), "Native watchdog did not abort the stream"
        raise httpx.ReadError("read ended after watchdog close")

    monkeypatch.setattr(helpers._StreamingCall, "_call_wire", stalled_wire)
    agent = _partial_agent()
    for expected_streak in (1, 2):
        agent._current_streamed_assistant_text = ""
        response = helpers.interruptible_streaming_api_call(
            agent, {"model": agent.model, "messages": []}
        )
        assert response.choices[0].message.content == "Visible partial."
        assert response.choices[0].message.tool_calls is None
        assert response.choices[0].finish_reason == "length"
        assert agent._consecutive_stale_streams == expected_streak

    with pytest.raises(RuntimeError, match="2 consecutive stale attempts"):
        helpers.interruptible_streaming_api_call(
            agent, {"model": agent.model, "messages": []}
        )
    assert len(opened) == 2


def test_partial_failure_without_stale_kill_preserves_prior_streak(monkeypatch):
    monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "2")
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")

    def dropped_wire(call, attempt_id):
        call._emit_text("Visible partial.")
        raise httpx.ReadError("ordinary connection reset")

    monkeypatch.setattr(helpers._StreamingCall, "_call_wire", dropped_wire)
    agent = _partial_agent()
    agent._consecutive_stale_streams = 1
    response = helpers.interruptible_streaming_api_call(
        agent, {"model": agent.model, "messages": []}
    )
    assert response.choices[0].message.content == "Visible partial."
    assert agent._consecutive_stale_streams == 1


def test_inline_partial_stale_count_precedes_abort(monkeypatch):
    monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "0.1")
    monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "50")
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
    agent = _partial_agent()
    agent.platform = "subagent"
    agent._consecutive_stale_streams = 2
    read_released = threading.Event()
    abort_finished = threading.Event()
    release_monitor = threading.Event()
    kill_finished = threading.Event()
    agent._abort_request_openai_client = lambda *a, **k: read_released.set()
    agent._close_request_openai_client = lambda *a, **k: None
    close = helpers._RequestClientRegistry.close_once
    kill = helpers._StreamingCall._kill_stale_stream

    def pause_after_abort(registry, reason):
        close(registry, reason)
        if reason == "stale_stream_kill":
            abort_finished.set()
            assert release_monitor.wait(8)

    def observe_kill(call, elapsed):
        try:
            kill(call, elapsed)
        finally:
            kill_finished.set()

    def stalled_wire(call, attempt_id):
        call.clients.set_client(object(), kind="openai")
        call._emit_text("Visible partial.")
        assert read_released.wait(6)
        raise httpx.ReadError("watchdog-unblocked read")

    monkeypatch.setattr(helpers._RequestClientRegistry, "close_once", pause_after_abort)
    monkeypatch.setattr(helpers._StreamingCall, "_kill_stale_stream", observe_kill)
    monkeypatch.setattr(helpers._StreamingCall, "_call_wire", stalled_wire)
    try:
        response = helpers.interruptible_streaming_api_call(
            agent, {"model": agent.model, "messages": []}
        )
        assert abort_finished.is_set()
        assert response.choices[0].finish_reason == "length"
        assert response.choices[0].message.content == "Visible partial."
        assert agent._consecutive_stale_streams == 3
        monkeypatch.setattr(
            helpers._StreamingCall, "_call_wire",
            lambda *a: SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop")]),
        )
        helpers.interruptible_streaming_api_call(
            agent, {"model": agent.model, "messages": []}
        )
        assert agent._consecutive_stale_streams == 0
        release_monitor.set()
        assert kill_finished.wait(4)
        assert agent._consecutive_stale_streams == 0
    finally:
        release_monitor.set()
