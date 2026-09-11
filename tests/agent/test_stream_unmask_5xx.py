"""Streaming 5xx unmasking: the non-streaming probe surfaces the provider's real error.

Regression shape (AssemblyAI LLM Gateway, 2026-09): a gateway that validates requests
only on its non-streaming path returns an opaque ``500 something went wrong`` when
streaming the SAME request. _handle_stream_error must re-issue once non-streaming on a
pre-delta 5xx so the actionable 4xx (or a successful response) reaches the user instead
of three identical opaque 500s.

Gating invariants (from cross-vendor review):
- never after partial delivery, on non-5xx, or without an HTTP status
- one probe per 60s window on the AGENT (outer retries build fresh _StreamingCall
  instances, so an instance flag would re-probe every attempt)
- chat-completions wire only (adoption replays chat-completions shapes)
- probe success delivers for the turn WITHOUT latching _disable_streaming
  (a transient gateway 500 must not permanently disable streaming)
- user interrupts re-raise; never swallowed
"""
import time
from types import SimpleNamespace

import pytest

from agent import chat_completion_helpers as h


def _make_call(api_kwargs, *, deltas_sent=False, api_mode="chat_completions"):
    call = h._StreamingCall.__new__(h._StreamingCall)
    call.agent = SimpleNamespace(
        provider="custom", model="gpt-5.6-sol", api_mode=api_mode,
        _interrupt_requested=False,
        _is_provider_stream_parse_error=lambda e: False,
        _buffer_status=lambda text: call.buffered.append(text),
    )
    call.buffered = []
    call.api_kwargs = api_kwargs
    call.result = {"response": None, "error": None, "partial_tool_names": []}
    call.deltas_were_sent = {"yes": deltas_sent}
    call.provider_tool_in_flight = {"yes": False}
    call._request_cancelled = {"value": False}
    return call


class _StreamErr(Exception):
    """Stands in for the openai 5xx the stream path raises."""

    def __init__(self, status):
        super().__init__(f"Error code: {status} - something went wrong")
        self.status_code = status


class _ProbeErr(Exception):
    def __init__(self, status):
        super().__init__(f"Error code: {status} - real validation message")
        self.status_code = status


def _run_handle_stream_error(call, e):
    return call._handle_stream_error(e, attempt=2, max_retries=2)


def _prime_probe_window(call):
    """Ensure the agent's probe window is open (no probe yet, or an old one)."""
    call.agent._stream_5xx_probe_ts = 0.0


def test_stream_5xx_probe_success_delivers_response_without_latch(monkeypatch):
    call = _make_call({"model": "m", "stream": True, "stream_options": {"x": 1}, "messages": []})
    _prime_probe_window(call)
    seen_kwargs = []

    def fake_probe(agent, kwargs):
        seen_kwargs.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok", reasoning_content=None))])

    monkeypatch.setattr(h, "interruptible_api_call", fake_probe)
    adopt_calls = []

    def fake_adopt(r):
        adopt_calls.append(r)
        call.agent._disable_streaming = True  # mirror the real latch inside adoption
        return r

    monkeypatch.setattr(call, "_adopt_final_response", fake_adopt)

    handled = not _run_handle_stream_error(call, _StreamErr(500))

    assert handled
    assert call.result["response"] is not None
    assert call.result["error"] is None
    assert seen_kwargs and "stream" not in seen_kwargs[0] and "stream_options" not in seen_kwargs[0]
    # One-turn recovery: adoption's internal latch is RESTORED, not kept.
    assert call.agent._disable_streaming is False
    assert call.buffered and "non-streaming retry succeeded" in call.buffered[0]


def test_stream_5xx_unmasked_by_probe_4xx(monkeypatch):
    call = _make_call({"model": "m", "messages": []})
    _prime_probe_window(call)
    real = _ProbeErr(400)

    def fake_probe(agent, kwargs):
        raise real

    monkeypatch.setattr(h, "interruptible_api_call", fake_probe)

    handled = _run_handle_stream_error(call, _StreamErr(500))

    assert not handled  # loop stops, error propagates
    assert call.result["error"] is real  # the REAL validation error replaces the opaque 500
    assert call.result["response"] is None


def test_stream_5xx_probe_5xx_keeps_original_error(monkeypatch):
    call = _make_call({"model": "m", "messages": []})
    _prime_probe_window(call)
    original = _StreamErr(500)

    def fake_probe(agent, kwargs):
        raise _ProbeErr(503)

    monkeypatch.setattr(h, "interruptible_api_call", fake_probe)

    handled = _run_handle_stream_error(call, original)

    assert not handled
    assert call.result["error"] is original  # probe 5xx is no more informative; keep the first error


@pytest.mark.parametrize("status", [400, 429])
def test_no_probe_for_client_errors(monkeypatch, status):
    call = _make_call({"model": "m", "messages": []})

    def fake_probe(agent, kwargs):
        raise AssertionError("probe must not run for non-5xx stream errors")

    monkeypatch.setattr(h, "interruptible_api_call", fake_probe)

    handled = _run_handle_stream_error(call, _StreamErr(status))

    assert not handled
    assert call.result["response"] is None


def test_no_probe_after_partial_delivery(monkeypatch):
    call = _make_call({"model": "m", "messages": []}, deltas_sent=True)

    def fake_probe(agent, kwargs):
        raise AssertionError("probe must not run after deltas were delivered")

    monkeypatch.setattr(h, "interruptible_api_call", fake_probe)

    handled = _run_handle_stream_error(call, _StreamErr(500))

    assert not handled


def test_no_probe_without_status_code(monkeypatch):
    call = _make_call({"model": "m", "messages": []})

    def fake_probe(agent, kwargs):
        raise AssertionError("probe must not run without a 5xx status")

    monkeypatch.setattr(h, "interruptible_api_call", fake_probe)

    handled = _run_handle_stream_error(call, Exception("connection exploded"))

    assert not handled


def test_probe_window_blocks_second_probe_within_60s(monkeypatch):
    call = _make_call({"model": "m", "messages": []})
    call.agent._stream_5xx_probe_ts = time.time()  # probe just happened (fresh instance = outer retry)

    def fake_probe(agent, kwargs):
        raise AssertionError("second probe within the window must not run")

    monkeypatch.setattr(h, "interruptible_api_call", fake_probe)

    handled = _run_handle_stream_error(call, _StreamErr(500))

    assert not handled
    assert call.result["error"] is not None  # original error propagates


def test_probe_window_rearms_after_60s(monkeypatch):
    call = _make_call({"model": "m", "messages": []})
    call.agent._stream_5xx_probe_ts = time.time() - 61.0

    def fake_probe(agent, kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok", reasoning_content=None))])

    monkeypatch.setattr(h, "interruptible_api_call", fake_probe)
    monkeypatch.setattr(call, "_adopt_final_response", lambda r: r)

    handled = not _run_handle_stream_error(call, _StreamErr(500))

    assert handled


def test_probe_skipped_for_non_chat_completions_wire(monkeypatch):
    call = _make_call({"model": "m", "messages": []}, api_mode="anthropic_messages")

    def fake_probe(agent, kwargs):
        raise AssertionError("probe must not run on the anthropic wire (adoption replays chat shapes)")

    monkeypatch.setattr(h, "interruptible_api_call", fake_probe)

    handled = _run_handle_stream_error(call, _StreamErr(500))

    assert not handled


def test_interrupt_during_probe_reraises(monkeypatch):
    call = _make_call({"model": "m", "messages": []})
    _prime_probe_window(call)

    def fake_probe(agent, kwargs):
        raise InterruptedError("Agent interrupted during API call")

    monkeypatch.setattr(h, "interruptible_api_call", fake_probe)

    with pytest.raises(InterruptedError):
        _run_handle_stream_error(call, _StreamErr(500))
