"""Streaming 5xx unmasking: the non-streaming probe surfaces the provider's real error.

Regression shape (AssemblyAI LLM Gateway, 2026-09): a gateway that validates requests
only on its non-streaming path returns an opaque ``500 something went wrong`` when
streaming the SAME request. _handle_stream_error must re-issue once non-streaming on a
pre-delta 5xx so the actionable 4xx (or a successful response) reaches the user instead
of three identical opaque 500s.
"""
from types import SimpleNamespace

import pytest

from agent import chat_completion_helpers as h


def _make_call(api_kwargs, *, deltas_sent=False):
    call = h._StreamingCall.__new__(h._StreamingCall)
    call.agent = SimpleNamespace(
        provider="custom", model="gpt-5.6-sol",
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


def test_stream_5xx_probe_success_delivers_response(monkeypatch):
    call = _make_call({"model": "m", "stream": True, "stream_options": {"x": 1}, "messages": []})
    seen_kwargs = []

    def fake_probe(agent, kwargs):
        seen_kwargs.append(kwargs)
        return "completed-response"

    monkeypatch.setattr(h, "interruptible_api_call", fake_probe)
    adopt_calls = []

    def fake_adopt(r):
        adopt_calls.append(r)
        call.agent._disable_streaming = True  # mirror the real latch
        return r

    monkeypatch.setattr(call, "_adopt_final_response", fake_adopt)

    handled = not _run_handle_stream_error(call, _StreamErr(500))

    assert handled
    assert call.result["response"] == "completed-response"
    assert call.result["error"] is None
    assert seen_kwargs and "stream" not in seen_kwargs[0] and "stream_options" not in seen_kwargs[0]
    assert call.agent._disable_streaming is True  # latched by _adopt_final_response
    assert call.buffered and "non-streaming retry succeeded" in call.buffered[0]


def test_stream_5xx_unmasked_by_probe_4xx(monkeypatch):
    call = _make_call({"model": "m", "messages": []})
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
