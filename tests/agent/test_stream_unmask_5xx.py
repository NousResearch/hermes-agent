"""Streaming 5xx unmasking: the non-streaming probe surfaces the provider's real error.

Regression shape (AssemblyAI LLM Gateway, 2026-09): a gateway that validates requests
only on its non-streaming path returns an opaque ``500 something went wrong`` when
streaming the SAME request. _handle_stream_error must re-issue once non-streaming on a
pre-delta 5xx so the actionable 4xx (or a successful response) reaches the user instead
of three identical opaque 500s.

Gating invariants (from cross-vendor review):
- never after partial delivery, on non-5xx, or without an HTTP status — except the
  status-less in-band empty-upstream error, which carries no status to key on
- one probe per 60s window on the AGENT (outer retries build fresh _StreamingCall
  instances, so an instance flag would re-probe every attempt)
- chat-completions wire only (adoption replays chat-completions shapes)
- probe success delivers for the turn WITHOUT latching _disable_streaming
  (a transient gateway 500 must not permanently disable streaming)
- the recovered delivery opens and closes its own stream pair (consumers must never
  see deltas after the failed attempt's terminal on_stream_end)
- user interrupts re-raise; never swallowed
"""
import time
from types import SimpleNamespace

import pytest
from openai import APIError

from agent import chat_completion_helpers as h


def _make_call(api_kwargs, *, deltas_sent=False, api_mode="chat_completions", emit_warning=None,
               delivered_text=""):
    call = h._StreamingCall.__new__(h._StreamingCall)
    call.agent = SimpleNamespace(
        provider="custom", model="gpt-5.6-sol", api_mode=api_mode,
        _interrupt_requested=False,
        _is_provider_stream_parse_error=lambda e: False,
        _buffer_status=lambda text: call.buffered.append(text),
        _disable_streaming=False, _stream_5xx_probe_ts=None,
        _emit_warning=emit_warning or (lambda text: None),
        _fire_stream_delta=lambda text: call.deltas.append(text),
        _current_streamed_assistant_text=delivered_text,
        _reset_stream_delivery_tracking=lambda: None,
    )
    call.buffered = []
    call.deltas = []
    call.first_delta_fired = {"done": True}
    call.on_first_delta = None
    call._stream_stale_timeout = 180.0
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


def test_stream_5xx_probe_success_delivers_response_without_latch(monkeypatch):
    call = _make_call({"model": "m", "stream": True, "stream_options": {"x": 1}, "messages": []})
    seen_kwargs = []

    def fake_probe(agent, kwargs):
        seen_kwargs.append((kwargs, call._stream_stale_timeout))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok", reasoning_content=None))])

    monkeypatch.setattr(h, "interruptible_api_call", fake_probe)

    handled = not _run_handle_stream_error(call, _StreamErr(500))

    assert handled
    assert call.result["response"] is not None
    assert call.result["error"] is None
    probe_kwargs, stale_during_probe = seen_kwargs[0]
    assert "stream" not in probe_kwargs and "stream_options" not in probe_kwargs
    assert call.deltas == ["ok"]
    # One-turn recovery: streaming is never latched off for the session.
    assert call.agent._disable_streaming is False
    # No chunks arrive during the probe: the stream monitor must not stale-kill it.
    assert stale_during_probe == float("inf") and call._stream_stale_timeout == 180.0
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


# ── In-band SSE error on an HTTP 200 (OpenRouter empty response) ────────────
#
# OpenRouter reports a provider-side empty generation as HTTP 200 plus an in-band
# SSE ``error`` frame. The OpenAI SDK reads ``data["error"]["message"]`` and raises a
# bare ``APIError`` with NO status (openai/_streaming.py), so the probe's
# ``status is None`` guard rejected the one error class the probe actually recovers.
# These tests drive the real SDK SSE decoder over a synthetic frame — no network.


def _inband_sse_api_error(message):
    """The exception the OpenAI SDK raises for a 200 response whose first SSE frame
    is ``data: {"error": {"message": ...}}`` — built through the real decoder."""
    import json

    import httpx
    from openai import OpenAI, Stream
    from openai.types.chat import ChatCompletionChunk

    body = f"data: {json.dumps({'error': {'message': message}})}\n\n".encode()
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(200, request=request,
                              headers={"content-type": "text/event-stream"}, content=body)
    stream = Stream(cast_to=ChatCompletionChunk, response=response,
                    client=OpenAI(api_key="test-key", max_retries=0))
    with pytest.raises(APIError) as exc_info:
        list(stream)
    err = exc_info.value
    # The premise of this regression: the exception carries no status at all.
    assert getattr(err, "status_code", None) is None
    return err


def test_inband_empty_response_is_reissued_nonstreaming(monkeypatch):
    """A status-less in-band empty-response error must reach the non-streaming probe."""
    call = _make_call({"model": "m", "stream": True, "messages": []})
    e = _inband_sse_api_error("Provider returned an empty response")
    probe_calls = []

    def fake_probe(agent, kwargs):
        probe_calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="recovered", reasoning_content=None))])

    monkeypatch.setattr(h, "interruptible_api_call", fake_probe)

    handled = not _run_handle_stream_error(call, e)

    assert handled  # the turn was recovered instead of surfacing the empty response
    assert len(probe_calls) == 1  # exactly one non-streaming re-issue
    assert call.deltas == ["recovered"]
    assert call.result["response"] is not None
    assert call.result["error"] is None
    # Streaming is not latched off: an empty upstream response is transient, not a
    # provider that rejects streams.
    assert call.agent._disable_streaming is False


def test_statusless_non_empty_error_is_not_reissued(monkeypatch):
    """A status-less error OUTSIDE the empty-upstream vocabulary must NOT be reissued:
    the probe must not widen into retry-on-any-APIError."""
    call = _make_call({"model": "m", "stream": True, "messages": []})
    e = _inband_sse_api_error("Request failed for a reason unrelated to emptiness")
    probes = []
    monkeypatch.setattr(h, "interruptible_api_call",
                        lambda agent, kwargs: probes.append(kwargs))

    handled = _run_handle_stream_error(call, e)

    assert handled is False  # error propagates, unreissued
    assert probes == []
    assert call.result["error"] is e
    assert call.result["response"] is None


def test_inband_empty_response_respects_probe_window(monkeypatch):
    """The 60s window still bounds the probe: a second in-band empty response inside
    it is not re-issued again."""
    call = _make_call({"model": "m", "stream": True, "messages": []})
    call.agent._stream_5xx_probe_ts = time.monotonic()
    probes = []
    monkeypatch.setattr(h, "interruptible_api_call",
                        lambda agent, kwargs: probes.append(kwargs))

    handled = _run_handle_stream_error(call, _inband_sse_api_error("Provider returned an empty response"))

    assert handled is False
    assert probes == []
    assert call.result["error"] is not None


def test_inband_empty_response_not_reissued_after_partial_delivery(monkeypatch):
    """Visible text already reached the consumer: no re-issue whatever the error shape
    (a re-issue would duplicate delivered text)."""
    call = _make_call({"model": "m", "stream": True, "messages": []},
                      deltas_sent=True, delivered_text="half an answer")
    probes = []
    monkeypatch.setattr(h, "interruptible_api_call",
                        lambda agent, kwargs: probes.append(kwargs))

    _run_handle_stream_error(call, _inband_sse_api_error("Provider returned an empty response"))

    assert probes == []
    assert call.result["error"] is not None
