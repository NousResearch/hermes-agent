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
- the recovered delivery opens and closes its own stream pair (consumers must never
  see deltas after the failed attempt's terminal on_stream_end)
- user interrupts re-raise; never swallowed
"""
import io
import logging
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
        _disable_streaming=False, _stream_5xx_probe_ts=None,
        _fire_stream_delta=lambda text: call.deltas.append(text),
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


@pytest.fixture
def plain_probe_log(monkeypatch):
    stream = io.StringIO()
    logger = logging.Logger("probe-log-regression", level=logging.INFO)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))
    logger.addHandler(handler)
    monkeypatch.setattr(h, "logger", logger)
    yield stream
    handler.close()


@pytest.mark.parametrize("status,handled,safe_status", [
    (400, True, 400), (503, False, 503), (None, False, None),
    (0, True, None), (600, False, None), (True, True, None),
    ("SYNTHETIC_STATUS_SECRET_MARKER", False, None),
])
def test_probe_error_logs_exclude_provider_payload(monkeypatch, plain_probe_log, status, handled, safe_status):
    class SyntheticProbeError(Exception):
        pass

    marker = "SYNTHETIC_PROVIDER_SECRET_MARKER"
    error = SyntheticProbeError(marker)
    error.status_code = status
    call = _make_call({"model": "m", "messages": []})

    def fail_probe(agent, kwargs):
        raise error

    monkeypatch.setattr(h, "interruptible_api_call", fail_probe)
    assert call._unmask_server_error_with_nonstreaming(_StreamErr(500)) is handled
    assert call.result["error"] is (error if handled else None)
    assert call.result["response"] is None
    assert call._stream_stale_timeout == 180.0
    output = plain_probe_log.getvalue()
    assert marker not in output and "SYNTHETIC_STATUS_SECRET_MARKER" not in output
    assert "Traceback" not in output
    assert "SyntheticProbeError" in output
    assert f"HTTP {safe_status if safe_status is not None else 'unknown'}" in output


@pytest.mark.parametrize("status", [502, None, "property-raises"])
def test_replay_error_logs_exclude_exception_traceback(monkeypatch, plain_probe_log, status):
    marker = "SYNTHETIC_REPLAY_SECRET_MARKER"

    class SyntheticReplayError(Exception):
        @property
        def status_code(self):
            if status == "property-raises":
                raise ValueError(marker)
            return status

    error = SyntheticReplayError(marker)
    call = _make_call({"model": "m", "messages": []})
    monkeypatch.setattr(h, "interruptible_api_call", lambda agent, kwargs: object())

    def fail_replay(response):
        raise error from _StreamErr(500)

    monkeypatch.setattr(call, "_replay_final_response", fail_replay)
    assert call._unmask_server_error_with_nonstreaming(_StreamErr(500)) is False
    assert call.result["response"] is None and call.result["error"] is None
    assert call._stream_stale_timeout == 180.0
    output = plain_probe_log.getvalue()
    assert marker not in output and "Traceback" not in output
    assert "SyntheticReplayError" in output
    assert f"HTTP {502 if status == 502 else 'unknown'}" in output
