"""Regression: mid-stream httpx.ReadError ("[Errno 32] Broken pipe") must be
classified TRANSIENT so the stream worker retries instead of failing the turn
with "Streaming failed before delivery".

Field evidence (errors.log 2026-09-06): the token-plan endpoint dropped the
SSE connection mid-response; httpcore.ReadError -> httpx.ReadError escaped the
transient predicate in _StreamingCall._handle_stream_error because ReadError is
neither a Timeout nor a ConnectError/RemoteProtocolError/ConnectionError.
"""

from __future__ import annotations

import types

import httpx
import pytest

from agent.chat_completion_helpers import _StreamingCall


def _make_streaming_call() -> _StreamingCall:
    """Minimal _StreamingCall with a stub agent (no real API client needed)."""
    agent = types.SimpleNamespace(
        provider="custom",
        model="test-model",
        _is_provider_stream_parse_error=lambda exc: False,
        _log_stream_retry=lambda **kw: None,
        _buffer_status=lambda msg: None,
    )
    call = _StreamingCall.__new__(_StreamingCall)
    call.agent = agent
    call.result = {"response": None, "error": None, "partial_tool_names": []}
    call._request_cancelled = {"value": False}
    call.first_delta_fired = {"done": False}
    call.deltas_were_sent = {"yes": False}
    call.provider_tool_in_flight = {"yes": False}
    call.last_chunk_time = {"t": 0.0}
    call._stream_stale_timeout = None
    call.managed_stream_holder = {"stream": None}
    call.clients = types.SimpleNamespace(diag={})
    return call


@pytest.fixture(autouse=True)
def _stub_retry_cleanup(monkeypatch):
    """Avoid touching the real retry/backoff machinery in the classification test."""
    monkeypatch.setattr(
        _StreamingCall, "_retry_after_drop", lambda self, *a, **kw: None, raising=False
    )


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ReadError("[Errno 32] Broken pipe"),
        httpx.WriteError("connection aborted"),
        httpx.ConnectError("dial failed"),
        httpx.RemoteProtocolError("server disconnected"),
        httpx.ReadTimeout("read timed out"),
        ConnectionResetError("peer reset"),
        BrokenPipeError(32, "Broken pipe"),
    ],
)
def test_transient_network_errors_are_retried(exc):
    call = _make_streaming_call()
    should_retry = call._handle_stream_error(exc, attempt=0, max_retries=2)
    assert should_retry is True, (
        f"{type(exc).__name__} was NOT classified transient: no stream retry "
        f"(error={call.result['error']!r})"
    )
    assert call.result["error"] is None


def test_read_error_exhausted_sets_result_error():
    """After the last attempt, ReadError must still surface as result error."""
    call = _make_streaming_call()
    should_retry = call._handle_stream_error(
        httpx.ReadError("[Errno 32] Broken pipe"), attempt=2, max_retries=2
    )
    assert should_retry is False
    assert isinstance(call.result["error"], httpx.ReadError)
