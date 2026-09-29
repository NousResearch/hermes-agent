"""Repro: EmptyStreamError currently retries HERMES_STREAM_RETRIES times (storm on main).

On origin/main the guard is `if attempt < max_retries` for BOTH transient and empty
streams, so a 200 OK with 0 bytes retries up to max_retries+1 times. This test
asserts the fixed behavior: empty streams cap at 1 attempt.
"""
import httpx
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent import chat_completion_helpers as cch
from agent.errors import EmptyStreamError


def _bare_call(agent):
    call = object.__new__(cch._StreamingCall)
    call.agent = agent
    call.api_kwargs = {}
    call.result = {}
    call.clients = SimpleNamespace(diag={}, close_once=MagicMock())
    call._request_cancelled = {"value": False}
    call.deltas_were_sent = {"yes": False}
    call.first_delta_fired = {"done": False}
    call.provider_tool_in_flight = {"yes": False}
    call._cancel_current_stream_attempt = MagicMock()
    call.last_chunk_time = {"t": 0.0}
    return call


def _agent():
    return SimpleNamespace(
        _interrupt_requested=False,
        _stream_options_unsupported=False,
        _emit_stream_drop=MagicMock(),
        _is_provider_stream_parse_error=lambda e: False,
        _log_stream_retry=MagicMock(),
        _buffer_diagnostic_status=MagicMock(),
        base_url="https://example.com",
    )


@pytest.mark.real_retry_backoff
def test_empty_stream_caps_retries_at_one():
    """EmptyStreamError must not retry more than once, even with max_retries=10."""
    call = _bare_call(_agent())
    err = EmptyStreamError("provider returned 0 bytes")
    # attempt 0 -> retry (first allowed attempt)
    assert call._handle_stream_error(err, 0, max_retries=10) is True
    # attempt 1 -> exhausted (only 1 attempt allowed for empty streams)
    assert call._handle_stream_error(err, 1, max_retries=10) is False
    assert call.result["error"] is err


@pytest.mark.real_retry_backoff
def test_transient_errors_still_use_full_budget():
    """Transient errors keep the full max_retries budget."""
    call = _bare_call(_agent())
    err = httpx.ConnectError("reset")
    for attempt in range(10):
        assert call._handle_stream_error(err, attempt, max_retries=10) is True
    assert call._handle_stream_error(err, 10, max_retries=10) is False
