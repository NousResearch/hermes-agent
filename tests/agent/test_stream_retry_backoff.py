"""Stream-level reconnects back off exponentially and stay interruptible (#60029)."""
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent import chat_completion_helpers as cch


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


def _agent(**kw):
    return SimpleNamespace(
        _interrupt_requested=False,
        _stream_options_unsupported=False,
        _emit_stream_drop=MagicMock(),
        _is_provider_stream_parse_error=lambda e: False,
        **kw,
    )


@pytest.mark.real_retry_backoff
def test_transient_drops_retry_with_capped_exponential_backoff():
    """Each reconnect waits 1s, 2s, 4s, 4s — including Anthropic SDK connection errors."""
    anthropic = pytest.importorskip("anthropic")

    waits = []
    call = _bare_call(_agent())
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    errors = [anthropic.APIConnectionError(request=req), httpx.ConnectError("reset"),
              httpx.ReadError("reset"), httpx.RemoteProtocolError("closed")]
    with patch.object(cch, "_wait_stream_retry_backoff", lambda agent, delay: waits.append(delay) or True):
        for attempt, err in enumerate(errors):
            assert call._handle_stream_error(err, attempt, max_retries=10) is True
    assert waits == [1.0, 2.0, 4.0, 4.0]
    assert call.last_chunk_time["t"] > 0.0  # stale clock restarted after the backoff
    assert "error" not in call.result


@pytest.mark.real_retry_backoff
def test_backoff_wait_returns_immediately_on_interrupt():
    agent = _agent()
    call = _bare_call(agent)
    threading.Timer(0.2, lambda: setattr(agent, "_interrupt_requested", True)).start()
    start = time.monotonic()
    call._retry_after_drop(httpx.ConnectError("x"), 5, 10, mid_tool_call=False, reason="t")
    assert time.monotonic() - start < 1.0
