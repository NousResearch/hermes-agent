"""Tests for richer stream-drop diagnostics in agent.log.

When a subagent's stream drops mid-tool-call, the WARNING in agent.log must
carry enough breadcrumbs to answer "WHY did it drop" without requiring a
verbose-mode rerun.  Specifically:

- Inner exception chain (httpx errors wrapped by openai SDK)
- Upstream HTTP headers (cf-ray, x-openrouter-provider, x-openrouter-id, ...)
- HTTP status of the dying response
- Bytes streamed and chunks received before the drop
- Elapsed time on the attempt + time-to-first-byte

Plus the user-visible UI line gains an ``after Xs`` suffix when timing data
is available, distinguishing "couldn't connect at all" from "died mid-stream
after N seconds" (very different root causes).
"""

from __future__ import annotations

import logging


from run_agent import AIAgent


def _make_agent() -> AIAgent:
    return AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )




class _FakeHeaders:
    def __init__(self, d): self._d = {k.lower(): v for k, v in d.items()}
    def get(self, k, default=None): return self._d.get(k.lower(), default)


class _FakeResponse:
    def __init__(self, headers, status=200):
        self.status_code = status
        self.headers = _FakeHeaders(headers)


def test_stream_diag_capture_response_collects_known_headers():
    agent = _make_agent()
    diag = AIAgent._stream_diag_init()
    resp = _FakeResponse({
        "cf-ray": "8f1a2b3c4d5e6f7g-LAX",
        "x-openrouter-provider": "Anthropic",
        "x-openrouter-id": "gen-abc123",
        "x-request-id": "req-xyz",
        "server": "cloudflare",
        "irrelevant-header": "ignored",
    })
    agent._stream_diag_capture_response(diag, resp)
    assert diag["http_status"] == 200
    assert diag["headers"]["cf-ray"] == "8f1a2b3c4d5e6f7g-LAX"
    assert diag["headers"]["x-openrouter-provider"] == "Anthropic"
    assert diag["headers"]["x-openrouter-id"] == "gen-abc123"
    assert diag["headers"]["server"] == "cloudflare"
    # Headers not in _STREAM_DIAG_HEADERS must not be captured (PII surface).
    assert "irrelevant-header" not in diag["headers"]




def test_flatten_exception_chain_walks_cause():
    inner = ConnectionError("upstream closed")
    middle = TimeoutError("timed out")
    middle.__cause__ = inner
    outer = RuntimeError("wrapper")
    outer.__cause__ = middle
    chain = AIAgent._flatten_exception_chain(outer)
    assert "RuntimeError" in chain
    assert "TimeoutError" in chain
    assert "ConnectionError" in chain
    assert " <- " in chain


def test_flatten_exception_chain_caps_depth():
    """Chain renders no more than 4 deep so log lines stay bounded."""
    e0 = ValueError("0")
    prev = e0
    for i in range(1, 8):
        nxt = ValueError(str(i))
        nxt.__cause__ = prev
        prev = nxt
    chain = AIAgent._flatten_exception_chain(prev)
    # 4 layers + 3 separators max.
    assert chain.count("<-") <= 3










def test_quiet_mode_does_not_clobber_runagent_logger_level():
    """Regression guard for the parent fix — must persist across this PR."""
    _ = _make_agent()
    for name in ("run_agent", "tools", "trajectory_compressor", "cron", "hermes_cli"):
        logger = logging.getLogger(name)
        assert logger.getEffectiveLevel() <= logging.WARNING


def test_retry_after_drop_reports_the_attempt_that_dropped():
    """First drop (0-indexed attempt 0) must be announced as attempt 1/N, not 2/N (#90215)."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from agent.chat_completion_helpers import _StreamingCall

    agent = MagicMock()
    agent._is_provider_stream_parse_error.return_value = False
    fake = SimpleNamespace(
        agent=agent,
        clients=SimpleNamespace(diag=None, close_once=lambda reason: None),
        _cancel_current_stream_attempt=lambda reason: None,
        last_chunk_time={"t": 0.0},
    )
    _StreamingCall._retry_after_drop(fake, ConnectionError("drop"), 0, 2, mid_tool_call=False, reason="t")
    assert agent._emit_stream_drop.call_args.kwargs["attempt"] == 1
    assert agent._emit_stream_drop.call_args.kwargs["max_attempts"] == 3
