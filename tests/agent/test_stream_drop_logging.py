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
import time
from unittest.mock import patch

import pytest


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






def test_private_context_stream_retry_omits_remote_messages_and_headers(caplog):
    from agent.redact import bind_volatile_sensitive_text

    agent = _make_agent()
    agent.provider = "openrouter"
    snapshot = "Latitude: 37.7749\nLongitude: -122.4194"
    inner = ConnectionError("provider fragment 37.77")
    outer = RuntimeError("wrapper fragment -122.41")
    outer.__cause__ = inner
    diag = AIAgent._stream_diag_init()
    diag["headers"] = {"x-request-id": "request-near-37.7749"}

    with bind_volatile_sensitive_text(snapshot), caplog.at_level(logging.WARNING):
        agent._log_stream_retry(
            kind="drop",
            error=outer,
            attempt=2,
            max_attempts=3,
            mid_tool_call=False,
            diag=diag,
        )

    msg = next(r.getMessage() for r in caplog.records if "Stream drop" in r.getMessage())
    assert "Provider error details withheld for private-context turn" in msg
    assert "RuntimeError <- ConnectionError" in msg
    assert "upstream=[withheld]" in msg
    assert "37.77" not in msg
    assert "-122.41" not in msg


def test_private_context_stream_end_hook_gets_only_safe_error_summary():
    from agent.chat_completion_helpers import _with_stream_emitters
    from agent.redact import bind_volatile_sensitive_text

    agent = _make_agent()
    payloads = []
    agent._emit_stream_start = lambda: None
    agent._emit_stream_end = lambda **payload: payloads.append(payload)
    snapshot = "Latitude: 37.7749\nLongitude: -122.4194"

    with bind_volatile_sensitive_text(snapshot), pytest.raises(RuntimeError):
        _with_stream_emitters(
            agent,
            lambda: (_ for _ in ()).throw(
                RuntimeError("provider echoed 37.77 / -122.41")
            ),
        )

    assert payloads == [
        {
            "final_text": "",
            "finished": False,
            "error": "Provider error details withheld for private-context turn",
        }
    ]


def test_emit_stream_drop_ui_includes_elapsed_when_available():
    agent = _make_agent()
    agent.provider = "openrouter"

    diag = AIAgent._stream_diag_init()
    diag["started_at"] = time.time() - 8.0  # 8s on the wire before drop

    with patch.object(agent, "_buffer_status") as mock_emit:
        agent._emit_stream_drop(
            error=ConnectionError("x"),
            attempt=2,
            max_attempts=3,
            mid_tool_call=True,
            diag=diag,
        )

    msg = mock_emit.call_args.args[0]
    # Suffix with elapsed time helps distinguish "couldn't connect" (0s)
    # from "died mid-stream after a while".
    assert "after" in msg and "s" in msg




def test_quiet_mode_does_not_clobber_runagent_logger_level():
    """Regression guard for the parent fix — must persist across this PR."""
    _ = _make_agent()
    for name in ("run_agent", "tools", "trajectory_compressor", "cron", "hermes_cli"):
        logger = logging.getLogger(name)
        assert logger.getEffectiveLevel() <= logging.WARNING
