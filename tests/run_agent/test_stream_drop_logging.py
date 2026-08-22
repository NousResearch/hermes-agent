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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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


def _stream_chunk(content=None, tool_calls=None, finish_reason=None):
    delta = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning_content=None,
        reasoning=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)],
        model="test/model",
        usage=None,
    )


def _tool_call_delta(index=0, tc_id=None, name=None, arguments=None):
    return SimpleNamespace(
        index=index,
        id=tc_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def test_stream_diag_init_returns_well_formed_dict():
    diag = AIAgent._stream_diag_init()
    assert "started_at" in diag
    assert diag["chunks"] == 0
    assert diag["bytes"] == 0
    assert diag["first_chunk_at"] is None
    assert diag["http_status"] is None
    assert diag["headers"] == {}


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


def test_log_stream_retry_includes_diagnostic_fields(caplog):
    agent = _make_agent()
    agent._delegate_depth = 1
    agent._subagent_id = "sa-3-deadbeef"
    agent.provider = "openrouter"

    diag = AIAgent._stream_diag_init()
    diag["http_status"] = 200
    diag["headers"] = {
        "cf-ray": "8f1a2b3c4d5e6f7g-LAX",
        "x-openrouter-provider": "Anthropic",
        "x-openrouter-id": "gen-xyz789",
    }
    diag["chunks"] = 12
    diag["bytes"] = 4096
    # Simulate 5s elapsed with first chunk at 0.5s.
    diag["started_at"] = time.time() - 5.0
    diag["first_chunk_at"] = diag["started_at"] + 0.5

    inner = ConnectionError("peer closed")
    outer = RuntimeError("Connection error.")
    outer.__cause__ = inner

    with caplog.at_level(logging.WARNING, logger="run_agent"):
        agent._log_stream_retry(
            kind="drop mid tool-call",
            error=outer,
            attempt=2,
            max_attempts=3,
            mid_tool_call=True,
            diag=diag,
        )

    msg = next(
        r.getMessage() for r in caplog.records
        if "Stream drop mid tool-call" in r.getMessage()
    )

    # Identity
    assert "subagent_id=sa-3-deadbeef" in msg
    assert "provider=openrouter" in msg

    # Inner-cause chain
    assert "RuntimeError" in msg and "ConnectionError" in msg

    # Counters and timing
    assert "http_status=200" in msg
    assert "bytes=4096" in msg
    assert "chunks=12" in msg
    # elapsed should be roughly 5s; allow some slack.
    assert "elapsed=" in msg
    assert "ttfb=0.50s" in msg

    # Upstream headers
    assert "cf-ray=8f1a2b3c4d5e6f7g-LAX" in msg
    assert "x-openrouter-provider=Anthropic" in msg
    assert "x-openrouter-id=gen-xyz789" in msg


def test_log_stream_retry_works_without_diag(caplog):
    """diag is optional — older callers / unit tests still work."""
    agent = _make_agent()
    agent._delegate_depth = 0
    agent.provider = "openrouter"

    with caplog.at_level(logging.WARNING, logger="run_agent"):
        agent._log_stream_retry(
            kind="drop",
            error=ConnectionError("x"),
            attempt=2,
            max_attempts=3,
            mid_tool_call=False,
        )

    msg = next(r.getMessage() for r in caplog.records if "Stream drop" in r.getMessage())
    # Without diag, the structured fields show "-" placeholders.
    assert "http_status=-" in msg
    assert "upstream=[-]" in msg
    assert "bytes=0" in msg
    assert "chunks=0" in msg
    assert "ttfb=-" in msg


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


class TestStreamDropAttemptNumbers:
    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_plain_transient_drop_numbers_first_attempt_and_retries(
        self, mock_close, mock_create, monkeypatch
    ):
        import httpx

        monkeypatch.setenv("HERMES_STREAM_RETRIES", "2")
        attempts = {"count": 0}

        def create_stream(*args, **kwargs):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise httpx.ConnectError("connection dropped")
            return iter([_stream_chunk(content="ok", finish_reason="stop")])

        client = MagicMock()
        client.chat.completions.create.side_effect = create_stream
        mock_create.return_value = client
        agent = _make_agent()
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        with patch.object(agent, "_emit_stream_drop", wraps=agent._emit_stream_drop) as emit:
            agent._interruptible_streaming_api_call({})

        assert attempts["count"] == 2
        assert emit.call_args.kwargs["attempt"] == 1
        assert emit.call_args.kwargs["max_attempts"] == 3
        assert emit.call_args.kwargs["mid_tool_call"] is False

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_mid_tool_call_drop_numbers_each_retry(
        self, mock_close, mock_create, monkeypatch
    ):
        import httpx

        monkeypatch.setenv("HERMES_STREAM_RETRIES", "2")
        attempts = {"count": 0}

        def create_stream(*args, **kwargs):
            attempts["count"] += 1
            if attempts["count"] < 3:
                def failing_stream():
                    yield _stream_chunk(content="Working")
                    yield _stream_chunk(tool_calls=[_tool_call_delta(
                        tc_id="call_1", name="write_file", arguments='{"path": '
                    )])
                    raise httpx.RemoteProtocolError("peer closed")
                return failing_stream()
            return iter([
                _stream_chunk(tool_calls=[_tool_call_delta(
                    tc_id="call_1", name="write_file", arguments='{"path": "/tmp/x"}'
                )]),
                _stream_chunk(finish_reason="tool_calls"),
            ])

        client = MagicMock()
        client.chat.completions.create.side_effect = create_stream
        mock_create.return_value = client
        agent = _make_agent()
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False
        agent._fire_stream_delta = lambda text: None

        with patch.object(agent, "_emit_stream_drop", wraps=agent._emit_stream_drop) as emit:
            agent._interruptible_streaming_api_call({})

        assert attempts["count"] == 3
        assert [call.kwargs["attempt"] for call in emit.call_args_list] == [1, 2]
        assert all(call.kwargs["max_attempts"] == 3 for call in emit.call_args_list)
        assert all(call.kwargs["mid_tool_call"] is True for call in emit.call_args_list)

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_always_failing_stream_has_retry_and_exhausted_attempt_numbers(
        self, mock_close, mock_create, monkeypatch
    ):
        import httpx

        monkeypatch.setenv("HERMES_STREAM_RETRIES", "2")

        def always_fails(*args, **kwargs):
            raise httpx.ConnectError("connection dropped")

        client = MagicMock()
        client.chat.completions.create.side_effect = always_fails
        mock_create.return_value = client
        agent = _make_agent()
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        with (
            patch.object(agent, "_emit_stream_drop", wraps=agent._emit_stream_drop) as emit,
            patch.object(agent, "_log_stream_retry", wraps=agent._log_stream_retry) as log,
        ):
            with pytest.raises(httpx.ConnectError):
                agent._interruptible_streaming_api_call({})

        assert [call.kwargs["attempt"] for call in emit.call_args_list] == [1, 2]
        assert log.call_args.kwargs["attempt"] == 3
        assert log.call_args.kwargs["max_attempts"] == 3
