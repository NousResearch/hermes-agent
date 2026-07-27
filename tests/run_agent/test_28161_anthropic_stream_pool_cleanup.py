"""Anthropic stream cleanup must not call _replace_primary_openai_client() and
must not hang on Anthropic-native configs (#28161), now via the request-local
client model (#67142).

Originally three cleanup sites in interruptible_streaming_api_call() called
_replace_primary_openai_client() unconditionally; for api_mode=anthropic_messages
that silently failed (no OPENAI_API_KEY) and left the in-flight httpx stream
unclosed, blocking the worker until the 900s read-timeout fired.

Since #67142, anthropic streams run on a per-request client: the stale/retry
cleanup closes the *request-local* client (worker-owned) and builds a fresh one
next attempt — the shared _anthropic_client is never closed/rebuilt from inside
a request (that poll-thread close was the TLS-FD→SQLite corruption vector). The
no-hang guarantee is preserved because the poll thread aborts the request
client's sockets, which unblocks the worker.

Tests cover:
- stream_retry cleanup  (connection error on fresh stream)
- stale_stream cleanup  (outer poll loop detects stale stream)

Fixes #28161. Extends #67142.
"""
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_anthropic_agent(**kwargs):
    from run_agent import AIAgent

    defaults = dict(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="claude-opus-4-7",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    defaults.update(kwargs)
    agent = AIAgent(**defaults)
    agent.api_mode = "anthropic_messages"
    agent._anthropic_client = MagicMock()
    agent._anthropic_api_key = "test-anthropic-key"
    # #67142: anthropic streams now run on a request-local client; route it to
    # the test mock so .messages.stream is exercised and its cleanup observed.
    agent._create_request_anthropic_client = lambda *a, **k: agent._anthropic_client
    return agent


def _good_stream_cm():
    """Context manager whose stream yields no events and returns a valid message."""
    cm = MagicMock()
    stream = MagicMock()
    stream.__iter__ = MagicMock(return_value=iter([]))
    msg = MagicMock()
    msg.content = []
    msg.stop_reason = "end_turn"
    msg.usage = SimpleNamespace(input_tokens=10, output_tokens=5)
    stream.get_final_message = MagicMock(return_value=msg)
    cm.__enter__ = MagicMock(return_value=stream)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


def _failing_stream_cm():
    """Context manager whose __enter__ raises ConnectError immediately."""
    cm = MagicMock()
    cm.__enter__ = MagicMock(
        side_effect=httpx.ConnectError("connection reset by peer")
    )
    return cm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAnthropicStreamPoolCleanup:
    """Anthropic cleanup must never touch the OpenAI primary or the shared
    Anthropic client, and must not hang (#28161 / #67142)."""

    @pytest.mark.filterwarnings(
        "ignore::pytest.PytestUnhandledThreadExceptionWarning"
    )
    def test_stream_retry_closes_request_client_not_openai(self):
        """Connection error during stream retry → close the request-local
        Anthropic client (worker-owned) and retry; never rebuild the shared
        Anthropic client, never touch the OpenAI primary."""
        agent = _make_anthropic_agent()

        attempt_count = [0]

        def _stream_side_effect(*args, **kwargs):
            attempt_count[0] += 1
            if attempt_count[0] == 1:
                return _failing_stream_cm()
            return _good_stream_cm()

        agent._anthropic_client.messages.stream.side_effect = _stream_side_effect

        with patch.object(agent, "_rebuild_anthropic_client") as mock_rebuild:
            with patch.object(
                agent, "_replace_primary_openai_client"
            ) as mock_replace:
                agent._interruptible_streaming_api_call({})

        mock_replace.assert_not_called()
        # #67142: the shared client is never rebuilt from inside a request; the
        # request-local client (routed to this mock) is closed instead.
        mock_rebuild.assert_not_called()
        agent._anthropic_client.close.assert_called()
        assert attempt_count[0] == 2  # retried once, then succeeded

    @pytest.mark.filterwarnings(
        "ignore::pytest.PytestUnhandledThreadExceptionWarning"
    )
    def test_stale_stream_aborts_request_client_not_openai(self, monkeypatch):
        """Stale-stream outer-poll detector → abort the request-local client's
        socket (unblocking the worker) and retry; never _replace_primary_openai
        and never rebuild the shared Anthropic client."""
        monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "0.1")

        agent = _make_anthropic_agent()
        unblock = threading.Event()
        attempt_count = [0]

        def _stream_side_effect(*args, **kwargs):
            attempt_count[0] += 1
            if attempt_count[0] == 1:
                # First attempt: stream that yields nothing (triggers stale
                # detector), then raises ConnectError once the poll thread
                # aborts the request client's socket.
                cm = MagicMock()
                stream = MagicMock()

                def _blocking_gen():
                    unblock.wait(timeout=5.0)
                    raise httpx.ConnectError("connection dropped after abort")
                    yield  # make this a generator so next() triggers the wait

                stream.__iter__ = MagicMock(return_value=_blocking_gen())
                cm.__enter__ = MagicMock(return_value=stream)
                cm.__exit__ = MagicMock(return_value=False)
                return cm
            # Second attempt: succeed
            return _good_stream_cm()

        agent._anthropic_client.messages.stream.side_effect = _stream_side_effect
        # #67142: the stale detector aborts the request-local client's sockets
        # from the poll thread (not close() on the shared client); simulate the
        # socket shutdown waking the blocked read.
        agent._abort_request_anthropic_client = lambda *a, **k: unblock.set()

        with patch.object(agent, "_rebuild_anthropic_client") as mock_rebuild:
            with patch.object(
                agent, "_replace_primary_openai_client"
            ) as mock_replace:
                agent._interruptible_streaming_api_call({})

        mock_replace.assert_not_called()
        # The shared Anthropic client is never rebuilt from inside a request.
        mock_rebuild.assert_not_called()
        assert attempt_count[0] >= 2  # stale-killed once, then retried
