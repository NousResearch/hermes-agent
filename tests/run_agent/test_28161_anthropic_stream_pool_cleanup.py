"""Anthropic stream cleanup must call _anthropic_client.close() + _rebuild_anthropic_client(),
not _replace_primary_openai_client(), to avoid 15-minute hangs on Anthropic-native configs.

Three cleanup sites in chat_completion_helpers.interruptible_streaming_api_call() were
calling _replace_primary_openai_client() unconditionally.  For api_mode=anthropic_messages
this silently fails (no OPENAI_API_KEY) and leaves the in-flight httpx stream unclosed,
blocking the worker thread until the 900s httpx read-timeout fires.

Tests cover:
- stream_retry_pool_cleanup  (connection error on fresh stream, L1836)
- stale_stream_pool_cleanup  (outer poll loop detects stale stream, L1987)

Fixes #28161
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
    """_replace_primary_openai_client must not be called for api_mode=anthropic_messages."""

    @pytest.mark.filterwarnings(
        "ignore::pytest.PytestUnhandledThreadExceptionWarning"
    )
    def test_stream_retry_calls_anthropic_rebuild_not_openai(self):
        """Connection error during stream retry → close+rebuild Anthropic client, not OpenAI."""
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
        mock_rebuild.assert_called_once()
        agent._anthropic_client.close.assert_called_once()

    @pytest.mark.filterwarnings(
        "ignore::pytest.PytestUnhandledThreadExceptionWarning"
    )
    def test_stale_stream_calls_anthropic_rebuild_not_openai(self, monkeypatch):
        """Stale-stream outer-poll detector → close+rebuild Anthropic client, not OpenAI."""
        monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "0.1")

        agent = _make_anthropic_agent()
        unblock = threading.Event()
        attempt_count = [0]

        def _stream_side_effect(*args, **kwargs):
            attempt_count[0] += 1
            if attempt_count[0] == 1:
                # First attempt: stream that yields nothing (triggers stale detector),
                # then raises ConnectError once _anthropic_client.close() unblocks it.
                cm = MagicMock()
                stream = MagicMock()

                def _blocking_gen():
                    unblock.wait(timeout=5.0)
                    raise httpx.ConnectError("connection dropped after close()")
                    yield  # make this a generator so next() triggers the wait

                stream.__iter__ = MagicMock(return_value=_blocking_gen())
                cm.__enter__ = MagicMock(return_value=stream)
                cm.__exit__ = MagicMock(return_value=False)
                return cm
            # Second attempt: succeed
            return _good_stream_cm()

        agent._anthropic_client.messages.stream.side_effect = _stream_side_effect
        # close() on the mock Anthropic client unblocks the inner thread.
        agent._anthropic_client.close.side_effect = unblock.set

        with patch.object(agent, "_rebuild_anthropic_client") as mock_rebuild:
            with patch.object(
                agent, "_replace_primary_openai_client"
            ) as mock_replace:
                agent._interruptible_streaming_api_call({})

        mock_replace.assert_not_called()
        # close() and rebuild called at least once by the stale detector.
        agent._anthropic_client.close.assert_called()
        assert mock_rebuild.call_count >= 1
