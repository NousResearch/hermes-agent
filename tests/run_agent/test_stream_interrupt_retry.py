"""Tests that /stop interrupts streaming retry loops immediately.

When the agent is interrupted during a streaming API call, the outer poll
loop closes the HTTP connection.  The inner `_call()` thread sees a
connection error and enters its retry loop.  Before this fix, the retry
loop would open a FRESH connection without checking `_interrupt_requested`,
making /stop take multiple retry cycles × read-timeout to actually stop
(510+ seconds observed on slow ollama-cloud providers).

The fix adds an `_interrupt_requested` check at the top of the retry loop
so the agent exits immediately instead of retrying.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _make_agent(**kwargs):
    """Create a minimal AIAgent for streaming tests."""
    from run_agent import AIAgent

    defaults = dict(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    defaults.update(kwargs)
    agent = AIAgent(**defaults)
    agent.api_mode = "chat_completions"
    return agent


class TestStreamInterruptBeforeRetry:
    """Verify _interrupt_requested is checked before each streaming retry."""

    @pytest.mark.filterwarnings(
        "ignore::pytest.PytestUnhandledThreadExceptionWarning"
    )
    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_interrupt_prevents_stream_retry(self, mock_close, mock_create):
        """When _interrupt_requested is set during a transient stream error,
        the retry loop must NOT retry — it should raise InterruptedError
        immediately instead of opening a fresh connection."""
        import httpx

        attempt_count = [0]

        def fail_once_then_interrupt(*args, **kwargs):
            attempt_count[0] += 1
            if attempt_count[0] == 1:
                # First attempt: simulate normal failure, then set interrupt
                # (as if /stop arrived while the retry loop processes the error)
                agent._interrupt_requested = True
                raise httpx.ConnectError("connection reset by /stop")
            # Should never reach here — the interrupt check should fire first
            raise httpx.ConnectError("unexpected retry — interrupt not checked!")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = fail_once_then_interrupt
        mock_create.return_value = mock_client

        agent = _make_agent()
        agent._interrupt_requested = False

        with pytest.raises(InterruptedError, match="interrupted"):
            agent._interruptible_streaming_api_call({})

        # Only 1 attempt should have been made — the interrupt should prevent retry
        assert attempt_count[0] == 1, (
            f"Expected 1 attempt but got {attempt_count[0]}. "
            "The retry loop retried despite _interrupt_requested being set."
        )

    @pytest.mark.filterwarnings(
        "ignore::pytest.PytestUnhandledThreadExceptionWarning"
    )
    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_interrupt_before_first_attempt(self, mock_close, mock_create):
        """If _interrupt_requested is already set when the streaming call
        starts, it should exit immediately without making any API call."""
        mock_client = MagicMock()
        mock_create.return_value = mock_client

        agent = _make_agent()
        agent._interrupt_requested = True  # Pre-set before call

        with pytest.raises(InterruptedError, match="interrupted"):
            agent._interruptible_streaming_api_call({})

        # No API call should have been made at all
        assert mock_client.chat.completions.create.call_count == 0

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_normal_retry_still_works_without_interrupt(self, mock_close, mock_create):
        """Without an interrupt, transient errors should still retry normally."""
        import httpx

        attempts = [0]

        def fail_twice_then_succeed(*args, **kwargs):
            attempts[0] += 1
            if attempts[0] <= 2:
                raise httpx.ConnectError("transient failure")
            # Third attempt succeeds
            chunks = [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            index=0,
                            delta=SimpleNamespace(
                                content="ok",
                                tool_calls=None,
                                reasoning_content=None,
                                reasoning=None,
                            ),
                            finish_reason=None,
                        )
                    ],
                    model="test/model",
                    usage=None,
                ),
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            index=0,
                            delta=SimpleNamespace(
                                content=None,
                                tool_calls=None,
                                reasoning_content=None,
                                reasoning=None,
                            ),
                            finish_reason="stop",
                        )
                    ],
                    model="test/model",
                    usage=None,
                ),
            ]
            stream = MagicMock()
            stream.__iter__ = MagicMock(return_value=iter(chunks))
            stream.response = MagicMock()
            stream.response.headers = {}
            return stream

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = fail_twice_then_succeed
        mock_create.return_value = mock_client

        agent = _make_agent()
        agent._interrupt_requested = False

        # Should succeed on the third attempt
        result = agent._interruptible_streaming_api_call({})
        assert result is not None
        assert attempts[0] == 3

    @pytest.mark.filterwarnings(
        "ignore::pytest.PytestUnhandledThreadExceptionWarning"
    )
    @patch("run_agent.AIAgent._replace_primary_openai_client")
    @patch("run_agent.AIAgent._abort_request_openai_client")
    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_stale_stream_attempt_cannot_emit_late_chunks_after_retry(
        self,
        mock_close,
        mock_create,
        mock_abort,
        mock_replace,
        monkeypatch,
    ):
        """A stale attempt must not keep writing deltas after it is killed.

        This reproduces the race where the outer stale detector aborts an SSE
        connection, but the old iterator still yields one more chunk before
        surfacing the connection error that triggers the retry.
        """
        import httpx
        import time

        from tests.run_agent.test_streaming import (
            _make_stream_chunk,
            _make_tool_call_delta,
        )

        monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "0.05")
        monkeypatch.setenv("HERMES_STREAM_RETRIES", "1")

        class LateChunkAfterStaleStream:
            response = SimpleNamespace(headers={})

            def __iter__(self):
                yield _make_stream_chunk(content="old start ")
                yield _make_stream_chunk(
                    tool_calls=[
                        _make_tool_call_delta(
                            index=0,
                            tc_id="call_1",
                            name="terminal",
                        )
                    ]
                )
                time.sleep(0.45)
                yield _make_stream_chunk(content="old late ")
                raise httpx.RemoteProtocolError("peer closed connection")

        retry_chunks = [
            _make_stream_chunk(content="new final"),
            _make_stream_chunk(finish_reason="stop", model="test/model"),
        ]
        class RetryStream:
            response = SimpleNamespace(headers={})

            def __iter__(self):
                return iter(retry_chunks)

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            LateChunkAfterStaleStream(),
            RetryStream(),
        ]
        mock_create.return_value = mock_client

        agent = _make_agent()
        agent._interrupt_requested = False
        deltas = []
        agent.stream_delta_callback = deltas.append

        response = agent._interruptible_streaming_api_call({})

        delivered = "".join(deltas)
        assert "old late" not in delivered
        assert "new final" in delivered
        assert response.choices[0].message.content == "new final"
        assert mock_abort.called


class TestStreamInterruptJoinsWorkerBeforeRaise:
    """#81521: interrupt must join the stream worker before raising.

    Raising InterruptedError immediately lets Relay turn teardown race a
    still-open physical LLM scope ("scope handle is not at the top of the
    stack") and cascade into the CLI EIO / redraw storm.
    """

    @pytest.mark.filterwarnings(
        "ignore::pytest.PytestUnhandledThreadExceptionWarning"
    )
    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_interrupt_joins_worker_before_raising(
        self, mock_close, mock_create, monkeypatch
    ):
        import threading

        import httpx

        join_timeouts: list[float | None] = []
        original_join = threading.Thread.join

        def spy_join(self, *args, **kwargs):
            timeout = kwargs.get("timeout", args[0] if args else None)
            join_timeouts.append(timeout)
            return original_join(self, *args, **kwargs)

        monkeypatch.setattr(threading.Thread, "join", spy_join)

        # The join is gated on Relay managed execution being live (it is
        # pointless — and delays interrupt detection — when no Relay
        # consumers are registered). Simulate a live runtime.
        from agent import relay_runtime as rr

        monkeypatch.setattr(
            rr,
            "get_runtime",
            lambda *a, **k: SimpleNamespace(
                managed_execution_enabled=lambda: True,
                get_session=lambda *a2, **k2: None,
                ensure_session=lambda *a2, **k2: None,
            ),
        )

        class HangUntilClosedStream:
            response = SimpleNamespace(headers={})

            def __iter__(self):
                # Block until the poll loop force-closes / cancels; then
                # surface a transport error like a real aborted SSE body.
                import time

                deadline = time.time() + 5.0
                while time.time() < deadline:
                    time.sleep(0.05)
                raise httpx.RemoteProtocolError("connection closed by interrupt")

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = HangUntilClosedStream()
        mock_create.return_value = mock_client

        agent = _make_agent()
        agent._interrupt_requested = False

        def fire_interrupt():
            import time

            time.sleep(0.2)
            agent._interrupt_requested = True

        threading.Thread(target=fire_interrupt, daemon=True).start()

        with pytest.raises(InterruptedError, match="interrupted"):
            agent._interruptible_streaming_api_call({})

        assert 2.0 in join_timeouts, (
            f"Expected a 2.0s worker join before InterruptedError; "
            f"saw join timeouts {join_timeouts!r}. Without the join, Relay "
            f"scope teardown races the stream worker (#81521)."
        )
