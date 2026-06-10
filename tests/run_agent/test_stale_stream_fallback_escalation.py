"""Tests that consecutive stale-stream kills escalate to trigger provider fallback.

When the stale-stream detector kills the connection multiple times within a
single streaming call, the provider is persistently unhealthy.  After a
configurable threshold (default 2), the polling loop sets result["error"] and
_request_cancelled so the conversation loop can invoke _try_activate_fallback().
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


def _make_stream_chunks():
    """Create mock stream chunks for a successful response."""
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


class TestStaleStreamFallbackEscalation:
    """Verify consecutive stale-stream kills escalate to trigger fallback."""

    @pytest.mark.filterwarnings(
        "ignore::pytest.PytestUnhandledThreadExceptionWarning"
    )
    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    @patch("run_agent.AIAgent._replace_primary_openai_client")
    def test_stale_stream_escalates_after_threshold(
        self, mock_replace, mock_close, mock_create
    ):
        """When stale stream kills exceed the threshold, the streaming call
        must raise RuntimeError so the conversation loop can trigger fallback."""
        import httpx
        import time

        call_count = [0]

        def always_stale(*args, **kwargs):
            call_count[0] += 1
            # Block long enough for the stale detector to fire multiple times
            time.sleep(30)
            raise httpx.ReadTimeout("should not reach here")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = always_stale
        mock_create.return_value = mock_client

        agent = _make_agent()
        agent._interrupt_requested = False

        # Set stale timeout very low and threshold to 2
        with patch.dict("os.environ", {
            "HERMES_STREAM_STALE_TIMEOUT": "0.1",
            "HERMES_STALE_STREAM_MAX_KILLS": "2",
        }):
            with pytest.raises(RuntimeError, match="persistently unhealthy"):
                agent._interruptible_streaming_api_call({})

    @pytest.mark.filterwarnings(
        "ignore::pytest.PytestUnhandledThreadExceptionWarning"
    )
    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    @patch("run_agent.AIAgent._replace_primary_openai_client")
    def test_high_threshold_allows_single_stale_recovery(
        self, mock_replace, mock_close, mock_create
    ):
        """With a high threshold, a single stale-stream event followed by
        recovery should succeed without escalation."""
        import httpx

        attempt_count = [0]

        def fail_once_then_succeed(*args, **kwargs):
            attempt_count[0] += 1
            if attempt_count[0] == 1:
                # First attempt: immediate failure (simulates connection drop
                # that the stale detector would catch in real usage)
                raise httpx.RemoteProtocolError("peer closed connection")
            # Second attempt: succeed
            return _make_stream_chunks()

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = fail_once_then_succeed
        mock_create.return_value = mock_client

        agent = _make_agent()
        agent._interrupt_requested = False

        # High threshold — single stale kill should not escalate
        with patch.dict("os.environ", {
            "HERMES_STREAM_STALE_TIMEOUT": "0.1",
            "HERMES_STALE_STREAM_MAX_KILLS": "5",
        }):
            result = agent._interruptible_streaming_api_call({})
            # Should succeed on second attempt
            assert result is not None
            assert attempt_count[0] == 2

    @pytest.mark.filterwarnings(
        "ignore::pytest.PytestUnhandledThreadExceptionWarning"
    )
    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    @patch("run_agent.AIAgent._replace_primary_openai_client")
    def test_escalation_stops_worker_retry(
        self, mock_replace, mock_close, mock_create
    ):
        """When escalation fires, the worker thread must exit without retrying
        (same as interrupt behavior via _request_cancelled)."""
        import httpx
        import time

        attempt_count = [0]

        def always_stale(*args, **kwargs):
            attempt_count[0] += 1
            time.sleep(30)
            raise httpx.ReadTimeout("stale")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = always_stale
        mock_create.return_value = mock_client

        agent = _make_agent()
        agent._interrupt_requested = False

        with patch.dict("os.environ", {
            "HERMES_STREAM_STALE_TIMEOUT": "0.1",
            "HERMES_STALE_STREAM_MAX_KILLS": "2",
        }):
            with pytest.raises(RuntimeError, match="persistently unhealthy"):
                agent._interruptible_streaming_api_call({})

        # The worker should have been killed after 2 stale kills.
        # It should NOT have made many attempts — the escalation should
        # have stopped the retry loop via _request_cancelled.
        assert attempt_count[0] <= 3, (
            f"Expected ≤3 attempts but got {attempt_count[0]}. "
            "The worker kept retrying after escalation."
        )

    @pytest.mark.filterwarnings(
        "ignore::pytest.PytestUnhandledThreadExceptionWarning"
    )
    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    @patch("run_agent.AIAgent._replace_primary_openai_client")
    def test_env_var_configures_threshold(
        self, mock_replace, mock_close, mock_create
    ):
        """HERMES_STALE_STREAM_MAX_KILLS env var configures the threshold."""
        import httpx
        import time

        def always_stale(*args, **kwargs):
            time.sleep(30)
            raise httpx.ReadTimeout("stale")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = always_stale
        mock_create.return_value = mock_client

        agent = _make_agent()
        agent._interrupt_requested = False

        # Set threshold to 5 — should take longer to escalate
        with patch.dict("os.environ", {
            "HERMES_STREAM_STALE_TIMEOUT": "0.1",
            "HERMES_STALE_STREAM_MAX_KILLS": "5",
        }):
            with pytest.raises(RuntimeError, match="persistently unhealthy"):
                agent._interruptible_streaming_api_call({})
