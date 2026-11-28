"""Tests for None chunk handling in streaming chat completions.

Some upstream gateways emit bare `data: null` SSE keep-alive events
mid-stream. The OpenAI SDK parses these as None and yields them to
the caller. The streaming loop must skip them gracefully.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _make_stream_chunk(content=None, finish_reason=None, model=None, usage=None):
    """Build a mock streaming chunk matching OpenAI's ChatCompletionChunk shape."""
    delta = SimpleNamespace(content=content, tool_calls=None, reasoning_content=None, reasoning=None)
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model=model, usage=usage)


def _make_empty_chunk(model=None, usage=None):
    """Build a chunk with no choices (usage-only final chunk)."""
    return SimpleNamespace(choices=[], model=model, usage=usage)


class TestNoneChunkGuard:
    """Verify that None chunks from SSE keep-alive events are skipped."""

    @patch("run_agent.AIAgent._create_openai_client", return_value=MagicMock())
    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_none_chunks_skipped(self, mock_close, mock_create, mock_init):
        """Bare data: null SSE keep-alive events are silently skipped."""
        from run_agent import AIAgent

        chunks = [
            _make_stream_chunk(content="Hello"),
            None,
            _make_stream_chunk(content=" world"),
            None,
            _make_stream_chunk(content="!", finish_reason="stop", model="test-model"),
            _make_empty_chunk(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3)),
        ]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)
        mock_create.return_value = mock_client

        agent = AIAgent(
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        response = agent._interruptible_streaming_api_call({})

        assert response.choices[0].message.content == "Hello world!"
        assert response.choices[0].finish_reason == "stop"
        assert response.usage is not None
        assert response.usage.completion_tokens == 3

    @patch("run_agent.AIAgent._create_openai_client", return_value=MagicMock())
    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_only_none_chunks_no_crash(self, mock_close, mock_create, mock_init):
        """Stream that is entirely None chunks does not raise AttributeError."""
        from run_agent import AIAgent

        chunks = [None, None, None]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)
        mock_create.return_value = mock_client

        agent = AIAgent(
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        try:
            agent._interruptible_streaming_api_call({})
        except AttributeError as e:
            pytest.fail(f"None chunks caused AttributeError: {e}")
        except Exception:
            pass  # other exceptions (e.g. empty response) are acceptable
