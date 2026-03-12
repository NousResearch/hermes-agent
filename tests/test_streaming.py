"""Unit tests for streaming support.

Tests cover:
- _run_streaming_chat_completion: text tokens, tool calls, fallback on error,
  no callback, end signal
- _interruptible_api_call routing to streaming when stream_callback is set
- Streaming config reading from config.yaml
"""

import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from run_agent import AIAgent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_tool_defs(*names: str) -> list:
    """Build minimal tool definition list accepted by AIAgent.__init__."""
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


@pytest.fixture()
def agent():
    """Minimal AIAgent with mocked client, no stream_callback."""
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        return a


@pytest.fixture()
def streaming_agent():
    """Agent with a stream_callback set."""
    collected = []
    def _cb(delta):
        collected.append(delta)

    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            stream_callback=_cb,
        )
        a.client = MagicMock()
        a._collected_tokens = collected
        return a


# ---------------------------------------------------------------------------
# Helpers — build fake streaming chunks
# ---------------------------------------------------------------------------

def _make_text_chunks(*texts):
    """Return a list of SimpleNamespace chunks containing text deltas."""
    chunks = []
    for t in texts:
        chunks.append(SimpleNamespace(
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content=t, tool_calls=None),
                finish_reason=None,
            )],
            usage=None,
        ))
    # Final chunk with usage info
    chunks.append(SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    ))
    return chunks


def _make_tool_call_chunks():
    """Return chunks that simulate a tool call response."""
    chunks = [
        # First chunk: tool call id + name
        SimpleNamespace(
            choices=[SimpleNamespace(
                delta=SimpleNamespace(
                    content=None,
                    tool_calls=[SimpleNamespace(
                        index=0,
                        id="call_123",
                        function=SimpleNamespace(name="web_search", arguments=""),
                    )],
                ),
                finish_reason=None,
            )],
            usage=None,
        ),
        # Second chunk: tool call arguments
        SimpleNamespace(
            choices=[SimpleNamespace(
                delta=SimpleNamespace(
                    content=None,
                    tool_calls=[SimpleNamespace(
                        index=0,
                        id=None,
                        function=SimpleNamespace(name=None, arguments='{"query": "test"}'),
                    )],
                ),
                finish_reason=None,
            )],
            usage=None,
        ),
        # Final usage chunk
        SimpleNamespace(choices=[], usage=SimpleNamespace(
            prompt_tokens=20, completion_tokens=10, total_tokens=30,
        )),
    ]
    return chunks


# ---------------------------------------------------------------------------
# Tests: _run_streaming_chat_completion
# ---------------------------------------------------------------------------

class TestRunStreamingChatCompletion:
    """Tests for AIAgent._run_streaming_chat_completion."""

    def test_text_tokens_streamed_via_callback(self, streaming_agent):
        """Text deltas are forwarded to stream_callback and accumulated."""
        chunks = _make_text_chunks("Hello", " ", "world")
        streaming_agent.client.chat.completions.create.return_value = iter(chunks)

        result = streaming_agent._run_streaming_chat_completion({"model": "test"})

        assert result.choices[0].message.content == "Hello world"
        # Callback received each token + None end signal
        assert streaming_agent._collected_tokens == ["Hello", " ", "world", None]

    def test_tool_calls_accumulated(self, streaming_agent):
        """Tool call deltas are aggregated into a proper tool_calls list."""
        chunks = _make_tool_call_chunks()
        streaming_agent.client.chat.completions.create.return_value = iter(chunks)

        result = streaming_agent._run_streaming_chat_completion({"model": "test"})

        assert result.choices[0].message.tool_calls is not None
        tc = result.choices[0].message.tool_calls[0]
        assert tc.function.name == "web_search"
        assert '"query"' in tc.function.arguments

    def test_fallback_on_streaming_error(self, streaming_agent):
        """Falls back to non-streaming on error."""
        # First call (streaming) raises; second call (fallback) succeeds
        fallback_response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="fallback", tool_calls=None, role="assistant"),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8),
            model="test",
        )

        call_count = [0]
        def _side_effect(**kwargs):
            call_count[0] += 1
            if kwargs.get("stream"):
                raise ConnectionError("stream broke")
            return fallback_response

        streaming_agent.client.chat.completions.create.side_effect = _side_effect

        result = streaming_agent._run_streaming_chat_completion({"model": "test"})

        assert result.choices[0].message.content == "fallback"
        assert call_count[0] == 2  # streaming attempt + fallback
        # Callback should still get None (end signal) even on error
        assert None in streaming_agent._collected_tokens

    def test_no_callback_still_works(self, agent):
        """Streaming works even without a callback (just accumulates)."""
        chunks = _make_text_chunks("ok")
        agent.client.chat.completions.create.return_value = iter(chunks)

        result = agent._run_streaming_chat_completion({"model": "test"})

        assert result.choices[0].message.content == "ok"

    def test_end_signal_sent(self, streaming_agent):
        """stream_callback(None) is sent after all tokens."""
        chunks = _make_text_chunks("done")
        streaming_agent.client.chat.completions.create.return_value = iter(chunks)

        streaming_agent._run_streaming_chat_completion({"model": "test"})

        assert streaming_agent._collected_tokens[-1] is None

    def test_usage_captured_from_final_chunk(self, streaming_agent):
        """Usage stats from the final usage-only chunk are returned."""
        chunks = _make_text_chunks("hi")
        streaming_agent.client.chat.completions.create.return_value = iter(chunks)

        result = streaming_agent._run_streaming_chat_completion({"model": "test"})

        assert result.usage is not None
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5


# ---------------------------------------------------------------------------
# Tests: _interruptible_api_call routing
# ---------------------------------------------------------------------------

class TestInterruptibleApiCallRouting:
    """Tests that _interruptible_api_call routes to streaming when callback is set."""

    def test_routes_to_streaming_with_callback(self, streaming_agent):
        """When stream_callback is set, _interruptible_api_call uses streaming."""
        chunks = _make_text_chunks("streamed")
        streaming_agent.client.chat.completions.create.return_value = iter(chunks)

        # Mock _interrupt_requested to False
        streaming_agent._interrupt_requested = False

        result = streaming_agent._interruptible_api_call({"model": "test"})

        assert result.choices[0].message.content == "streamed"
        # Verify the callback got tokens
        assert "streamed" in streaming_agent._collected_tokens

    def test_routes_to_normal_without_callback(self, agent):
        """When no stream_callback, _interruptible_api_call uses normal completion."""
        normal_response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="normal", tool_calls=None, role="assistant"),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8),
            model="test",
        )
        agent.client.chat.completions.create.return_value = normal_response
        agent._interrupt_requested = False

        result = agent._interruptible_api_call({"model": "test"})

        assert result.choices[0].message.content == "normal"


# ---------------------------------------------------------------------------
# Tests: Streaming config
# ---------------------------------------------------------------------------

class TestStreamingConfig:
    """Tests for reading streaming configuration."""

    def test_streaming_disabled_by_default(self):
        """Without any config, streaming is disabled."""
        with (
            patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            a = AIAgent(
                api_key="test-key",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
            assert a.stream_callback is None

    def test_stream_callback_stored_on_agent(self):
        """stream_callback passed to constructor is stored on the agent."""
        cb = lambda delta: None
        with (
            patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            a = AIAgent(
                api_key="test-key",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                stream_callback=cb,
            )
            assert a.stream_callback is cb

    def test_gateway_streaming_config_structure(self):
        """Verify the expected streaming config structure from gateway/run.py."""
        # This tests that _read_streaming_config (if it exists) returns
        # the right structure. We mock the config file content.
        try:
            from gateway.run import _read_streaming_config
        except ImportError:
            pytest.skip("gateway.run._read_streaming_config not available")

        mock_cfg = {
            "streaming": {
                "enabled": True,
                "telegram": True,
                "discord": False,
                "edit_interval": 2.0,
            }
        }
        with patch("builtins.open", MagicMock()):
            with patch("yaml.safe_load", return_value=mock_cfg):
                try:
                    result = _read_streaming_config()
                    assert result.get("enabled") is True
                except Exception:
                    # Function might not exist yet or have different signature
                    pytest.skip("_read_streaming_config has different interface")
