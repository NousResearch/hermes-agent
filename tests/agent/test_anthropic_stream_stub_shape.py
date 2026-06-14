"""Regression tests for issue #45908 — partial-stream stub api_mode awareness.

When a malformed Anthropic stream (non-contiguous content_block indices)
crashes ``stream.get_final_message()`` with ``IndexError``, the partial-stream
recovery stub must match the active ``api_mode``:

- ``anthropic_messages`` → Anthropic-shaped (``content`` list, ``stop_reason``)
- ``chat_completions``   → OpenAI-shaped (``choices`` list)

Previously the stub was always OpenAI-shaped, causing
``AnthropicTransport.validate_response()`` to reject it (missing ``content``),
which triggered 10 useless retries in the conversation loop.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_constants import PARTIAL_STREAM_STUB_ID


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_anthropic_agent():
    """Create a minimal agent configured for anthropic_messages."""
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url="https://api.anthropic.com",
        model="claude-sonnet-4-20250514",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "anthropic_messages"
    agent._interrupt_requested = False
    return agent


def _make_chat_completions_agent():
    """Create a minimal agent configured for chat_completions."""
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    return agent


def _anthropic_text_delta(text):
    """Create a mock Anthropic content_block_delta event."""
    return SimpleNamespace(
        type="content_block_delta",
        delta=SimpleNamespace(type="text_delta", text=text),
    )


def _make_anthropic_mock_client(events=None):
    """Create a mock Anthropic client that raises IndexError on get_final_message.

    Args:
        events: Optional list of stream events to yield before the crash.
            Events must be present for the stub path to trigger (deltas_were_sent).
    """
    mock_stream = MagicMock()
    mock_stream.__iter__ = MagicMock(return_value=iter(events or []))
    mock_stream.get_final_message.side_effect = IndexError(
        "list index out of range"
    )

    mock_client = MagicMock()
    mock_stream_ctx = MagicMock()
    mock_stream_ctx.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream_ctx.__exit__ = MagicMock(return_value=False)
    mock_client.messages.stream.return_value = mock_stream_ctx

    return mock_client


# ── Stub shape tests ─────────────────────────────────────────────────────


class TestAnthropicPartialStreamStubShape:
    """The stub returned by ``_interruptible_streaming_api_call`` when
    ``stream.get_final_message()`` raises ``IndexError`` in anthropic_messages
    mode must be Anthropic-shaped so the validator accepts it.

    All tests provide stream events (text deltas) so that
    ``deltas_were_sent["yes"]`` is True and the stub path fires.
    """

    def test_anthropic_stub_has_content_list(self, monkeypatch):
        """#45908: Anthropic stub must have .content (list), not .choices."""
        agent = _make_anthropic_agent()
        agent._current_streamed_assistant_text = "Partial response text"

        events = [
            _anthropic_text_delta("Partial "),
            _anthropic_text_delta("response text"),
        ]
        agent._anthropic_client = _make_anthropic_mock_client(events)

        monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
        with patch.object(agent, "_try_refresh_anthropic_client_credentials"):
            response = agent._interruptible_streaming_api_call({})

        # Must have Anthropic-shaped content list
        assert hasattr(response, "content"), (
            "Anthropic stub must have .content attribute"
        )
        assert isinstance(response.content, list), (
            "Anthropic stub .content must be a list"
        )
        assert len(response.content) >= 1, (
            "Anthropic stub .content must be non-empty"
        )
        assert response.content[0].type == "text"
        assert response.content[0].text == "Partial response text"

        # Must NOT have OpenAI-shaped choices
        assert not hasattr(response, "choices") or response.choices is None, (
            "Anthropic stub must not have .choices"
        )

        # Must have stop_reason for finish_reason mapping
        assert response.stop_reason == "max_tokens"

        # Must preserve PARTIAL_STREAM_STUB_ID
        assert response.id == PARTIAL_STREAM_STUB_ID

    def test_anthropic_stub_passes_validator(self, monkeypatch):
        """#45908: The Anthropic stub must pass AnthropicTransport.validate_response."""
        from agent.transports.anthropic import AnthropicTransport

        agent = _make_anthropic_agent()
        agent._current_streamed_assistant_text = "Some text"
        agent._anthropic_client = _make_anthropic_mock_client(
            [_anthropic_text_delta("Some text")]
        )

        monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
        with patch.object(agent, "_try_refresh_anthropic_client_credentials"):
            response = agent._interruptible_streaming_api_call({})

        transport = AnthropicTransport()
        assert transport.validate_response(response), (
            "Anthropic stub must pass validate_response"
        )

    def test_anthropic_stub_normalizes_correctly(self, monkeypatch):
        """#45908: The Anthropic stub must normalize through AnthropicTransport."""
        from agent.transports.anthropic import AnthropicTransport

        agent = _make_anthropic_agent()
        agent._current_streamed_assistant_text = "Recovered text"
        agent._anthropic_client = _make_anthropic_mock_client(
            [_anthropic_text_delta("Recovered text")]
        )

        monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
        with patch.object(agent, "_try_refresh_anthropic_client_credentials"):
            response = agent._interruptible_streaming_api_call({})

        transport = AnthropicTransport()
        normalized = transport.normalize_response(response)

        assert normalized.content == "Recovered text"
        assert normalized.finish_reason == "length"

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_chat_completions_stub_unchanged(self, _mock_close, mock_create, monkeypatch):
        """chat_completions mode still returns OpenAI-shaped stub."""

        def _stalling_stream():
            yield SimpleNamespace(
                choices=[SimpleNamespace(
                    index=0,
                    delta=SimpleNamespace(
                        content="Some text", tool_calls=None,
                        reasoning_content=None, reasoning=None,
                    ),
                    finish_reason=None,
                )],
                model=None, usage=None,
            )
            raise RuntimeError("simulated stall")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = lambda *a, **kw: _stalling_stream()
        mock_create.return_value = mock_client

        agent = _make_chat_completions_agent()
        agent._current_streamed_assistant_text = "Some text"

        monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
        response = agent._interruptible_streaming_api_call({})

        # Must have OpenAI-shaped choices
        assert hasattr(response, "choices")
        assert isinstance(response.choices, list)
        assert len(response.choices) >= 1
        assert response.choices[0].message.content == "Some text"


# ── _is_provider_stream_parse_error tests ─────────────────────────────────


class TestIsProviderStreamParseErrorIndexError:
    """_is_provider_stream_parse_error must classify IndexError from
    non-contiguous content_block indices as a stream parse error."""

    def test_index_error_classified_as_parse_error(self):
        """#45908: IndexError from SDK accumulator is a stream parse error."""
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://api.anthropic.com",
            model="claude-sonnet-4-20250514",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "anthropic_messages"

        err = IndexError("list index out of range")
        assert agent._is_provider_stream_parse_error(err) is True

    def test_index_error_not_anthropic_mode_returns_false(self):
        """IndexError should only be classified for anthropic_messages mode."""
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://example.com/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"

        err = IndexError("list index out of range")
        assert agent._is_provider_stream_parse_error(err) is False

    def test_value_error_still_works(self):
        """Existing ValueError classification must be preserved."""
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://api.anthropic.com",
            model="claude-sonnet-4-20250514",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "anthropic_messages"

        err = ValueError("expected ident at line 1 column 149")
        assert agent._is_provider_stream_parse_error(err) is True

    def test_unicode_encode_error_still_rejected(self):
        """UnicodeEncodeError must still be rejected."""
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://api.anthropic.com",
            model="claude-sonnet-4-20250514",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "anthropic_messages"

        err = UnicodeEncodeError("utf-8", "\x80", 0, 1, "invalid")
        assert agent._is_provider_stream_parse_error(err) is False
