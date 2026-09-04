"""Sentinel-only streams must never persist as the visible answer (#102566).

Behavior contract: when a normal (finish_reason=stop) tool-free stream
assembles to exactly the internal ``[response interrupted]`` sentinel (full
22-char form or the 21-char mid-sentinel truncation), the bytes came from the
transport — proxy error-injection or an interrupted-but-resumed stream — not
from the model. The turn must surface a retryable ProviderStreamError (or a
history-safe empty length stub when fragments were already delivered), never
a persisted ``[response interrupted]`` assistant row.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _make_chunk(content=None, tool_calls=None, finish_reason=None, model="test/model"):
    """Build a SimpleNamespace mimicking an OpenAI streaming chunk."""
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(model=model, choices=[choice])


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
    """Minimal AIAgent with mocked OpenAI client and tool loading."""
    with (
        patch(
            "run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        return a


def _placeholder_stream(contents, finish_reason="stop"):
    chunks = [_make_chunk(content=c) for c in contents]
    chunks.append(_make_chunk(finish_reason=finish_reason))
    return chunks


class TestIsInterruptedPlaceholderText:
    def test_exact_sentinel_matches(self):
        from agent.chat_completion_helpers import _is_interrupted_placeholder_text

        assert _is_interrupted_placeholder_text("[response interrupted]") is True

    def test_truncated_sentinel_matches(self):
        from agent.chat_completion_helpers import _is_interrupted_placeholder_text

        assert _is_interrupted_placeholder_text("[response interrupted") is True

    def test_padded_sentinel_matches(self):
        from agent.chat_completion_helpers import _is_interrupted_placeholder_text

        assert _is_interrupted_placeholder_text("  [response interrupted]\n") is True

    def test_normal_text_does_not_match(self):
        from agent.chat_completion_helpers import _is_interrupted_placeholder_text

        assert _is_interrupted_placeholder_text("Hello world!") is False
        assert _is_interrupted_placeholder_text("") is False
        assert _is_interrupted_placeholder_text(None) is False

    def test_longer_text_merely_containing_sentinel_does_not_match(self):
        """Quoting the sentinel (e.g. discussing this bug) must keep working."""
        from agent.chat_completion_helpers import _is_interrupted_placeholder_text

        assert (
            _is_interrupted_placeholder_text(
                "I saw [response interrupted] in my transcript, why?"
            )
            is False
        )


class TestPlaceholderOnlyStream:
    def test_placeholder_only_stream_raises_provider_stream_error(self, agent):
        """Single-delta sentinel with finish_reason=stop is stream corruption."""
        from agent.chat_completion_helpers import ProviderStreamError

        agent.client.chat.completions.create.return_value = iter(
            _placeholder_stream(["[response interrupted]"])
        )
        agent.stream_delta_callback = MagicMock()

        with pytest.raises(ProviderStreamError) as exc_info:
            agent._interruptible_streaming_api_call({"messages": []})

        assert exc_info.value.body["error"]["code"] == "placeholder_stream_corruption"
        # Nothing was delivered to the display: no fake answer, no pollution.
        agent.stream_delta_callback.assert_not_called()

    def test_truncated_placeholder_variant_raises(self, agent):
        """21-char mid-sentinel cut is the same corruption class."""
        from agent.chat_completion_helpers import ProviderStreamError

        agent.client.chat.completions.create.return_value = iter(
            _placeholder_stream(["[response interrupted"])
        )
        agent.stream_delta_callback = MagicMock()

        with pytest.raises(ProviderStreamError):
            agent._interruptible_streaming_api_call({"messages": []})

        agent.stream_delta_callback.assert_not_called()

    def test_split_placeholder_never_persists_in_stub(self, agent):
        """Sentinel split across deltas escapes per-delta suppression: each
        fragment is delivered, so the error lands on the length-stub path —
        but the stub must carry NO content (empty stubs are skipped from
        history), never the assembled placeholder."""
        agent.client.chat.completions.create.return_value = iter(
            _placeholder_stream(["[response ", "interrupted]"])
        )
        agent.stream_delta_callback = MagicMock()

        stub = agent._interruptible_streaming_api_call({"messages": []})

        assert stub.choices[0].finish_reason == "length"
        assert stub.choices[0].message.content in (None, "")

    def test_normal_text_stream_unaffected(self, agent):
        """Control: ordinary text still completes and streams normally."""
        agent.client.chat.completions.create.return_value = iter(
            _placeholder_stream(["Hello", " world!"])
        )
        agent.stream_delta_callback = MagicMock()

        response = agent._interruptible_streaming_api_call({"messages": []})

        assert response.choices[0].message.content == "Hello world!"
        assert response.choices[0].finish_reason == "stop"
        assert agent.stream_delta_callback.called

    def test_nonstop_finish_reason_not_flagged(self, agent):
        """Only a *normal* stop turn is corruption; other terminals keep
        their existing handling."""
        from agent.chat_completion_helpers import _provider_stream_placeholder_error

        assert (
            _provider_stream_placeholder_error(
                "[response interrupted]",
                "length",
                has_tool_calls=False,
            )
            is None
        )

    def test_tool_call_turn_not_flagged(self, agent):
        """Tool-call turns have their own corruption handling; the guard
        stays out of their way."""
        from agent.chat_completion_helpers import _provider_stream_placeholder_error

        assert (
            _provider_stream_placeholder_error(
                "[response interrupted]",
                "stop",
                has_tool_calls=True,
            )
            is None
        )
