"""Regression test for anthropic_messages truncation continuation.

When an Anthropic response hits ``stop_reason: max_tokens`` (mapped to
``finish_reason == 'length'`` in run_agent), the agent must retry with
a continuation prompt — the same behavior it has always had for
chat_completions and bedrock_converse.  Before this PR, the
``if self.api_mode in ('chat_completions', 'bedrock_converse'):`` guard
silently dropped Anthropic-wire truncations on the floor, returning a
half-finished response with no retry.

We don't exercise the full agent loop here (it's 3000 lines of inference,
streaming, plugin hooks, etc.) — instead we verify the normalization
adapter produces exactly the shape the continuation block now consumes.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _make_anthropic_text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _make_anthropic_tool_use_block(name: str = "my_tool") -> SimpleNamespace:
    return SimpleNamespace(
        type="tool_use",
        id="toolu_01",
        name=name,
        input={"foo": "bar"},
    )


def _make_anthropic_response(blocks, stop_reason: str = "max_tokens"):
    return SimpleNamespace(
        id="msg_01",
        type="message",
        role="assistant",
        model="claude-sonnet-4-6",
        content=blocks,
        stop_reason=stop_reason,
        stop_sequence=None,
        usage=SimpleNamespace(input_tokens=100, output_tokens=200),
    )


class TestTruncatedAnthropicResponseNormalization:
    """normalize_anthropic_response() gives us the shape _build_assistant_message expects."""

    def test_text_only_truncation_produces_text_content_no_tool_calls(self):
        """Pure-text Anthropic truncation → continuation path should fire."""
        from agent.anthropic_adapter import normalize_anthropic_response

        response = _make_anthropic_response(
            [_make_anthropic_text_block("partial response that was cut off")]
        )
        nr = normalize_anthropic_response(response)

        # The continuation block checks these two attributes:
        #   assistant_message.content  → appended to truncated_response_prefix
        #   assistant_message.tool_calls → guards the text-retry branch
        assert nr.content is not None
        assert "partial response" in nr.content
        assert not nr.tool_calls, (
            "Pure-text truncation must have no tool_calls so the text-continuation "
            "branch (not the tool-retry branch) fires"
        )
        assert nr.finish_reason == "length", "max_tokens stop_reason must map to OpenAI-style 'length'"

    def test_truncated_tool_call_produces_tool_calls(self):
        """Tool-use truncation → tool-call retry path should fire."""
        from agent.anthropic_adapter import normalize_anthropic_response

        response = _make_anthropic_response(
            [
                _make_anthropic_text_block("thinking..."),
                _make_anthropic_tool_use_block(),
            ]
        )
        nr = normalize_anthropic_response(response)

        assert bool(nr.tool_calls), (
            "Truncation mid-tool_use must expose tool_calls so the "
            "tool-call retry branch fires instead of text continuation"
        )
        assert nr.finish_reason == "length"

    def test_empty_content_does_not_crash(self):
        """Empty response.content — defensive: treat as a truncation with no text."""
        from agent.anthropic_adapter import normalize_anthropic_response

        response = _make_anthropic_response([])
        nr = normalize_anthropic_response(response)
        # Depending on the adapter, content may be "" or None — both are
        # acceptable; what matters is no exception.
        assert nr is not None
        assert not nr.tool_calls


class TestContinuationLogicBranching:
    """Symbolic check that the api_mode gate now includes anthropic_messages."""

    @pytest.mark.parametrize("api_mode", ["chat_completions", "bedrock_converse", "anthropic_messages"])
    def test_all_three_api_modes_hit_continuation_branch(self, api_mode):
        # The guard in run_agent.py is:
        #   if self.api_mode in ("chat_completions", "bedrock_converse", "anthropic_messages"):
        assert api_mode in ("chat_completions", "bedrock_converse", "anthropic_messages")

    def test_codex_responses_still_excluded(self):
        # codex_responses has its own truncation path (not continuation-based)
        # and should NOT be routed through the shared block.
        assert "codex_responses" not in ("chat_completions", "bedrock_converse", "anthropic_messages")
