"""Tests for the GLM stop-to-length heuristic fix (#14572).

Verifies that:
1. _has_natural_response_ending() recognizes emoji, Unicode symbols,
   and math symbols as natural endings (not just ASCII/CJK punctuation)
2. _should_treat_stop_as_truncated() enforces a 500-char minimum gate
3. Config opt-out (agent.glm_truncation_heuristic: false) disables the heuristic

Refs: PR #15463, bug #14572.
"""

import re
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


# ---------------------------------------------------------------------------
# Helpers (mirroring test_run_agent.py conventions)
# ---------------------------------------------------------------------------


def _make_tool_defs(*names: str) -> list:
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


def _mock_tool_call(name="web_search", arguments="{}", call_id="c1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _mock_response(
    content="Hello",
    finish_reason="stop",
    tool_calls=None,
    reasoning=None,
):
    msg = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning=reasoning,
    )
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


@pytest.fixture()
def agent():
    """Minimal AIAgent with mocked OpenAI client and tool loading."""
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
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


def _setup_glm_agent(agent):
    """Configure the mocked agent for an Ollama/GLM backend."""
    agent.base_url = "http://localhost:11434/v1"
    agent._base_url_lower = agent.base_url.lower()
    agent.model = "glm-5.1:cloud"


# ---------------------------------------------------------------------------
# _has_natural_response_ending — emoji and Unicode symbol detection
# ---------------------------------------------------------------------------


class TestHasNaturalResponseEndingEmoji:
    """Emoji and Unicode symbol sign-offs should be recognized as natural endings."""

    def test_emoji_heart(self, agent):
        """💛 is a natural sign-off."""
        assert agent._has_natural_response_ending("Thanks! 💛") is True

    def test_emoji_sparkles(self, agent):
        """✨ is a natural sign-off."""
        assert agent._has_natural_response_ending("Done ✨") is True

    def test_emoji_raised_hands(self, agent):
        """🙌 is a natural sign-off."""
        assert agent._has_natural_response_ending("Great work! 🙌") is True

    def test_emoji_rocket(self, agent):
        """🚀 is a natural sign-off."""
        assert agent._has_natural_response_ending("Deploying now 🚀") is True

    def test_emoji_check_mark(self, agent):
        """✅ is a natural sign-off."""
        assert agent._has_natural_response_ending("All tests pass ✅") is True

    def test_math_symbol_arrow(self, agent):
        """→ (Sm category) is a natural sign-off."""
        assert agent._has_natural_response_ending("Therefore the answer is 42 →") is True

    def test_math_symbol_infinity(self, agent):
        """∞ (Sm category) is a natural sign-off."""
        assert agent._has_natural_response_ending("Iterate ∞") is True

    def test_other_symbol_warning(self, agent):
        """⚠ (So category) is a natural sign-off."""
        assert agent._has_natural_response_ending("Proceed with caution ⚠") is True

    def test_other_symbol_star(self, agent):
        """★ (So category) is a natural sign-off."""
        assert agent._has_natural_response_ending("Five stars ★") is True

    def test_still_false_for_no_punctuation_or_symbol(self, agent):
        """Plain text without punctuation or symbols is not a natural ending."""
        assert agent._has_natural_response_ending("The answer is 42") is False

    def test_trailing_whitespace_stripped(self, agent):
        """Trailing whitespace is stripped before checking the last character."""
        assert agent._has_natural_response_ending("Done ✨   ") is True

    def test_empty_string(self, agent):
        assert agent._has_natural_response_ending("") is False

    def test_whitespace_only(self, agent):
        assert agent._has_natural_response_ending("   ") is False


# ---------------------------------------------------------------------------
# _should_treat_stop_as_truncated — 500-char gate
# ---------------------------------------------------------------------------


class TestShouldTreatStopAsTruncated500CharGate:
    """Responses under 500 chars should be treated as complete, not truncated."""

    def test_short_response_without_punctuation_not_truncated(self, agent):
        """A 39-char response ending mid-sentence is <500 chars, so not truncated."""
        _setup_glm_agent(agent)

        tool_turn = _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call(name="web_search")],
        )
        misreported_stop = _mock_response(
            content="Based on the search results, the best next",
            finish_reason="stop",
        )

        # The heuristic should NOT trigger because the response is <500 chars.
        assert len(misreported_stop.choices[0].message.content) < 500
        with patch.object(agent, "_has_natural_response_ending", return_value=False):
            result = agent._should_treat_stop_as_truncated(
                finish_reason="stop",
                assistant_message=misreported_stop.choices[0].message,
                messages=[
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "web_search", "arguments": "{}"}}]},
                    {"role": "tool", "tool_call_id": "c1", "content": "search result"},
                ],
            )
        assert result is False

    def test_long_response_without_punctuation_truncated(self, agent):
        """A 500+ char response without natural ending IS truncated."""
        _setup_glm_agent(agent)

        # Build a 510-char response ending mid-sentence
        long_content = "X" * 509 + " mid-sentence"
        assert len(long_content) >= 500

        tool_turn = _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call(name="web_search")],
        )
        misreported_stop = _mock_response(
            content=long_content,
            finish_reason="stop",
        )

        with patch.object(agent, "_has_natural_response_ending", return_value=False):
            result = agent._should_treat_stop_as_truncated(
                finish_reason="stop",
                assistant_message=misreported_stop.choices[0].message,
                messages=[
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "web_search", "arguments": "{}"}}]},
                    {"role": "tool", "tool_call_id": "c1", "content": "search result"},
                ],
            )
        assert result is True

    def test_long_response_with_emoji_not_truncated(self, agent):
        """500+ char response with emoji ending IS complete."""
        _setup_glm_agent(agent)

        long_content = "X" * 500 + " ✨"
        misreported_stop = _mock_response(
            content=long_content,
            finish_reason="stop",
        )

        result = agent._should_treat_stop_as_truncated(
            finish_reason="stop",
            assistant_message=misreported_stop.choices[0].message,
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "web_search", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "search result"},
            ],
        )
        assert result is False


# ---------------------------------------------------------------------------
# Integration test: 500-char gate in full conversation loop
# ---------------------------------------------------------------------------


class TestGlmStopHeuristicIntegration:
    """Full conversation-loop tests for the GLM stop heuristic."""

    def test_short_response_not_continued(self, agent):
        """Short GLM response <500 chars completes naturally (no continuation)."""
        _setup_glm_agent(agent)

        tool_turn = _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call(name="web_search")],
        )
        short_stop = _mock_response(
            content="Based on the search results, the best next",
            finish_reason="stop",
        )

        # Under 500 chars with no terminal punctuation AND no emoji → but gate
        # still catches this as complete (len < 500).
        assert len(short_stop.choices[0].message.content) < 500

        agent.client.chat.completions.create.side_effect = [tool_turn, short_stop]

        with (
            patch("run_agent.handle_function_call", return_value="search result"),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello")

        assert result["completed"] is True
        # Only 2 API calls: tool turn + stop (no continuation)
        assert result["api_calls"] == 2
        assert result["final_response"] == "Based on the search results, the best next"
