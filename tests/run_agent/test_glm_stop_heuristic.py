"""Tests for Ollama/GLM stop-to-length truncation heuristic (bug #14572).

The heuristic in _should_treat_stop_as_truncated() reclassifies certain
Ollama/GLM finish_reason='stop' responses as truncated, then requests
continuations.  This is necessary because Ollama/GLM sometimes reports
genuinely truncated output as stop rather than length.

However, the heuristic was too aggressive: any response not ending with
a whitelisted punctuation character was treated as truncated.  Since
the whitelist only covered ASCII and CJK punctuation, responses ending
with emoji (like yellow_heart, sparkles, raised_hands), Markdown links,
or just bare words were all false-positive triggers -- each one wasted
up to 3 continuation API calls.

These tests verify the fix:
  1. _has_natural_response_ending() now recognizes emoji and other
     common Unicode sign-off characters as natural endings.
  2. _should_treat_stop_as_truncated() has a minimum-length gate
     (short responses within max_tokens budget are NOT suspicious).
  3. A config flag agent.glm_truncation_heuristic allows users to
     disable the heuristic entirely if they hit edge cases.
"""

import pytest
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_tool_call(name="web_search", arguments="{}", call_id="c1"):
    tc = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    tc.id = call_id
    tc.type = "function"
    return tc


def _mock_response(content="", finish_reason="stop", tool_calls=None):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.finish_reason = finish_reason
    msg.function_call = None
    msg.refusal = None
    msg.role = "assistant"
    msg.audio = None
    msg.annotations = []
    return msg


def _make_agent(base_url="http://localhost:11434/v1", model="glm-5.1:cloud"):
    """Create a minimal AIAgent wired for Ollama/GLM."""
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=MagicMock(),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url=base_url,
            model=model,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._base_url_lower = base_url.lower()
    agent.client = MagicMock()
    return agent


# ===========================================================================
# _has_natural_response_ending
# ===========================================================================

class TestHasNaturalResponseEnding:
    """Tests for AIAgent._has_natural_response_ending (static method)."""

    @pytest.mark.parametrize("ending", [
        ".", "!", "?", ":", ")", "\"", "'", "]", "}",
        "\u3002", "\uff01", "\uff1f", "\uff1a", "\uff09", "\u3011", "\u300d", "\u300f", "\u300b",
    ])
    def test_ascii_and_cjk_punctuation(self, ending):
        assert AIAgent._has_natural_response_ending(f"Hello world{ending}") is True

    @pytest.mark.parametrize("emoji", [
        "\U0001f49b",  # 💛
        "\u2728",       # ✨
        "\U0001f64c",  # 🙌
        "\U0001f919",  # 🤙
        "\U0001f525",  # 🔥
        "\U0001f4aa",  # 💪
        "\U0001f680",  # 🚀
        "\U0001f60e",  # 😎
        "\U0001f60a",  # 😊
        "\U0001f44b",  # 👋
        "\u2764\ufe0f", # ❤️
        "\U0001f64f",  # 🙏
        "\U0001f44d",  # 👍
        "\U0001f4af",  # 💯
        "\U0001f389",  # 🎉
        "\U0001fae1",  # 🫡
    ])
    def test_emoji_sign_offs_are_natural_endings(self, emoji):
        """Emoji at end of response should count as natural ending (#14572)."""
        assert AIAgent._has_natural_response_ending(
            f"Here's your answer. Let me know if you need more {emoji}"
        ) is True

    @pytest.mark.parametrize("symbol", [
        "\u2192",  # → (RIGHTWARDS ARROW, Sm)
        "\u2190",  # ← (LEFTWARDS ARROW, Sm)
        "\u2794",  # ➔ (RIGHTWARD ARROW, So — broad arrow variant)
        "\u221e",  # ∞ (INFINITY, Sm)
        "\u2248",  # ≈ (ALMOST EQUAL TO, Sm)
        "\u2713",  # ✓ (CHECK MARK, So — was in old hardcoded list)
        "\u2764",  # ❤ (HEAVY BLACK HEART, So — was in old hardcoded list)
        "\u2605",  # ★ (BLACK STAR, So — was in old hardcoded list)
    ])
    def test_math_symbols_and_sign_off_glyphs_are_natural(self, symbol):
        """Math symbols (Sm) and sign-off glyphs (So) are natural endings.

        The Sm category covers arrows (→ ←) and math symbols (∞ ≈) that
        commonly appear at the end of structured responses. The So category
        covers check marks (✓), hearts (❤), and stars (★) that were previously
        in a now-removed hardcoded string.
        """
        assert AIAgent._has_natural_response_ending(
            f"Here's the result {symbol}"
        ) is True

    @pytest.mark.parametrize("text", [
        "```python\nprint('hello')\n```",
        "```",
        "Here is the output:\n```\nDone\n```",
    ])
    def test_code_block_endings(self, text):
        assert AIAgent._has_natural_response_ending(text) is True

    @pytest.mark.parametrize("bare_word", [
        "the best next",
        "here are the results from the search",
        "updating the config file now",
    ])
    def test_bare_words_without_punctuation_are_not_natural(self, bare_word):
        assert AIAgent._has_natural_response_ending(bare_word) is False

    def test_empty_string(self):
        assert AIAgent._has_natural_response_ending("") is False

    def test_whitespace_only(self):
        assert AIAgent._has_natural_response_ending("   \n\t  ") is False

    @pytest.mark.parametrize("dash_ending", [
        "Here's the list:\n- Item 1\n- Item 2\n-",
    ])
    def test_trailing_dash_in_list(self, dash_ending):
        """A trailing dash from a markdown list is not a natural ending."""
        assert AIAgent._has_natural_response_ending(dash_ending) is False

    @pytest.mark.parametrize("url_ending", [
        "Check out https://example.com/docs",
        "More info at http://test.com",
    ])
    def test_urls_without_punctuation(self, url_ending):
        """URLs ending without punctuation look like truncation."""
        assert AIAgent._has_natural_response_ending(url_ending) is False


# ===========================================================================
# _should_treat_stop_as_truncated
# ===========================================================================

class TestShouldTreatStopAsTruncated:
    """Tests for AIAgent._should_treat_stop_as_truncated."""

    def test_emoji_ending_does_not_trigger_heuristic(self):
        """A response ending with emoji should NOT be treated as truncated."""
        agent = _make_agent()
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "", "tool_calls": [_mock_tool_call()]},
            {"role": "tool", "content": "search result", "tool_call_id": "c1"},
        ]
        response = _mock_response(
            content="Here's what I found! Let me know if you need more details \U0001f49b",
            finish_reason="stop",
        )
        assert agent._should_treat_stop_as_truncated("stop", response, messages) is False

    def test_bare_word_ending_still_triggers_heuristic(self):
        """A genuinely suspicious bare-word ending SHOULD still trigger continuation.
        Must be >=500 chars to pass the minimum-length gate."""
        agent = _make_agent()
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "", "tool_calls": [_mock_tool_call()]},
            {"role": "tool", "content": "search result", "tool_call_id": "c1"},
        ]
        # Long response that clearly trails off mid-sentence
        long_text = (
            "Based on the search results, I found several relevant findings that "
            "connect to your query. The first result indicates that the configuration "
            "parameters need to be adjusted to match the new requirements. Specifically, "
            "the timeout value should be increased from 30 to 120 seconds, and the "
            "retry count should be set to 5 rather than 3. Additionally, the endpoint "
            "URL needs to be updated to reflect the new server location, which is now "
            "at https://api.v2.example.com instead of the previous one. The migration "
            "guide suggests running the update script with the --force flag to ensure "
            "all changes are applied correctly, and then verifying by checking the "
            "configuration of the"
        )
        response = _mock_response(
            content=long_text,
            finish_reason="stop",
        )
        assert agent._should_treat_stop_as_truncated("stop", response, messages) is True

    def test_period_ending_does_not_trigger(self):
        """A response ending with period is clearly natural."""
        agent = _make_agent()
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "", "tool_calls": [_mock_tool_call()]},
            {"role": "tool", "content": "result", "tool_call_id": "c1"},
        ]
        response = _mock_response(
            content="Based on the search results, the best next step is to update the config.",
            finish_reason="stop",
        )
        assert agent._should_treat_stop_as_truncated("stop", response, messages) is False

    def test_short_response_within_budget_does_not_trigger(self):
        """A short response (well within max_tokens) ending with a bare word
        is almost certainly complete -- it didn't hit any token limit.
        This addresses the false-positive case where the heuristic fires
        on short conversational replies that just happen to lack punctuation."""
        agent = _make_agent()
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "", "tool_calls": [_mock_tool_call()]},
            {"role": "tool", "content": "result", "tool_call_id": "c1"},
        ]
        response = _mock_response(
            content="Sure, I'll look into that for you",
            finish_reason="stop",
        )
        assert agent._should_treat_stop_as_truncated("stop", response, messages) is False

    def test_config_flag_disables_heuristic(self):
        """When glm_truncation_heuristic is set to False in config,
        the heuristic should never trigger, regardless of content."""
        agent = _make_agent()
        agent._glm_truncation_heuristic_enabled = False
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "", "tool_calls": [_mock_tool_call()]},
            {"role": "tool", "content": "result", "tool_call_id": "c1"},
        ]
        # Long suspicious text that would normally trigger
        long_text = (
            "Based on the search results, I found several relevant findings that "
            "connect to your query. The first result indicates that the configuration "
            "parameters need to be adjusted to match the new requirements. Specifically, "
            "the timeout value should be increased from 30 to 120 seconds, and the "
            "retry count should be set to 5 rather than 3. Additionally, the endpoint "
            "URL needs to be updated to reflect the new server location, which is now "
            "at https://api.v2.example.com instead of the previous one. The migration "
            "guide suggests running the update script with the --force flag to ensure "
            "all changes are applied correctly, and then verifying by checking the "
            "configuration of the"
        )
        response = _mock_response(
            content=long_text,
            finish_reason="stop",
        )
        assert agent._should_treat_stop_as_truncated("stop", response, messages) is False

    def test_non_ollama_provider_never_triggers(self):
        """Non-Ollama providers should never have this heuristic applied."""
        agent = _make_agent(base_url="https://api.openai.com/v1", model="gpt-4o-mini")
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "", "tool_calls": [_mock_tool_call()]},
            {"role": "tool", "content": "result", "tool_call_id": "c1"},
        ]
        response = _mock_response(
            content="Based on the search results, the best next step is to update the",
            finish_reason="stop",
        )
        assert agent._should_treat_stop_as_truncated("stop", response, messages) is False

    def test_finish_reason_length_always_false(self):
        """finish_reason='length' should not re-trigger; it's already length."""
        agent = _make_agent()
        messages = [{"role": "user", "content": "hello"}]
        response = _mock_response(content="truncated text", finish_reason="length")
        assert agent._should_treat_stop_as_truncated("length", response, messages) is False

    def test_no_tool_messages_never_triggers(self):
        """Without tool messages in context, the heuristic doesn't apply."""
        agent = _make_agent()
        messages = [
            {"role": "user", "content": "hello"},
        ]
        response = _mock_response(
            content="Here is the answer without punctuation",
            finish_reason="stop",
        )
        assert agent._should_treat_stop_as_truncated("stop", response, messages) is False

    def test_tool_call_response_never_triggers(self):
        """If the assistant message has tool calls, it's not a text response."""
        agent = _make_agent()
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "", "tool_calls": [_mock_tool_call()]},
            {"role": "tool", "content": "result", "tool_call_id": "c1"},
        ]
        response = _mock_response(
            content="",
            finish_reason="stop",
            tool_calls=[_mock_tool_call(name="another_search", call_id="c2")],
        )
        assert agent._should_treat_stop_as_truncated("stop", response, messages) is False

    def test_short_response_with_emoji_does_not_trigger(self):
        """A short response (<500 chars) ending with emoji is not truncated.

        Tests the 500-char minimum-length gate: short conversational replies
        that happen to end with emoji are complete, not truncated continuations.
        """
        agent = _make_agent()
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "", "tool_calls": [_mock_tool_call()]},
            {"role": "tool", "content": "result", "tool_call_id": "c1"},
        ]
        response = _mock_response(
            content="The configuration file is located at /etc/app/config.yaml. Let me know if you need more help \u2728",
            finish_reason="stop",
        )
        assert agent._should_treat_stop_as_truncated("stop", response, messages) is False

    def test_think_tags_stripped_period_ending(self):
        """Reasoning tags stripped, visible text ends with period."""
        agent = _make_agent()
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "", "tool_calls": [_mock_tool_call()]},
            {"role": "tool", "content": "result", "tool_call_id": "c1"},
        ]
        response = _mock_response(
            content="<tool_call>Let me think about this problem carefully.</think>The answer is 42.",
            finish_reason="stop",
        )
        # After stripping think blocks, visible text ends with "." -- natural
        assert agent._should_treat_stop_as_truncated("stop", response, messages) is False

    def test_think_tags_stripped_bare_word_still_triggers(self):
        """Reasoning tags stripped, bare word still looks truncated (>500 chars after strip)."""
        agent = _make_agent()
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "", "tool_calls": [_mock_tool_call()]},
            {"role": "tool", "content": "result", "tool_call_id": "c1"},
        ]
        long_thought = "I need to think very carefully about this problem and consider all the possible angles and implications of each approach before giving my final answer to this complex question that requires deep analysis"
        long_visible = (
            "Based on the search results, I found several relevant findings that "
            "connect to your query. The first result indicates that the configuration "
            "parameters need to be adjusted to match the new requirements. Specifically, "
            "the timeout value should be increased from 30 to 120 seconds, and the "
            "retry count should be set to 5 rather than 3. Additionally, the endpoint "
            "URL needs to be updated to reflect the new server location, which is now "
            "at https://api.v2.example.com instead of the previous one. The migration "
            "guide suggests running the update script with the --force flag to ensure "
            "all changes are applied correctly, and then verifying by checking the "
            "configuration of the"
        )
        combined = "<think>" + long_thought + "</think>\n\n" + long_visible
        response = _mock_response(
            content=combined,
            finish_reason="stop",
        )
        assert agent._should_treat_stop_as_truncated("stop", response, messages) is True
