"""Tests for _has_content_after_think_block / _strip_think_blocks consistency.

Ensures both functions recognise the same set of reasoning tag variants so
that models using <thinking>, <reasoning>, or <REASONING_SCRATCHPAD>
(Qwen, local models, etc.) are handled consistently.
"""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def agent():
    """Create a minimal AIAgent with only the methods under test."""
    with patch("run_agent.AIAgent.__init__", return_value=None):
        from run_agent import AIAgent
        a = AIAgent.__new__(AIAgent)
        return a


# ── _has_content_after_think_block ──────────────────────────────────────

class TestHasContentAfterThinkBlock:
    """The function must return False when the response contains ONLY
    reasoning tags (any variant) and no user-visible content."""

    def test_empty_string(self, agent):
        assert agent._has_content_after_think_block("") is False

    def test_none(self, agent):
        assert agent._has_content_after_think_block(None) is False

    def test_plain_text(self, agent):
        assert agent._has_content_after_think_block("Hello world") is True

    # ── <think> (DeepSeek-R1 style) ──
    def test_think_only(self, agent):
        assert agent._has_content_after_think_block("<think>reasoning</think>") is False

    def test_think_with_content(self, agent):
        assert agent._has_content_after_think_block(
            "<think>reasoning</think>\nHere is my answer."
        ) is True

    # ── <thinking> (Qwen, local models) ──
    def test_thinking_only(self, agent):
        """Previously returned True (bug) — <thinking> wasn't stripped."""
        assert agent._has_content_after_think_block(
            "<thinking>long reasoning here</thinking>"
        ) is False

    def test_thinking_with_content(self, agent):
        assert agent._has_content_after_think_block(
            "<thinking>reasoning</thinking>\nActual answer."
        ) is True

    def test_THINKING_upper(self, agent):
        assert agent._has_content_after_think_block(
            "<THINKING>reasoning</THINKING>"
        ) is False

    # ── <reasoning> ──
    def test_reasoning_only(self, agent):
        assert agent._has_content_after_think_block(
            "<reasoning>step by step</reasoning>"
        ) is False

    def test_reasoning_with_content(self, agent):
        assert agent._has_content_after_think_block(
            "<reasoning>step by step</reasoning>\nThe answer is 42."
        ) is True

    # ── <REASONING_SCRATCHPAD> ──
    def test_scratchpad_only(self, agent):
        assert agent._has_content_after_think_block(
            "<REASONING_SCRATCHPAD>work</REASONING_SCRATCHPAD>"
        ) is False

    def test_scratchpad_with_content(self, agent):
        assert agent._has_content_after_think_block(
            "<REASONING_SCRATCHPAD>work</REASONING_SCRATCHPAD>\nDone."
        ) is True

    # ── Mixed tags ──
    def test_mixed_tags_only(self, agent):
        content = (
            "<think>step 1</think>\n"
            "<thinking>step 2</thinking>\n"
            "<reasoning>step 3</reasoning>"
        )
        assert agent._has_content_after_think_block(content) is False

    def test_mixed_tags_with_content(self, agent):
        content = (
            "<think>step 1</think>\n"
            "<thinking>step 2</thinking>\n"
            "Final answer: 42"
        )
        assert agent._has_content_after_think_block(content) is True

    # ── Multiline reasoning ──
    def test_multiline_thinking(self, agent):
        content = "<thinking>\nline 1\nline 2\nline 3\n</thinking>"
        assert agent._has_content_after_think_block(content) is False

    def test_whitespace_only_after_tags(self, agent):
        content = "<think>reasoning</think>\n   \n  \t  "
        assert agent._has_content_after_think_block(content) is False


# ── Consistency check ───────────────────────────────────────────────────

class TestConsistencyWithStripThinkBlocks:
    """_has_content_after_think_block(x) should return False whenever
    _strip_think_blocks(x).strip() is empty, and vice versa."""

    SAMPLES = [
        "<think>x</think>",
        "<thinking>x</thinking>",
        "<THINKING>X</THINKING>",
        "<reasoning>x</reasoning>",
        "<REASONING_SCRATCHPAD>x</REASONING_SCRATCHPAD>",
        "<think>a</think><thinking>b</thinking>",
        "<think>a</think> real content",
        "<thinking>a</thinking> real content",
        "plain text",
        "",
    ]

    def test_consistent_detection(self, agent):
        for sample in self.SAMPLES:
            stripped = agent._strip_think_blocks(sample).strip()
            has_content = agent._has_content_after_think_block(sample)
            assert has_content == bool(stripped), (
                f"Inconsistency for {sample!r}: "
                f"_has_content={has_content}, stripped={stripped!r}"
            )
