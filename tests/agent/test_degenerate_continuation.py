"""Invariant tests for degenerate-continuation detection (PR #103929).

Cover the two detection rules that changed in this PR:
  * echo is checked BEFORE the length cap, so long-form repetition is caught
    (regression: the >600-char early return used to exempt it);
  * the stall-phrase heuristic only fires on a short, single-line, non-fenced
    reply, so a real answer that merely CONTAINS a stall word is not over-fired.

These are behaviour-contract tests, not snapshots.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from agent.turn_final_response import _is_degenerate_continuation


def _agent(text: str) -> MagicMock:
    """Minimal agent whose _strip_think_blocks returns the given text."""
    agent = MagicMock()
    agent._strip_think_blocks.return_value = text
    return agent


def test_long_form_echo_is_caught():
    """A >600-char reply that echoes the prior turn is degenerate.

    Regression: the length cap used to run BEFORE the echo check, exempting
    exactly the long-form repetition loop this detector exists to catch.
    """
    prior = "the quick brown fox jumps over the lazy dog. " * 40  # ~1800 chars
    current = prior[:1400]  # ~78% of prior (>600 chars), contained in prior
    agent = _agent(current)
    assert _is_degenerate_continuation(agent, current, prior) is True


def test_long_reply_without_echo_is_not_degenerate():
    """A clearly substantial, non-echo reply is never flagged (fail-open)."""
    long_text = "the quick brown fox jumps over the lazy dog. " * 40
    agent = _agent(long_text)
    assert _is_degenerate_continuation(agent, long_text, "") is False


def test_multiline_answer_containing_phrase_is_not_degenerate():
    """A real answer that merely contains 'i am sorry' is not a stall.

    Regression: raw substring matching flagged any <=600-char reply that
    contained a stall word, over-firing on "I'm sorry — here's the fix:
    <newline> ... <code>".
    """
    content = (
        "I'm sorry for the confusion. Here is the fix:\n"
        "```python\nx = 1\n```"
    )
    agent = _agent(content)
    assert _is_degenerate_continuation(agent, content, "") is False


def test_thin_stall_is_degenerate():
    """A short, single-line refusal/stall is still degenerate."""
    agent = _agent("i cannot do this")
    assert _is_degenerate_continuation(agent, "i cannot do this", "") is True


def test_short_single_line_phrase_is_degenerate():
    """A short single-line reply that is just a stall phrase still counts."""
    agent = _agent("i am sorry, i cannot")
    assert _is_degenerate_continuation(agent, "i am sorry, i cannot", "") is True


def test_echo_on_slightly_shorter_rephrasing_is_caught():
    """Echo detection still catches a near-identical rephrase of the prior."""
    prior = "the answer is forty-two and a bit more context here."
    current = "the answer is forty-two and a bit more context"  # prior minus '.'
    agent = _agent(current)
    assert _is_degenerate_continuation(agent, current, prior) is True
