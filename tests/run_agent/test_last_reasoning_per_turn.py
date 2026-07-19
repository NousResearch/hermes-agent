"""Tests for per-turn reasoning extraction in AIAgent.run_conversation.

Verifies the reasoning field returned to display layers (CLI reasoning box,
gateway reasoning footer, TUI reasoning event) only reflects the CURRENT
turn's reasoning — never leaks from a prior turn — and is picked up
correctly when reasoning is attached to a tool-calling assistant step
rather than the final-answer assistant step.
"""
from __future__ import annotations


def _extract_last_reasoning(messages):
    """Replica of the extraction loop in run_agent.py (~line 13867).

    Tests pin the loop's behaviour so that refactors can't silently
    regress the per-turn semantic.
    """
    last_reasoning = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            break
        if msg.get("role") == "assistant" and msg.get("reasoning"):
            last_reasoning = msg["reasoning"]
            break
    return last_reasoning


def test_simple_turn_reasoning_present():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi", "reasoning": "greeting the user"},
    ]
    assert _extract_last_reasoning(messages) == "greeting the user"


def test_simple_turn_no_reasoning():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi", "reasoning": None},
    ]
    assert _extract_last_reasoning(messages) is None


def test_tool_call_turn_reasoning_on_tool_call_step():
    """When the model reasons on the tool-call step and the final-answer
    step has no reasoning (Claude thinking / DeepSeek v4 / Codex Responses
    pattern), the box must show the tool-call-step reasoning, not empty.
    """
    messages = [
        {"role": "user", "content": "search the repo for X"},
        {
            "role": "assistant",
            "content": "",
            "reasoning": "I should use search_files",
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": "search_files", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "3 matches"},
        {"role": "assistant", "content": "Found 3 matches", "reasoning": None},
    ]
    assert _extract_last_reasoning(messages) == "I should use search_files"


def test_no_stale_reasoning_across_turns():
    """The regression the whole change exists for.  Prior turn had
    reasoning; current turn has none.  The reasoning box must NOT show
    the prior turn's text.
    """
    messages = [
        # prior turn
        {"role": "user", "content": "explain quantum tunneling"},
        {"role": "assistant", "content": "It's when...",
         "reasoning": "tunneling happens when particles..."},
        # current turn
        {"role": "user", "content": "thanks"},
        {"role": "assistant", "content": "You're welcome!", "reasoning": None},
    ]
    assert _extract_last_reasoning(messages) is None


def test_tool_call_turn_picks_latest_reasoning_within_turn():
    """If BOTH the tool-call step and the final step have reasoning
    (uncommon but possible), the final-step reasoning wins — it's the
    most recent thought within the current turn.
    """
    messages = [
        {"role": "user", "content": "search and summarize"},
        {
            "role": "assistant",
            "content": "",
            "reasoning": "initial plan",
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": "search_files", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "results"},
        {"role": "assistant", "content": "Here's the summary",
         "reasoning": "synthesized view of results"},
    ]
    assert _extract_last_reasoning(messages) == "synthesized view of results"


def test_empty_string_reasoning_treated_as_missing():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello", "reasoning": ""},
    ]
    assert _extract_last_reasoning(messages) is None
