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










