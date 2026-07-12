"""Tests for the coding-session signal fed into the skill-review prompt
(agent/background_review.py: _detect_coding_signal, spawn_background_review_thread)."""

from __future__ import annotations

import json

from agent.background_review import _detect_coding_signal, spawn_background_review_thread


def _assistant_tool_call(tool_name, arguments, call_id="call_1"):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": tool_name, "arguments": json.dumps(arguments)},
            }
        ],
    }


def _tool_result(call_id, content):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def test_no_signal_for_non_coding_conversation():
    snapshot = [
        {"role": "user", "content": "What's the weather like?"},
        {"role": "assistant", "content": "Sunny."},
    ]
    assert _detect_coding_signal(snapshot) == ""


def test_signal_on_file_edit_tool_call():
    snapshot = [
        {"role": "user", "content": "fix the bug"},
        _assistant_tool_call("Edit", {"file_path": "foo.py", "old_string": "a", "new_string": "b"}),
    ]
    signal = _detect_coding_signal(snapshot)
    assert "Coding-session signal" in signal
    assert "systematic-debugging" in signal
    assert "test-driven-development" in signal


def test_signal_on_test_runner_bash_call_includes_outcome():
    snapshot = [
        {"role": "user", "content": "run the tests"},
        _assistant_tool_call("Bash", {"command": "pytest tests/test_foo.py -q"}, call_id="call_9"),
        _tool_result("call_9", "1 failed, 3 passed\nAssertionError: expected 2 got 3"),
    ]
    signal = _detect_coding_signal(snapshot)
    assert "pytest tests/test_foo.py -q" in signal
    assert "AssertionError" in signal


def test_no_signal_for_non_test_bash_call():
    snapshot = [
        _assistant_tool_call("Bash", {"command": "ls -la"}, call_id="call_2"),
    ]
    assert _detect_coding_signal(snapshot) == ""


def test_signal_appended_to_skill_prompt_only(monkeypatch):
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent._SKILL_REVIEW_PROMPT = "SKILL BASE"
    agent._COMBINED_REVIEW_PROMPT = "COMBINED BASE"
    agent._MEMORY_REVIEW_PROMPT = "MEMORY BASE"

    coding_snapshot = [_assistant_tool_call("Edit", {"file_path": "x.py"})]

    _, skill_prompt = spawn_background_review_thread(
        agent, coding_snapshot, review_memory=False, review_skills=True,
    )
    assert skill_prompt.startswith("SKILL BASE")
    assert "Coding-session signal" in skill_prompt

    _, combined_prompt = spawn_background_review_thread(
        agent, coding_snapshot, review_memory=True, review_skills=True,
    )
    assert combined_prompt.startswith("COMBINED BASE")
    assert "Coding-session signal" in combined_prompt

    _, memory_prompt = spawn_background_review_thread(
        agent, coding_snapshot, review_memory=True, review_skills=False,
    )
    assert memory_prompt == "MEMORY BASE"


def test_no_signal_appended_for_non_coding_session():
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent._SKILL_REVIEW_PROMPT = "SKILL BASE"

    plain_snapshot = [{"role": "user", "content": "summarize this article"}]
    _, prompt = spawn_background_review_thread(
        agent, plain_snapshot, review_memory=False, review_skills=True,
    )
    assert prompt == "SKILL BASE"
