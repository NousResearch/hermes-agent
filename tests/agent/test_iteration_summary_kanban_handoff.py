"""A kanban worker's iteration-limit summary keeps the hand-off text of its terminal board call.

The worker guidance tells a worker to end every turn with ``kanban_complete`` /
``kanban_block``, and the summary request keeps the tools declared (prompt-cache lineage), so a
worker often answers "summarize what you found" with that call instead of text. The summary read
discarded every tool call: the worker's answer was lost, the empty-summary retry was spent, and
the turn ended on the fixed fallback line.
"""

import json
import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.chat_completion_helpers import _EMPTY_SUMMARY_RESPONSE
from agent.kanban_stop import _HANDOFF_TEXT_FIELDS, _TERMINAL_KANBAN_TOOLS, terminal_handoff_text

_ANSWER = "Two of the three exports were rebuilt; the third needs the missing date range."


def _call(name, arguments):
    return SimpleNamespace(id="call_1", type="function",
                           function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


def _completion(*tool_calls, content=None):
    message = SimpleNamespace(role="assistant", content=content, tool_calls=list(tool_calls) or None)
    return SimpleNamespace(choices=[SimpleNamespace(index=0, message=message, finish_reason="tool_calls")],
                           usage=SimpleNamespace(completion_tokens=24), model="m")


@pytest.fixture
def agent():
    with patch("agent.process_bootstrap.OpenAI"):
        from run_agent import AIAgent

        agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
                        quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=[])
        agent.api_mode = "chat_completions"
        yield agent


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_summary")


def _summarize(agent, *responses):
    agent.client.chat.completions.create.side_effect = list(responses)
    messages = [{"role": "user", "content": "work kanban task t_summary"}]
    return agent._handle_max_iterations(messages, 60), messages


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("kanban_complete", {"summary": _ANSWER, "result": "ignored when a summary is given"}),
        ("kanban_complete", {"result": _ANSWER}),
        ("kanban_block", {"reason": _ANSWER}),
        ("kanban_request_review", {"summary": _ANSWER}),
        ("kanban_request_changes", {"reason": _ANSWER}),
    ],
)
def test_a_workers_terminal_call_is_its_summary(agent, worker, name, arguments):
    text, messages = _summarize(agent, _completion(_call(name, arguments)))

    assert text == _ANSWER
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == _ANSWER
    # Answered on the first request: the empty-summary retry is not spent.
    assert agent.client.chat.completions.create.call_count == 1


@pytest.mark.parametrize(
    "bridge_arguments",
    [
        {"calls": [{"name": "kanban_block", "arguments": {"reason": _ANSWER}}]},
        {"calls": json.dumps([{"name": "kanban_block", "arguments": {"reason": _ANSWER}}])},
        {"name": "kanban_block", "arguments": json.dumps({"reason": _ANSWER})},
    ],
    ids=["calls-array", "calls-json-string", "legacy-single-shape"],
)
def test_a_terminal_call_through_the_tool_search_bridge_is_read(agent, worker, bridge_arguments):
    text, _ = _summarize(agent, _completion(_call("tool_call", bridge_arguments)))

    assert text == _ANSWER


def test_the_hand_off_wins_over_narration_next_to_it(agent, worker):
    response = _completion(_call("kanban_block", {"reason": _ANSWER}), content="I'll hand this card back now.")

    assert _summarize(agent, response)[0] == _ANSWER


def test_outside_a_worker_the_tool_calls_are_still_discarded(agent, monkeypatch, caplog):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    response = _completion(_call("kanban_complete", {"summary": _ANSWER}))

    with caplog.at_level(logging.WARNING, logger="agent.chat_completion_helpers"):
        text, _ = _summarize(agent, response, response)

    assert text == _EMPTY_SUMMARY_RESPONSE
    assert agent.client.chat.completions.create.call_count == 2
    assert "Iteration summary emitted tool calls; discarding them" in caplog.text


def test_a_fork_turn_keeps_the_discard(agent, worker):
    # A background review inherits the worker's task env but is not the worker.
    agent._turn_origin = "background_review"
    response = _completion(_call("kanban_complete", {"summary": _ANSWER}))

    assert _summarize(agent, response, response)[0] == _EMPTY_SUMMARY_RESPONSE


def test_a_delegated_child_keeps_the_discard(agent, worker):
    from agent.delegation_context import delegated_child_context

    response = _completion(_call("kanban_complete", {"summary": _ANSWER}))
    with delegated_child_context():
        assert _summarize(agent, response, response)[0] == _EMPTY_SUMMARY_RESPONSE


def test_a_workers_lookup_call_is_still_discarded(agent, worker):
    response = _completion(_call("search_files", {"pattern": "export"}))

    assert _summarize(agent, response, response)[0] == _EMPTY_SUMMARY_RESPONSE
    assert agent.client.chat.completions.create.call_count == 2


def test_every_terminal_board_tool_has_a_hand_off_field():
    assert set(_HANDOFF_TEXT_FIELDS) == _TERMINAL_KANBAN_TOOLS


@pytest.mark.parametrize(
    ("tool_calls", "expected"),
    [
        ([{"function": {"name": "kanban_complete", "arguments": {"summary": " done "}}}], "done"),
        ([{"name": "kanban_block", "arguments": json.dumps({"reason": "need input"})}], "need input"),
        ([{"function": {"name": "kanban_complete", "arguments": "{not json"}}], ""),
        ([{"function": {"name": "kanban_block", "arguments": {"reason": "   "}}}], ""),
        ([{"function": {"name": "tool_call", "arguments": {"calls": "{not json"}}}], ""),
        (None, ""),
    ],
)
def test_terminal_handoff_text_reads_message_shaped_calls(tool_calls, expected):
    assert terminal_handoff_text(tool_calls) == expected
