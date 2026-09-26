"""Resolved clarification is user input, not disposable tool output (#124169)."""
import json

import pytest

from agent.context_compressor import ContextCompressor


def _exchange(call_id, name, content):
    return [
        {"role": "assistant", "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": call_id, "content": content},
    ]


@pytest.fixture
def compressor():
    return ContextCompressor(model="test/model", config_context_length=100000, quiet_mode=True)


@pytest.mark.parametrize("batch", [False, True])
def test_resolved_answers_survive_repeated_pruning_and_reach_summary(compressor, batch):
    answers = [f"Decision {i}: " + "Keep the existing authorization boundary. " * 3 for i in range(5)]
    payload = ({"responses": [{"question": f"Scope {i}?", "user_response": answer} for i, answer in enumerate(answers)]}
               if batch else {"question": "Which scopes?", "user_response": answers})
    content = json.dumps(payload)
    messages = _exchange("consent", "clarify", content) + _exchange("noise", "terminal", "x" * 3000)
    for _ in range(2):
        messages, _ = compressor._prune_old_tool_results(messages, protect_tail_count=0, min_prune_chars=1)
        assert messages[1]["content"] == content
    summary_input = compressor._serialize_for_summary(messages)
    for answer in answers:
        assert answer in summary_input
    assert messages[3]["content"] != "x" * 3000


def test_duplicate_answers_keep_each_question_context(compressor):
    content = json.dumps({"user_response": "Do not deploy without my approval. " * 10})
    messages = _exchange("first", "clarify", content) + _exchange("second", "clarify", content)
    result, count = compressor._prune_old_tool_results(messages, protect_tail_count=0)
    assert [result[i]["content"] for i in (1, 3)] == [content, content]
    assert count == 0


def test_pressure_and_lean_tail_keep_resolved_input(compressor):
    content = json.dumps({"user_response": "Do not delete the archive. " * 100})
    messages = _exchange("consent", "clarify", content)
    for i in range(8):
        messages += _exchange(str(i), "terminal", "x" * 3000)
    result, _ = compressor._prune_old_tool_results(messages, protect_tail_count=len(messages), protect_tail_tokens=1)
    assert result[1]["content"] == content
    assert compressor._demote_stale_tail_tools(messages, 0)[1]["content"] == content


@pytest.mark.parametrize("payload", [
    {"error": "internal error " * 100},
    {"user_response": "[user did not respond within 15m]"},
    {"user_response": ""},
    {"user_response": {"internal": "x" * 1000}},
])
def test_unresolved_results_still_demote(compressor, payload):
    payload["question"] = "May I proceed? " * 100
    result, _ = compressor._prune_old_tool_results(_exchange("c", "clarify", json.dumps(payload)), protect_tail_count=0)
    assert result[1]["content"] == "[clarify] asked user a question"
