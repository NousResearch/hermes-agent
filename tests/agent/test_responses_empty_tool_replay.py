"""Responses replay must not invent a blank assistant turn before a tool call.

Regression investigation for #103483: synthetic Muse tool-loop A/B reproduction.
Also covers #75202: the lone-reasoning follower must be non-empty for strict
Responses-compatible providers that reject "" with 400.
"""
import json
from pathlib import Path

import pytest
from agent.codex_responses_adapter import _chat_messages_to_responses_input

@pytest.mark.parametrize('content', ['', None])
def test_reasoning_tool_round_has_no_invented_assistant_message(content):
    messages = [{
        'role': 'assistant', 'content': content,
        'codex_reasoning_items': [{'type': 'reasoning', 'id': 'rs_test',
                                  'encrypted_content': 'synthetic-encrypted-fixture', 'summary': []}],
        'tool_calls': [{'id': 'call_test', 'type': 'function',
                        'function': {'name': 'record_step', 'arguments': '{"step":1}'}}],
    }, {'role': 'tool', 'tool_call_id': 'call_test', 'content': '{"ok":true}'}]
    items = _chat_messages_to_responses_input(messages)
    assert [item.get('type', item.get('role')) for item in items] == [
        'reasoning', 'function_call', 'function_call_output']
    assert items[1]['call_id'] == items[2]['call_id']


def test_reasoning_without_tool_keeps_nonempty_following_item():
    items = _chat_messages_to_responses_input([{
        'role': 'assistant', 'content': '',
        'codex_reasoning_items': [{'type': 'reasoning', 'id': 'rs_test',
                                  'encrypted_content': 'synthetic-encrypted-fixture', 'summary': []}],
    }])
    assert items[-1] == {'role': 'assistant', 'content': ' '}


def test_failing_turn_fixture_emits_no_invented_carrier():
    """Real failing-turn shape from #103483 (fixture by @cristianbdev, see
    https://gist.github.com/cristianbdev/6036f7aa3838935adfaeb9b6800f7450 —
    3 reasoning+tool assistant rows, 5 tool calls/outputs, content redacted).

    The adapter must invent no blank carrier, must keep every reasoning item
    directly followed by a function_call, and must preserve all call/output
    pairings.
    """
    fixture_path = (
        Path(__file__).resolve().parent.parent
        / "fixtures" / "spark_failing_turn_shape.json"
    )
    messages = json.loads(fixture_path.read_text(encoding="utf-8"))
    items = _chat_messages_to_responses_input(messages)
    assert {"role": "assistant", "content": ""} not in items
    for i, item in enumerate(items):
        if item.get("type") == "reasoning":
            # Reasoning items may cluster; the next non-reasoning item after
            # each of them must be the function_call that follows it — never
            # an invented blank assistant message.
            nxt = next(it for it in items[i + 1:] if it.get("type") != "reasoning")
            assert nxt.get("type") == "function_call"
    calls = [item["call_id"] for item in items if item.get("type") == "function_call"]
    outputs = [item["call_id"] for item in items if item.get("type") == "function_call_output"]
    assert len(calls) == 5
    assert sorted(calls) == sorted(outputs)
