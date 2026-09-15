"""Responses replay must not invent a blank assistant turn before a tool call.

Regression investigation for #103483: synthetic Muse tool-loop A/B reproduction.
Also covers #75202: the lone-reasoning follower must be non-empty for strict
Responses-compatible providers that reject "" with 400.
"""
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
