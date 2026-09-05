"""Pending writes must be surfaced without claiming that they were applied."""
import json
import pytest
from agent.background_review import summarize_background_review_actions

@pytest.mark.parametrize('message', ['Staged for approval: skill proposal', 'Staged for approval: updated skill proposal'])
def test_staged_proposal_summary_has_a_working_upstream_review_command(message):
    messages = [
        {'role': 'assistant', 'tool_calls': [{'id': 'pending', 'type': 'function', 'function': {'name': 'skill_manage', 'arguments': json.dumps({'action': 'patch', 'name': 'example'})}}]},
        {'role': 'tool', 'tool_call_id': 'pending', 'content': json.dumps({'success': True, 'message': message})},
    ]
    actions = summarize_background_review_actions(messages, [])
    assert len(actions) == 1
    assert 'staged for approval' in actions[0].lower()
    assert '/skills pending' in actions[0]
    assert 'desktop Review tab' not in actions[0]
    assert summarize_background_review_actions(messages, [], 'off') == []
    assert summarize_background_review_actions(messages, messages) == []
