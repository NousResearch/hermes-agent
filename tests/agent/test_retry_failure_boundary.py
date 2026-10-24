"""Retry can omit only its authenticated, durable Hermes failure boundary (#137277)."""
from copy import deepcopy

import pytest

from agent.turn_failure_copy import FAILED_TURN_DISPLAY_KIND, failed_turn_notice
from agent.retry_context import retry_failure_row_ids
from tui_gateway.methods_prompt_retry import _retry_context_is_durable, _retry_history_signature


def _history():
    rows = [
        {'role': 'user', 'content': 'count once', '_row_id': 1},
        {'role': 'assistant', 'tool_calls': [{'id': 'call-1'}], '_row_id': 2},
        {'role': 'tool', 'content': 'done', 'tool_call_id': 'call-1', '_row_id': 3,
         'display_metadata': {'tool_result_received': True, 'data_use': ['no_training']},
         'data_use': ['no_training']},
    ]
    return [*rows, {'role': 'assistant', 'content': failed_turn_notice(rows), '_row_id': 4,
                   'display_kind': FAILED_TURN_DISPLAY_KIND,
                   'display_metadata': {'error_surface': {'layer': 'provider', 'code': 'rate_limit'}}}]


def test_only_the_typed_rate_limit_boundary_is_eligible_for_wire_omission():
    rows = _history()
    before = deepcopy(rows)
    assert retry_failure_row_ids(rows) == (4,)
    assert _retry_context_is_durable(rows)
    assert rows == before  # projection never deletes the durable display row
    ordinary = [*rows, {'role': 'user', 'content': 'new intent', '_row_id': 5}]
    assert retry_failure_row_ids(ordinary) == ()


@pytest.mark.parametrize(('key', 'value'), [
    ('content', 'A model reply with similar text'),
    ('display_metadata', {'error_surface': 'malformed'}), ('_row_id', None),
    ('tool_calls', [{'id': 'new-call'}]),
    ('display_metadata', {'error_surface': {'layer': 'gateway', 'code': 'rate_limit'}}),
])
def test_unproven_typed_boundaries_fail_closed(key, value):
    rows = _history()
    rows[-1][key] = value
    assert retry_failure_row_ids(rows) is None
    assert not _retry_context_is_durable(rows)


@pytest.mark.parametrize('change', ['kind', 'data-use'])
def test_boundary_identity_and_data_use_changes_invalidate_the_checkpoint(change):
    rows = _history()
    signature = _retry_history_signature(rows)
    if change == 'kind':
        rows[-1]['display_kind'] = 'ordinary'
    else:
        rows[2]['data_use'] = ['other-policy']
    assert _retry_history_signature(rows) != signature


def test_repeated_failures_omit_all_proven_notices_in_this_turn_only():
    rows = _history()
    rows.extend([{'role': 'assistant', 'tool_calls': [{'id': 'call-2'}], '_row_id': 5},
                 {'role': 'tool', 'content': 'done again', 'tool_call_id': 'call-2', '_row_id': 6}])
    rows.append({**deepcopy(rows[3]), '_row_id': 7, 'content': failed_turn_notice(rows)})
    assert retry_failure_row_ids(rows) == (4, 7)
    rows[3]['display_metadata']['error_surface']['code'] = 'unknown'
    assert retry_failure_row_ids(rows) is None


def test_proven_notices_survive_repair_before_a_new_tool_result():
    from types import SimpleNamespace
    from agent.agent_runtime_helpers import repair_message_sequence_with_cursor
    rows = _history()
    boundary = rows[-1]
    rows.extend([{'role': 'assistant', 'tool_calls': [{'id': 'call-2', 'type': 'function',
                  'function': {'name': 'terminal', 'arguments': '{}'}}], '_row_id': 5},
                 {'role': 'tool', 'content': 'done again', 'tool_call_id': 'call-2', '_row_id': 6}])
    before = deepcopy(rows)
    agent = SimpleNamespace(_resume_user_row_id=1, _resume_failure_row_ids=(4,))
    assert repair_message_sequence_with_cursor(agent, rows) == 0
    assert rows == before
    assert rows[3] is boundary


def test_retry_still_repairs_new_malformed_input_without_mutating_the_boundary():
    from types import SimpleNamespace
    from agent.agent_runtime_helpers import repair_message_sequence_with_cursor
    rows = _history()
    boundary = rows[-1]
    rows.extend([{'role': 'user', 'content': 'one'}, {'role': 'user', 'content': 'two'}])
    agent = SimpleNamespace(_resume_user_row_id=1, _resume_failure_row_ids=(4,), _last_flushed_db_idx=4)
    assert repair_message_sequence_with_cursor(agent, rows) > 0
    assert rows[3] is boundary
    assert len(rows) == 5
    assert 'one' in rows[-1]['content'] and 'two' in rows[-1]['content']
    assert agent._last_flushed_db_idx == 4
