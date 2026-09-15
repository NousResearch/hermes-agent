"""Copilot rotates opaque item IDs within a call; output_index remains stable.

Reproduced with a live Responses request: one announced call plus its done
item became two executable calls, with the stale announcement carrying {}.
"""
import json

import pytest

from agent.codex_runtime import _consume_codex_event_stream


@pytest.mark.parametrize(('completion', 'done_order'), [
    ('item_done', (0, 1)),
    pytest.param('item_done', (1, 0), id='item_done_reversed'),
    ('arguments_done', (0, 1)),
    ('deltas', (0, 1)),
])
def test_rotating_item_ids_preserve_distinct_calls_and_arguments(completion, done_order):
    events = []
    for index in range(2):
        events.append({
            'type': 'response.output_item.added', 'output_index': index,
            'item': {'type': 'function_call', 'id': f'announced_{index}',
                     'call_id': f'call_{index}', 'name': 'diagnostic_echo', 'arguments': ''},
        })
    # Interleave argument chunks so associating by "latest call" cannot pass.
    for chunk in range(2):
        for index in range(2):
            events.append({
                'type': 'response.function_call_arguments.delta', 'output_index': index,
                'item_id': f'delta_{index}_{chunk}',
                'delta': ['{"text":', f'"probe-{index}"}}'][chunk],
            })
    if completion != 'deltas':
        for index in range(2):
            events.append({
                'type': 'response.function_call_arguments.done', 'output_index': index,
                'item_id': f'arguments_done_{index}',
                'arguments': json.dumps({'text': f'confirmed-{index}'}),
            })
    if completion == 'item_done':
        # Completion order must not reorder the announced calls, even when
        # rotated-ID done events confirm every call and leave none pending.
        for index in done_order:
            events.append({
                'type': 'response.output_item.done', 'output_index': index,
                'item': {'type': 'function_call', 'id': f'completed_{index}',
                         'call_id': f'call_{index}', 'name': 'diagnostic_echo',
                         'arguments': json.dumps({'text': f'authoritative-{index}'})},
            })
    events.append({'type': 'response.completed', 'response': {'status': 'completed'}})
    response = _consume_codex_event_stream(events, model='diagnostic')
    def field(item, name):
        return item[name] if isinstance(item, dict) else getattr(item, name)
    expected = {'item_done': 'authoritative', 'arguments_done': 'confirmed', 'deltas': 'probe'}[completion]
    assert [(field(item, 'call_id'), json.loads(field(item, 'arguments')))
            for item in response.output] == [
                (f'call_{i}', {'text': f'{expected}-{i}'}) for i in range(2)
            ]
