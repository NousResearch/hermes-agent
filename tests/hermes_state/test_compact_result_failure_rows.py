"""Regression for #147057: a context-overflow failure turn must not store the full session
history. ``compact_result`` runs its anchor trim on failure rows too, and the tail an overflow
turn settles on is the open machinery row its exemption from durable-turn closure left behind —
no user-originated anchor survives, the anchor resolves to -1, ``messages[anchor + 1:]`` keeps
every row and ``_messages_are_turn_suffix`` is still stamped (flag true, semantics false). A
context-pressure failure row drops ``messages`` instead; its storable output is the reply/error
copy. A healthy completed row keeps the exact trim behaviour."""
import pytest

from hermes_state_terminal import compact_result


def _overflow_result(**markers):
    history = [
        {'role': 'user', 'content': 'INTERNAL NOTIFICATION\nwake', 'display_kind': 'internal_notification'},
        {'role': 'assistant', 'content': 'ack'},
        {'role': 'user', 'content': 'INTERNAL NOTIFICATION\nwake\nfooter\nask + footer\nfooter',
         'display_kind': 'internal_notification'},
    ]
    value = {'final_response': 'context window exhausted', 'messages': history, 'completed': False,
             'failed': True, 'partial': True, 'error': 'context window exhausted', 'api_calls': 3,
             'failure_retryable': False}
    value.update(markers)
    return {'result': value, 'usage': {'input_tokens': 3}}


@pytest.mark.parametrize('markers', [
    {'failure_reason': 'context_overflow', 'compression_exhausted': True},
    {'failure_reason': 'compression_exhausted'},
    {'compression_exhausted': True},
], ids=['overflow_reason', 'exhausted_reason', 'exhausted_flag'])
def test_overflow_failure_row_stores_no_full_history(markers):
    stored = compact_result(_overflow_result(**markers), user_message='ask')
    value = stored['result']
    assert 'messages' not in value, 'a failed overflow turn must not store the session transcript'
    assert not value.get('_messages_are_turn_suffix')
    assert value['final_response'] == 'context window exhausted'
    assert value['failed'] is True and value['completed'] is False
    assert stored['usage'] == {'input_tokens': 3}


def test_completed_row_trim_unchanged():
    messages = [
        {'role': 'user', 'content': 'old ask'},
        {'role': 'assistant', 'content': 'old answer'},
        {'role': 'user', 'content': 'fresh ask'},
        {'role': 'assistant', 'content': 'fresh answer'},
    ]
    stored = compact_result({'result': {'final_response': 'fresh answer', 'messages': messages,
                                        'completed': True}}, user_message='fresh ask')
    assert stored['result']['messages'] == [{'role': 'assistant', 'content': 'fresh answer'}]
    assert stored['result']['_messages_are_turn_suffix'] is True
