"""An owner ``rewind`` of a composite compaction carrier keeps its handoff."""
from agent.context_compressor import HISTORICAL_TASK_HEADING, SUMMARY_PREFIX, _SUMMARY_END_MARKER
from persistence.sqlite.authority._mutation import mutate_runtime_session
from persistence.sqlite.authority._runtime_ends import begin_runtime_epoch
from persistence.sqlite.conversation._store.messages import append_message, get_messages, get_messages_as_conversation
from persistence.sqlite.conversation._store.sessions import create_session, get_session


def _rewind(holder, target, request_id, **extra):
    epoch = begin_runtime_epoch(holder, instance_id='owner')
    session = get_session(holder, 's')
    return mutate_runtime_session(holder, epoch=epoch, principal_id='human', session_id='s', request_id=request_id,
        expected_revision=session.get('runtime_revision') or 0, expected_generation=session['runtime_generation'],
        operation='rewind', payload={'target_message_id': target, **extra})[0]


def test_rewind_of_carrier_keeps_handoff_and_reports_the_live_ask(store):
    carrier = f"{SUMMARY_PREFIX}\n{HISTORICAL_TASK_HEADING}\nold task\n\n{_SUMMARY_END_MARKER}\n\nREAL ASK"
    holder = store.holder()
    create_session(holder, 's', 'test')
    append_message(holder, 's', 'user', 'older ask')
    append_message(holder, 's', 'assistant', 'older reply')
    target = append_message(holder, 's', 'user', carrier)
    append_message(holder, 's', 'assistant', 'reply')
    receipt = _rewind(holder, target, 'rewind')
    assert receipt['rewound_count'] == 2
    assert receipt['target_message']['content'] == 'REAL ASK'
    active = get_messages_as_conversation(holder, 's')
    assert [m['role'] for m in active] == ['user', 'assistant', 'user']
    assert SUMMARY_PREFIX in active[-1]['content'] and 'REAL ASK' not in active[-1]['content']
    assert get_session(holder, 's')['runtime_generation'] == 1
    # A rewind to before the compacted span still drops the summary.
    _rewind(holder, get_messages(holder, 's')[0]['id'], 'rewind-all')
    assert get_messages_as_conversation(holder, 's') == []
