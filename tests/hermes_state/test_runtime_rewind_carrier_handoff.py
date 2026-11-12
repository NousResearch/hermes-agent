"""An owner ``rewind`` of a composite compaction carrier keeps its handoff (main parity)."""
from agent.context_compressor import HISTORICAL_TASK_HEADING, SUMMARY_PREFIX, _SUMMARY_END_MARKER
from hermes_state import SessionDB
import hermes_state_runtime as rt


def _rewind(db, target, request_id, **extra):
    epoch = rt.begin_runtime_epoch(db, instance_id='owner')
    session = db.get_session('s')
    return rt.mutate_runtime_session(db, epoch=epoch, principal_id='human', session_id='s', request_id=request_id,
        expected_revision=session.get('runtime_revision') or 0, expected_generation=session['runtime_generation'],
        operation='rewind', payload={'target_message_id': target, **extra})


def test_rewind_of_carrier_keeps_handoff_and_reports_the_live_ask(tmp_path):
    carrier = f"{SUMMARY_PREFIX}\n{HISTORICAL_TASK_HEADING}\nold task\n\n{_SUMMARY_END_MARKER}\n\nREAL ASK"
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('s', source='test')
        db.append_message('s', 'user', 'older ask')
        db.append_message('s', 'assistant', 'older reply')
        target = db.append_message('s', 'user', carrier)
        db.append_message('s', 'assistant', 'reply')
        receipt = _rewind(db, target, 'rewind')
        assert receipt['rewound_count'] == 2
        assert receipt['target_message']['content'] == 'REAL ASK'
        active = db.get_messages_as_conversation('s')
        assert [m['role'] for m in active] == ['user', 'assistant', 'user']
        assert SUMMARY_PREFIX in active[-1]['content'] and 'REAL ASK' not in active[-1]['content']
        assert db.get_session('s')['runtime_generation'] == 1
        # A rewind to before the compacted span still drops the summary.
        _rewind(db, db.get_messages('s')[0]['id'], 'rewind-all')
        assert db.get_messages_as_conversation('s') == []
