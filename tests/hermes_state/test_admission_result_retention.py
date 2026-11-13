"""A settled turn's stored result never outlives its session's retention.

The exact-retry horizon for stored results was the fixed 7-day ``RETRY_HORIZON_S``. With
``sessions.retention_days: 3`` a reporter's result rows lived at least twice as long as their
sessions. The retention prune now caps the result cache at that window (floor: one day, ceiling:
the 7-day horizon). The deletion tombstones and their fence are unchanged.
"""
import time
from contextlib import closing

from hermes_state import SessionDB
from hermes_state_runtime import admit_session_input, begin_runtime_epoch, claim_session_input
from gateway.session_results import admission_result, retain_result

DAY = 86400


def _age(db, session_id, at):
    db._conn.execute("UPDATE sessions SET started_at=?, last_activity_at=? WHERE id=?", (at, at, session_id))
    db._conn.execute("UPDATE messages SET timestamp=? WHERE session_id=?", (at, session_id))
    db._conn.commit()


def _settled(db, epoch, session_id, request_id='turn'):
    admit_session_input(db, epoch=epoch, principal_id='api', session_id=session_id,
                        request_id=request_id, payload={'text': request_id})
    row = claim_session_input(db, epoch=epoch, session_id=session_id)
    retain_result(db, epoch=epoch, row=row, result={'result': {'final_response': 'reply', 'messages': []}, 'usage': {}})
    return row['admission_id']


def test_a_result_does_not_outlive_the_retention_window(tmp_path):
    with closing(SessionDB(tmp_path / 'state.db')) as db:
        epoch = begin_runtime_epoch(db, instance_id='owner')
        for sid in ('stale', 'fresh'):
            db.create_session(sid, source='api_server')  # open: prune never deletes it
            db.append_message(sid, 'user', 'hello')
        stale, fresh = _settled(db, epoch, 'stale'), _settled(db, epoch, 'fresh')
        _age(db, 'stale', time.time() - 4 * DAY)
        _age(db, 'fresh', time.time() - 2 * DAY)

        db.maybe_auto_prune_and_vacuum(retention_days=3, vacuum=False)

        assert admission_result(db, stale) is None, 'a result outlived sessions.retention_days'
        assert admission_result(db, fresh)['result']['final_response'] == 'reply'
        assert db.get_session('stale') is not None  # the session itself follows its own rules


def test_a_compressed_conversation_keeps_its_results_while_its_tip_is_active(tmp_path):
    """Turns of a compressed conversation admit on its root and land on the tip their claim
    recorded; a quiet root with an active tip is a conversation still in use."""
    with closing(SessionDB(tmp_path / 'state.db')) as db:
        epoch = begin_runtime_epoch(db, instance_id='owner')
        db.create_session('root', source='api_server')
        db.create_session('tip', source='api_server')
        admission = _settled(db, epoch, 'root')
        db._conn.execute("UPDATE session_admissions SET lineage_json='[\"root\",\"tip\"]' WHERE admission_id=?",
                         (admission,))
        db._conn.commit()
        _age(db, 'root', time.time() - 8 * DAY)
        db.append_message('tip', 'user', 'still talking')

        db.maybe_auto_prune_and_vacuum(retention_days=3, vacuum=False)

        assert admission_result(db, admission)['result']['final_response'] == 'reply'


def test_a_filtered_prune_does_not_cap_other_sessions_results(tmp_path):
    """``hermes sessions prune --source cli`` says nothing about an API session's retention."""
    with closing(SessionDB(tmp_path / 'state.db')) as db:
        epoch = begin_runtime_epoch(db, instance_id='owner')
        db.create_session('api', source='api_server')
        db.append_message('api', 'user', 'hello')
        admission = _settled(db, epoch, 'api')
        _age(db, 'api', time.time() - 4 * DAY)

        db.prune_sessions(older_than_days=3, source='cli')

        assert admission_result(db, admission)['result']['final_response'] == 'reply'


def test_the_horizon_follows_retention_between_a_day_and_the_retry_horizon():
    from hermes_state_mutation_retirement import RETRY_HORIZON_S
    from hermes_state_terminal import result_horizon_s
    assert result_horizon_s(3) == 3 * DAY
    assert result_horizon_s(0) == DAY
    assert result_horizon_s(90) == RETRY_HORIZON_S
