"""Bot receipts recover only their exact committed admission result."""
from types import SimpleNamespace

from hermes_state import SessionDB
from hermes_state_runtime import (
    admit_session_input, begin_runtime_epoch, claim_session_input,
    recover_session_inputs,
)
from gateway.session_results import retain_result
from gateway.session_bot import _result


def test_bot_reply_recovery_uses_exact_terminal_admission(tmp_path):
    path = tmp_path / 'state.db'
    db = SessionDB(path)
    db.create_session('bot', source='gui')
    epoch = begin_runtime_epoch(db, instance_id='first')
    records = []
    for key, reply, settle in [('one', 'exact first reply', True),
                               ('two', 'later reply is not first', True),
                               ('three', 'uncommitted reply', False)]:
        admission = admit_session_input(db, epoch=epoch, principal_id='owner', session_id='bot',
                                       request_id=key, payload={'text': key})
        row = claim_session_input(db, epoch=epoch, session_id='bot')
        # retain_result IS the settlement: the result commits with terminal status or not at all.
        if settle:
            retain_result(db, epoch=epoch, row=row, result={'result': {'final_response': reply}, 'usage': {}})
        records.append(dict(status='canonical', admission_id=admission['admission_id'],
                            delivery_id=key, profile_home=str(tmp_path), session_id='bot', message=key))
    db.close()
    db = SessionDB(path)
    try:
        epoch = begin_runtime_epoch(db, instance_id='restart')
        recover_session_inputs(db, epoch=epoch)
        authority = SimpleNamespace(db=db)
        assert [_result(authority, record)['message'] for record in records] == ['one', 'two', 'three']
        assert _result(authority, records[0])['reply'] == 'exact first reply'
        assert _result(authority, records[0])['status'] == 'settled'
        assert _result(authority, records[1])['reply'] == 'later reply is not first'
        assert _result(authority, records[2])['status'] == 'ambiguous'
        assert _result(authority, records[2])['reply'] == ''
    finally:
        db.close()
