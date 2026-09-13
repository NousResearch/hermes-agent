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


def test_cancelled_queued_admission_is_a_terminal_cancelled_receipt_not_ambiguous(tmp_path):
    """A queued Bot admission cancelled before execution definitively did not run: after an
    owner restart, recovery and the exact delivery retry both report ``cancelled`` (terminal),
    never ``ambiguous``; a rejected admission without a result blob is ``failed``."""
    import asyncio
    from hermes_state_runtime import cancel_session_input, settle_session_input
    from tools.bot_live_delivery import _locked, _read, _write
    from gateway.session_bot import recover_bot_deliveries
    path = tmp_path / 'state.db'
    db = SessionDB(path)
    db.create_session('bot', source='gui')
    epoch = begin_runtime_epoch(db, instance_id='first')
    running = admit_session_input(db, epoch=epoch, principal_id='owner', session_id='bot',
                                  request_id='bot:' + 'a' * 32, payload={'text': 'running'})
    queued = admit_session_input(db, epoch=epoch, principal_id='owner', session_id='bot',
                                 request_id='bot:' + 'b' * 32, payload={'text': 'queued'})
    row = claim_session_input(db, epoch=epoch, session_id='bot')
    assert row['admission_id'] == running['admission_id']
    cancel_session_input(db, epoch=epoch, admission_id=queued['admission_id'])
    settle_session_input(db, epoch=epoch, admission_id=row['admission_id'], generation=row['generation'], outcome='rejected')
    records = {}
    with _locked(tmp_path) as root:
        for key, admission in (('a' * 32, running), ('b' * 32, queued)):
            records[key] = dict(status='canonical', admission_id=admission['admission_id'], delivery_id=key,
                                profile_home=str(tmp_path), session_id='bot', message=key, principal_id='owner')
            _write(root / f'{key}.json', records[key])
    db.close()
    db = SessionDB(path)
    try:
        epoch = begin_runtime_epoch(db, instance_id='restart')
        recover_session_inputs(db, epoch=epoch)
        authority = SimpleNamespace(db=db, waiters={})
        asyncio.run(recover_bot_deliveries(authority))
        with _locked(tmp_path) as root:
            recovered = {key: _read(root / f'{key}.json')['status'] for key in records}
        assert recovered == {'a' * 32: 'failed', 'b' * 32: 'cancelled'}
        # The exact retry reads the same classification from the durable admission row.
        assert _result(authority, records['b' * 32])['status'] == 'cancelled'
        assert _result(authority, records['a' * 32])['status'] == 'failed'
    finally:
        db.close()
