"""Startup Bot receipt recovery says what it could not answer, and where.

A receipt the store cannot answer is logged with its file and traceback (the next start reads it
again). A pending receipt whose session row is gone with no deletion marker is settled as
``ambiguous`` like a deleted chat's, but that is store damage, so it is logged, not silent.
"""
import asyncio
from types import SimpleNamespace

from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch
from tools.bot_live_delivery import _locked, _read, _write


def _recover_one(tmp_path, record):
    from gateway.session_bot import recover_bot_deliveries
    db = SessionDB(tmp_path / 'state.db')
    try:
        begin_runtime_epoch(db, instance_id='owner')
        if record['session_id'] == 'live':
            db.create_session('live', source='cli')
        with _locked(tmp_path) as root:
            _write(root / f"{record['delivery_id']}.json", record)
        asyncio.run(recover_bot_deliveries(SimpleNamespace(db=db, waiters={})))
        with _locked(tmp_path) as root:
            return _read(root / f"{record['delivery_id']}.json")
    finally:
        db.close()


def _record(key, session_id):
    return dict(status='queued', delivery_id=key, profile_home='', session_id=session_id,
                admission_id='missing', principal_id='owner', message='ping')


def test_an_unanswerable_receipt_is_logged_with_its_file_and_traceback(tmp_path, caplog):
    key = 'c' * 32
    record = dict(_record(key, 'live'), profile_home=str(tmp_path))
    assert _recover_one(tmp_path, record)['status'] == 'queued'
    line = next(r for r in caplog.records if 'not recovered' in r.getMessage())
    assert f'{key}.json' in line.getMessage() and line.exc_info is not None, line.getMessage()


def test_a_receipt_whose_session_vanished_without_a_marker_is_logged(tmp_path, caplog):
    key = 'd' * 32
    record = dict(_record(key, 'ghost'), profile_home=str(tmp_path))
    assert _recover_one(tmp_path, record)['status'] == 'ambiguous'
    assert any('ghost' in r.getMessage() and 'no deletion marker' in r.getMessage() for r in caplog.records), \
        caplog.text
