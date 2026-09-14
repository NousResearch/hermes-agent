"""Structured results are claim-owned and visible only after settlement."""
import pytest

from hermes_state import SessionDB
from hermes_state_runtime import (
    RuntimeStoreError, admit_session_input, begin_runtime_epoch,
    claim_session_input, recover_session_inputs,
)


def test_result_survives_restart_without_reexecuting(tmp_path):
    from gateway.session_results import retain_result, admission_result
    path = tmp_path / 'state.db'
    db = SessionDB(path)
    db.create_session('api-session', source='api_server')
    epoch = begin_runtime_epoch(db, instance_id='first')
    admitted = admit_session_input(db, epoch=epoch, principal_id='api', session_id='api-session',
                                   request_id='retry', payload={'text': 'hello'})
    row = claim_session_input(db, epoch=epoch, session_id='api-session')
    result = {'final_response': 'reply', 'messages': [], 'usage': {'input_tokens': 7, 'output_tokens': 3}}
    retain_result(db, epoch=epoch, row=row, result=result)
    assert admission_result(db, admitted['admission_id']) == result
    db.close()
    db = SessionDB(path)
    try:
        epoch = begin_runtime_epoch(db, instance_id='second')
        recover_session_inputs(db, epoch=epoch)
        retried = admit_session_input(db, epoch=epoch, principal_id='api', session_id='api-session',
                                      request_id='retry', payload={'text': 'hello'})
        assert retried['admission_id'] == admitted['admission_id']
        assert admission_result(db, retried['admission_id']) == result
        assert claim_session_input(db, epoch=epoch, session_id='api-session') is None
    finally:
        db.close()


def test_stale_result_cannot_overwrite_settled_or_unknown_claim(tmp_path):
    from gateway.session_results import retain_result, admission_result
    db = SessionDB(tmp_path / 'state.db')
    try:
        db.create_session('api-session', source='api_server')
        epoch = begin_runtime_epoch(db, instance_id='first')
        admit_session_input(db, epoch=epoch, principal_id='api', session_id='api-session',
                            request_id='one', payload={'text': 'hello'})
        row = claim_session_input(db, epoch=epoch, session_id='api-session')
        retain_result(db, epoch=epoch, row=row, result={'final_response': 'first'})
        with pytest.raises(RuntimeStoreError, match='stale_generation'):
            retain_result(db, epoch=epoch, row=row, result={'final_response': 'late'})
        assert admission_result(db, row['admission_id']) == {'final_response': 'first'}
        newer = begin_runtime_epoch(db, instance_id='second')
        with pytest.raises(RuntimeStoreError, match='stale_epoch'):
            retain_result(db, epoch=epoch, row=row, result={})
        assert newer != epoch
    finally:
        db.close()
