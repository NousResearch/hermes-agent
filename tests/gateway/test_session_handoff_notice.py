"""Durable handoff identity survives DB reopen; uncertain admissions are never replayed."""
import select
import subprocess
import sys
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB
from hermes_state_runtime import (RuntimeStoreError, admit_session_input, begin_runtime_epoch,
                                  claim_session_input, recover_session_inputs)
from gateway.session_handoff import ensure_session_handoff


@pytest.fixture
def authority(tmp_path):
    db = SessionDB(tmp_path / 'state.db')
    db.create_session('session', source='local')
    epoch = begin_runtime_epoch(db, instance_id='first')
    owner = SimpleNamespace(db=db, epoch=epoch, profile_id='fixture',
                            logical_owner=lambda sid: 'session', physical_target=lambda ref: 'session')
    yield owner
    db.close()


def test_one_notice_after_reopen_and_duplicate_resume(authority):
    assert ensure_session_handoff(authority, 'session') is None
    path = authority.db.db_path
    authority.db.close()
    authority.db = SessionDB(path)
    authority.epoch = begin_runtime_epoch(authority.db, instance_id='second')
    first = ensure_session_handoff(authority, 'session')
    assert first['from_epoch'] == authority.epoch - 1
    assert first['to_epoch'] == authority.epoch
    assert ensure_session_handoff(authority, 'compressed-alias') == first
    rows = authority.db.get_messages('session')
    notices = [r for r in rows if r.get('display_kind') == 'gateway_handoff']
    assert len(notices) == 1
    assert notices[0]['message_uid'] == first['message_uid']
    assert notices[0]['role'] == 'session_meta'
    model_history, _display_history = authority.db.get_resume_conversations('session')
    assert model_history == []


def test_failed_notice_transaction_does_not_advance_seen(authority, monkeypatch):
    ensure_session_handoff(authority, 'session')
    authority.epoch = begin_runtime_epoch(authority.db, instance_id='second')
    original = authority.db._insert_message_rows
    def fail_after_insert(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError('fixture full disk')
    monkeypatch.setattr(authority.db, '_insert_message_rows', fail_after_insert)
    with pytest.raises(OSError):
        ensure_session_handoff(authority, 'session')
    assert authority.db.get_messages('session') == []
    monkeypatch.setattr(authority.db, '_insert_message_rows', original)
    assert ensure_session_handoff(authority, 'session')['from_epoch'] == authority.epoch - 1


def test_killed_owner_retains_unknown_and_queued_inputs(authority, tmp_path):
    ensure_session_handoff(authority, 'session')
    db_path = authority.db.db_path
    path = str(db_path)
    authority.db.close()
    code = '''
import sys
from pathlib import Path
from hermes_state import SessionDB
from hermes_state_runtime import admit_session_input, claim_session_input
with SessionDB(Path(sys.argv[1])) as db:
    epoch = int(sys.argv[2])
    admit_session_input(db, epoch=epoch, principal_id='human', session_id='session', request_id='sent', payload={'text':'fixture'})
    claim_session_input(db, epoch=epoch, session_id='session')
    admit_session_input(db, epoch=epoch, principal_id='human', session_id='session', request_id='unsent', payload={'text':'next'})
    print('CLAIMED', flush=True)
    sys.stdin.read()
'''
    child = subprocess.Popen([sys.executable, '-c', code, path, str(authority.epoch)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert select.select([child.stdout], [], [], 10)[0], 'fixture never claimed'
        line = child.stdout.readline().strip()
        assert line == 'CLAIMED', (
            f'fixture died before claim: rc={child.poll()} '
            f'stderr={child.stderr.read()!r}')
        child.kill()
        child.wait(timeout=10)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
        child.stdin.close()
        child.stdout.close()
        child.stderr.close()
    authority.db = SessionDB(db_path)
    old = authority.epoch
    authority.epoch = begin_runtime_epoch(authority.db, instance_id='successor')
    recover_session_inputs(authority.db, epoch=authority.epoch)
    notice = ensure_session_handoff(authority, 'session')
    assert notice['recovery_summary'] == {'queued': 1, 'unknown': 1}
    with pytest.raises(RuntimeStoreError, match='unknown_execution'):
        claim_session_input(authority.db, epoch=authority.epoch, session_id='session')
    with pytest.raises(RuntimeStoreError, match='stale_epoch'):
        claim_session_input(authority.db, epoch=old, session_id='session')


def test_retirement_removes_notice_metadata(authority):
    ensure_session_handoff(authority, 'session')
    authority.epoch = begin_runtime_epoch(authority.db, instance_id='second')
    ensure_session_handoff(authority, 'session')
    authority.db.delete_session('session')
    with authority.db._read_ctx() as conn:
        assert conn.execute("SELECT key FROM state_meta WHERE key LIKE 'session_handoff%'").fetchall() == []
