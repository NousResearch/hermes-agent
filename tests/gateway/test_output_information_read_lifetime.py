"""Informational projections of retained rows, not producer/cleanup execution proof."""
import json
import sqlite3
from types import SimpleNamespace

import pytest

from gateway.session_authority import SessionAuthority
from hermes_state_runtime import RuntimeStoreError
from tests.gateway.test_canonical_hosted_outputs import owner
from tests.gateway.test_canonical_output_lifecycle import connection, dispatch


def retain_obligations(db):
    # Representative durable read records only: never fabricated task settlement
    # or an invocation of cleanup/publication under changed authority.
    def seed(conn):
        conn.execute('INSERT INTO hosted_room_artifact_retries (room_id,task_id,execution_generation,'
                     'member_id,attempts,next_attempt_at,blocked,created_at,updated_at,metadata_json,operation,reason_code) '
                     'VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
            ('room', 'retained-retry', 1, 'writer', 2, 200, 1, 100, 100, '{}', 'publish', 'retained'))
        conn.execute('INSERT INTO state_meta(key,value) VALUES (?,?)',
            ('gateway.hosted.output_cleanup.v1:status-test', json.dumps(dict(room_id='room',
                task_id='retained-cleanup', member_id='writer', execution_generation=1,
                state='pending', reason_code='unlink_pending', attempts=3, next_attempt_at=300))))
    db._execute_write(seed)


@pytest.mark.asyncio
@pytest.mark.parametrize('gate', ['healthy', 'drain', 'same_store_service', 'uninstalled_service'])
async def test_state_preserves_both_informational_obligations(tmp_path, monkeypatch, gate):
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        client = connection(authority, service)
        db = authority.db
        retain_obligations(db)
        replacement = SessionAuthority(runner, profile_id=authority.profile_id,
            instance_id='replacement', db=db, epoch=authority.epoch)
        if gate == 'drain':
            runner._draining = True
            # Status is not a policy eligibility/mutation grant.
            with pytest.raises(RuntimeStoreError, match='runtime_draining'):
                with service._output_policy_read():
                    pass
        elif gate == 'same_store_service':
            service.authority = replacement
        elif gate == 'uninstalled_service':
            authority.hosted_room_service = None
        raw = db._conn
        before, trace, forbidden_calls = raw.total_changes, [], []
        raw.set_trace_callback(trace.append)
        def forbidden(*args, **kwargs):
            forbidden_calls.append(True)
            raise AssertionError('informational status must not open, write, execute or unlink')
        try:
            with monkeypatch.context() as guarded:
                guarded.setattr(sqlite3, 'connect', forbidden)
                guarded.setattr(db, '_execute_write', forbidden)
                guarded.setattr(db, '_open_writer_conn', forbidden)
                guarded.setattr('pathlib.Path.unlink', forbidden)
                guarded.setattr(service, '_output_metadata', forbidden)
                guarded.setattr(service, '_reconcile_stopped_output', forbidden)
                reply = await dispatch(client, 'state', room_id='room')
            assert reply['result']['room']['room_id'] == 'room', reply
            actions = reply['result']['driver_status']['pending_actions']
            retry, = [a for a in actions if a['kind'] == 'output_retry']
            cleanup, = [a for a in actions if a.get('task_id') == 'retained-cleanup']
            assert retry['blocked'] and retry['attempts'] == 2 and retry['reason_code'] == 'retained'
            assert cleanup['state'] == 'pending' and cleanup['reason_code'] == 'unlink_pending'
            assert not any(a['kind'] in {'retry', 'discard', 'approve'} for a in actions)
            assert not forbidden_calls and raw.total_changes == before
            assert all(s.lstrip().split()[0].upper() in {'SELECT', 'BEGIN', 'ROLLBACK'} for s in trace), trace
        finally:
            raw.set_trace_callback(None)
            service.authority = authority
            authority.hosted_room_service = service
            runner._draining = False
            service.runtime._thread = None


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['retry', 'cleanup'])
@pytest.mark.parametrize('gate', ['healthy', 'drain', 'closed', 'replaced', 'quarantine', 'uninstalled', 'epoch'])
async def test_direct_information_read_never_recovers_owner(tmp_path, monkeypatch, kind, gate):
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        db = authority.db
        retain_obligations(db)
        if gate == 'drain':
            runner._draining = True
        elif gate == 'closed':
            db.close()
        elif gate == 'replaced':
            db.db_path.rename(tmp_path / 'detached.db')
        elif gate == 'quarantine':
            monkeypatch.setattr(db, '_db_corrupt', True)
        elif gate == 'uninstalled':
            authority.hosted_room_service = None
        elif gate == 'epoch':
            authority.epoch += 1
        calls = []
        def forbidden(*args, **kwargs):
            calls.append(True)
            raise AssertionError('no writer reopen or side effect')
        reader = service.output_retry_status if kind == 'retry' else service.output_cleanup_status
        with monkeypatch.context() as guarded:
            guarded.setattr(db, '_open_writer_conn', forbidden)
            guarded.setattr(db, '_execute_write', forbidden)
            guarded.setattr(sqlite3, 'connect', forbidden)
            guarded.setattr('pathlib.Path.unlink', forbidden)
            if gate in {'healthy', 'drain'}:
                rows = reader('room')
                assert any(r['task_id'] == 'retained-' + kind for r in rows)
            else:
                with pytest.raises((RuntimeStoreError, sqlite3.Error)):
                    reader('room')
        assert calls == []
