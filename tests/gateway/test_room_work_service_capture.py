"""Canonical hosted producer reaches passive capture without an explicit publisher."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch


@pytest.mark.asyncio
async def test_canonical_send_and_stop_retain_driver_evidence(tmp_path, monkeypatch):
    from gateway.session_hosted_service import CanonicalHostedRoomService
    from gateway import hosted_room_work_records as work
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    with SessionDB(tmp_path / 'state.db') as db:
        authority = SimpleNamespace(db=db, profile_id=str(tmp_path), epoch=begin_runtime_epoch(db, instance_id='test'))
        service = CanonicalHostedRoomService(authority, asyncio.get_running_loop())
        service.authorize_room('alice', 'room', create=True)
        service.create_room(room_id='room', name='Room', members=[
            {'member_id': 'one', 'profile': 'default', 'handle': 'one'},
            {'member_id': 'two', 'profile': 'other', 'handle': 'two',
             'target': {'kind': 'peer', 'installation_id': 'peer', 'profile': 'other',
                        'peer_id': 'peer', 'capability_digest': 'a' * 64}}])
        service.send(room_id='room', event_id='input', payload={'text': '@one hello', 'thread_id': 'thread'})
        def evidence():
            with db._read_ctx() as conn:
                row = conn.execute(f'SELECT record_json FROM {work.SOURCE_TABLE}').fetchone()
                assert row is not None
                return work.validate(json.loads(row[0]))
        before = evidence()
        assert before['availability'] == 'available'
        assert before['tasks'][0]['phase'] == 'queued'
        service.stop_room('room', cancel_id='stop')
        after = evidence()
        assert after['revision'] > before['revision']
        assert after['tasks'][0]['phase'] == 'cancelled'
        assert after['stop']['cancel_id'] == 'stop'
        assert after['stop']['revocation_complete'] is False
