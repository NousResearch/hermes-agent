"""The heartbeat poller coalesces behind an in-flight heartbeat admission wherever it lives: on the
route's logical owner after compression, and as a LOCAL route's local_automation_v1 row."""
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.session_authority import SessionAuthority
from hermes_cli.heartbeat import HeartbeatState, save_heartbeat
from hermes_state import SessionDB
from hermes_state_runtime import admit_session_input, begin_runtime_epoch


@pytest.mark.asyncio
@pytest.mark.parametrize('envelope', [
    {'native_text_v1': {'automation': {'heartbeat': 'root'}}},
    {'local_automation_v1': {'heartbeat': 'root'}},
], ids=['native', 'local'])
async def test_inflight_heartbeat_on_the_logical_owner_blocks_another(tmp_path, monkeypatch, envelope):
    from gateway.run_heartbeat_acceptance import admit_heartbeat
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    db = SessionDB(tmp_path / 'state.db')
    try:
        db.create_session('root', 'telegram')
        db.publish_compression_child(
            parent_session_id='root', child_session_id='tip', source='telegram',
            require_compression_lease=False, model='offline', model_config={},
            system_prompt='offline', messages=[{'role': 'user', 'content': 'retained'}])
        epoch = begin_runtime_epoch(db, instance_id='test')
        admit_session_input(db, epoch=epoch, principal_id='automation:route', session_id='root',
                            request_id='hb-1', payload={'text': 'check', **envelope})
        save_heartbeat('tip', HeartbeatState(prompt='check', interval_seconds=60, created_at=1))
        admitted = []
        authority = SimpleNamespace(db=db)
        authority.logical_owner = lambda sid: SessionAuthority.logical_owner(authority, sid)

        async def admit_automation(adapter, event, identity):
            admitted.append(identity)
        authority.admit_automation = admit_automation
        runner = SimpleNamespace(session_authority=authority)
        source = SessionSource(platform=Platform.TELEGRAM, chat_id='1', user_id='1')
        await admit_heartbeat(runner, object(), source, 'tip', 'route')
        assert admitted == []
    finally:
        db.close()
