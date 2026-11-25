"""The heartbeat poller coalesces behind an in-flight heartbeat admission wherever it lives: on the
route's logical owner after compression, and as a LOCAL route's local_automation_v1 row."""
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.session_authority import SessionAuthority
from hermes_cli.heartbeat import HeartbeatState, save_heartbeat
from persistence.conversation.lineage import CompressionChild
from persistence.conversation.messages import NewMessage
from tests.fixtures.storage_seed import seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


@pytest.mark.asyncio
@pytest.mark.parametrize('envelope', [
    {'native_text_v1': {'automation': {'heartbeat': 'root'}}},
    {'local_automation_v1': {'heartbeat': 'root'}},
], ids=['native', 'local'])
async def test_inflight_heartbeat_on_the_logical_owner_blocks_another(tmp_path, monkeypatch, envelope):
    from gateway.run_heartbeat_acceptance import admit_heartbeat
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    store = storage_handle(tmp_path)
    seed_session(store, 'root', 'telegram')
    store.lineage.publish_compression_child(
        'root', CompressionChild(session_id='tip', source='telegram', model='offline', model_config={},
                                 system_prompt='offline'),
        [NewMessage.from_live({'role': 'user', 'content': 'retained'})], require_compression_lease=False)
    epoch = store.runtime.begin_epoch(instance_id='test')
    store.admissions.admit(epoch=epoch, principal_id='automation:route', session_id='root',
                           request_id='hb-1', payload={'text': 'check', **envelope})
    save_heartbeat('tip', HeartbeatState(prompt='check', interval_seconds=60, created_at=1))
    admitted = []
    authority = SimpleNamespace(persistence=store)
    authority.logical_owner = lambda sid: SessionAuthority.logical_owner(authority, sid)

    async def admit_automation(adapter, event, identity):
        admitted.append(identity)
    authority.admit_automation = admit_automation
    runner = SimpleNamespace(session_authority=authority)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='1', user_id='1')
    await admit_heartbeat(runner, object(), source, 'tip', 'route')
    assert admitted == []
