"""A real lender's displayed approval is answerable only while the home attempt is live."""
import asyncio
import time

import pytest

from gateway import hosted_room_driver as driver
from gateway.hosted_room_peer import GatewayRoomCatalog, catalog_mapping
from tests.gateway.test_peer_client_approvals import lender as lender_fixture
from tests.tui_gateway.test_hosted_room_service import _server
from tui_gateway.hosted_room_peer_transport import PeerMemberRoute
from tui_gateway.hosted_room_service import HostedRoomService

lender = lender_fixture


@pytest.mark.asyncio
@pytest.mark.parametrize('stopped', [False, 'stop', 'runtime'])
async def test_lent_approval_lifetime_through_real_http(lender, tmp_path, monkeypatch, stopped):
    catalog = GatewayRoomCatalog.from_mapping(catalog_mapping(
        target_profile='default', installation_id='lender', persistent_process=True))
    home = HostedRoomService(_server(), db_path=tmp_path / 'home.db')
    home.local_profiles = lambda: ('default',)
    home.register_peer_route(room_id='room-1', member_id='member-peer',
        route=PeerMemberRoute(home_install_id='home', member_id='member-peer',
            target_install_id='lender', target_profile='default', capability_digest=catalog.catalog_digest,
            cancellation_scope_id='cancel', trace_id='trace', grant=lender.grant),
        client=lender.client, target_url=lender.client.base_url, catalog=catalog)
    home.create_room(room_id='room-1', name='Peer approval lifetime', members=[
        {'member_id': 'default', 'profile': 'default', 'handle': 'home'},
        {'member_id': 'member-peer', 'profile': 'default', 'handle': 'reviewer', 'target': {
            'kind': 'peer', 'peer_id': 'peer', 'installation_id': 'lender',
            'profile': 'default', 'capability_digest': catalog.catalog_digest}}])
    home.send(room_id='room-1', event_id='ask', payload={'text': '@reviewer run', 'thread_id': 'ask'})
    task, = driver.list_tasks(home.db_path, room_id='room-1', status='queued')
    binding, = home.bindings()
    lease = driver.acquire_lease(home.db_path, room_id='room-1', gateway_id=binding.gateway_id,
        authority_epoch=binding.authority_epoch, process_generation='worker', ttl_seconds=30, clock=time.time)
    driver.start_task(home.db_path, task['identity'], lease, expected_cancel_generation=0, clock=time.time)
    task = driver.get_task(home.db_path, task['identity'])
    task_id, generation = task['identity'].task_id, task['execution_generation']
    receipt = lender.client._runs[('task', 7)]
    transport = home._resolve_member_transport(binding, task)
    lender.client._runs[(task_id, generation)] = {
        **receipt, 'task_id': task_id, 'execution_generation': generation}
    observed = await asyncio.to_thread(transport.info, profile='default',
        session_id=lender.ref.session_id, source='bot_room')
    home.runtime._report_pending_action(task, session_id=lender.ref.session_id, info=observed)
    action, = home.status('room-1')['pending_actions']
    assert action['approval']['command'] == 'echo approved'
    params = {key: action[key] for key in ('member_id', 'task_id', 'execution_generation', 'request_id')}
    lender.wire.calls.clear()
    if stopped:
        # Keep the lender prompt open and the home stopping, with no later poll clearing its cache.
        for hook in ('_interrupt_stopping_task', '_settle_stopping_completion', '_peer_stop_acknowledged'):
            monkeypatch.setattr(home.runtime, hook, lambda *_args: False)
        if stopped == 'stop':
            assert home.stop_room('room-1', cancel_id='stop') == 1
        else:
            driver.begin_task_cancel(home.db_path, task['identity'], cancel_id='runtime-stop',
                                     expected_cancel_generation=0, clock=time.time)
        assert driver.get_task(home.db_path, task['identity'])['status'] == 'stopping'
        assert home.status('room-1')['pending_actions'] == []
        with pytest.raises(RuntimeError, match='room approval is no longer pending'):
            await asyncio.to_thread(home.approve_room_task, 'room-1', choice='once', **params)
        assert lender.wire.calls == []
        assert lender.answers == []
        assert ('room-1', 'member-peer') not in home._pending_actions
    else:
        answer = await asyncio.to_thread(home.approve_room_task, 'room-1', choice='once', **params)
        assert answer['status'] == 'resolved'
        assert lender.wire.calls == [('POST', '/v1/runs/run_approval/approval')]
        assert lender.answers == [('approval', 'prompt', 'once')]
        assert home.status('room-1')['pending_actions'] == []
