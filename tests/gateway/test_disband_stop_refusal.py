"""Disband's expected Stop refusal crosses both RPC surfaces with a stable reason."""

import asyncio

import pytest

from gateway import hosted_room_driver as driver, hosted_rooms
from tests.gateway.test_hosted_room_disband_publication import _surface
from tests.gateway.test_session_group_peers import (
    gateway as gateway_fixture, linked_peer as linked_peer_fixture, call)
from tests.tui_gateway.test_hosted_room_service import _FakeRPC
from tui_gateway import server

gateway, linked_peer = gateway_fixture, linked_peer_fixture


@pytest.mark.asyncio
@pytest.mark.parametrize('surface,code', [('canonical', 4001), ('standalone', 4113)])
async def test_disband_stop_pending_is_typed_and_retryable(gateway, linked_peer, monkeypatch, surface, code):
    service = gateway.service
    _surface(gateway, monkeypatch, surface)
    assert (await call(gateway.owner, 'groups.send', room_id='linked', event_id='ask',
                       payload={'text': '@host wait', 'thread_id': 'thread'}))['accepted']
    task, = driver.list_tasks(service.store, placement=service.placement, room_id='linked', status='queued')
    binding, = service.bindings()
    lease = service.runtime._ensure_lease(binding)
    driver.start_task(service.store, task['identity'], lease, placement=service.placement,
                      expected_cancel_generation=0, clock=service.runtime.clock)

    rpc = _FakeRPC()
    rpc.info = lambda **kwargs: {'active': True, 'task_id': task['identity'].task_id}
    rpc.interrupt = lambda **kwargs: None
    rpc.sessions['default', 'Group: linked'] = {'session_id': 'host-session'}
    service.runtime.transport_resolver = lambda *args: rpc

    async def disband():
        frame = {'id': 1, 'method': 'groups.disband', 'params': {'room_id': 'linked'}}
        if surface == 'canonical':
            return await gateway.owner.dispatch(frame)
        return await asyncio.to_thread(server.handle_request, frame)

    response = await disband()
    assert response['error']['code'] == code
    assert response['error']['data']['reason'] == 'room_stop_pending'
    assert response['error']['message'] == 'Group chat work is still stopping; retry deletion after Stop completes.'
    assert hosted_rooms.room_state(
        service.store, placement=service.placement, room_id='linked').get('disbanded_at') is None
    stopped = driver.get_task(service.store, task['identity'], placement=service.placement)
    assert stopped['status'] == 'stopping'
    driver.complete_task_cancel(service.store, task['identity'], placement=service.placement,
                                cancel_id=stopped['cancel_id'],
                                expected_cancel_generation=stopped['cancel_generation'], clock=service.runtime.clock)
    assert 'tombstone' in (await disband())['result']
