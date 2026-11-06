"""Discard overlapping a Retry whose admission the member then refuses: the receipt Discard saved
names a run the member never kept, so a later Discard asks by key again (closing it) instead of
trusting that receipt, and Disband can finish. The closed key still refuses a late POST."""

import asyncio

import pytest

from gateway import hosted_room_driver as driver
from gateway.platforms import api_server_runs
from hermes_state_runtime import RuntimeStoreError
from tests.gateway.test_session_group_peer_discard import (
    _never_admitted_deferred_turn, gateway as gateway_fixture, linked_peer as linked_peer_fixture,
)
from tests.gateway.test_session_group_peers import call
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
from tui_gateway.hosted_room_peer_transport import build_member_dispatch

gateway, linked_peer = gateway_fixture, linked_peer_fixture


def _member_admits_through_its_authority(gateway, monkeypatch, admission):
    """The member's POST /v1/runs reserves the key, then admits to its session authority (bound
    only once the dispatch was validated, as on a gateway that runs session authority)."""
    from types import SimpleNamespace
    adapter = gateway.adapter

    async def no_live_owner(*_a, **_kw):
        # Session authority runs here, but this run is not admitted to it yet.
        monkeypatch.setattr(adapter, 'gateway_runner', SimpleNamespace(
            session_authority=object(), session_authorities=SimpleNamespace(active=lambda: None)))
    monkeypatch.setattr(adapter, '_admit_to_live_bot_chat', no_live_owner)

    async def admit_then_unbind(*args):
        try:
            return await admission(*args)
        finally:
            monkeypatch.setattr(adapter, 'gateway_runner', None)
    monkeypatch.setattr(api_server_runs, '_admit_run_to_authority', admit_then_unbind)


@pytest.mark.asyncio
@pytest.mark.parametrize('linked_peer', [True], indirect=True)
async def test_discard_after_a_refused_overlapping_admission_settles_the_turn(
        gateway, monkeypatch, linked_peer):
    service, db, identity = await _never_admitted_deferred_turn(gateway, monkeypatch, linked_peer)
    control = dict(room_id='linked', member_id='reviewer', task_id=identity.task_id, execution_generation=1)
    admit, overlapped = api_server_runs._admit_run_to_authority, []

    async def refuse(*_a, **_kw):
        raise RuntimeStoreError('permission_denied')

    async def discard_then_refuse(adapter, launch, run_id, *args):
        # Retry's POST reserved the key; Discard finds that reservation before admission ends.
        overlapped.append((run_id, await call(gateway.owner, 'groups.discard', **control)))
        monkeypatch.setattr('gateway.session_api_turn.admit_api_turn_async', refuse)
        return await admit(adapter, launch, run_id, *args)

    _member_admits_through_its_authority(gateway, monkeypatch, discard_then_refuse)
    assert await call(gateway.owner, 'groups.retry', **control) == 'peer_unreachable'
    (run_id, first), = overlapped
    assert first == 'session_busy'
    client = service.peer_clients[('linked', 'reviewer')]
    assert client._receipt(identity.task_id, 1)['run_id'] == run_id  # the receipt Discard saved
    assert driver.get_task(db, identity)['status'] == 'deferred'
    assert gateway.adapter._run_statuses == {}  # the member refused it: nothing runs there

    discarded = await call(gateway.owner, 'groups.discard', **control)
    assert isinstance(discarded, dict) and discarded['discarded'], discarded
    assert driver.get_task(db, identity)['status'] == 'cancelled'
    route = service.peer_routes[('linked', 'reviewer')]
    task = driver.get_task(db, identity)
    late = build_member_dispatch(
        binding=service.bindings()[0], route=route, room_id='linked', task_id=identity.task_id,
        target_profile='default', execution_generation=1, source_event_seq=task['payload']['source_event_seq'],
        prompt=task['payload']['prompt'], trace_id=route.trace_id)
    client._runs.clear()
    with pytest.raises(PeerRunsHTTPError) as refused:
        await asyncio.to_thread(client.dispatch, grant=route.grant, dispatch=late.as_mapping())
    assert refused.value.error_code == 'idempotency_key_conflict'
    assert gateway.adapter._run_statuses == {}
    assert 'tombstone' in await call(gateway.owner, 'groups.disband', room_id='linked')


@pytest.mark.asyncio
@pytest.mark.parametrize('linked_peer', [True], indirect=True)
async def test_disband_after_a_refused_overlapping_admission_finishes(gateway, monkeypatch, linked_peer):
    _service, db, identity = await _never_admitted_deferred_turn(gateway, monkeypatch, linked_peer)
    control = dict(room_id='linked', member_id='reviewer', task_id=identity.task_id, execution_generation=1)
    admit = api_server_runs._admit_run_to_authority

    async def refuse(*_a, **_kw):
        raise RuntimeStoreError('permission_denied')

    async def discard_then_refuse(adapter, launch, run_id, *args):
        assert await call(gateway.owner, 'groups.discard', **control) == 'session_busy'
        monkeypatch.setattr('gateway.session_api_turn.admit_api_turn_async', refuse)
        return await admit(adapter, launch, run_id, *args)

    _member_admits_through_its_authority(gateway, monkeypatch, discard_then_refuse)
    assert await call(gateway.owner, 'groups.retry', **control) == 'peer_unreachable'
    assert 'tombstone' in await call(gateway.owner, 'groups.disband', room_id='linked')
    assert driver.list_tasks(db, room_id='linked') == [] or all(
        t['status'] == 'cancelled' for t in driver.list_tasks(db, room_id='linked'))
    assert gateway.adapter._run_statuses == {}
