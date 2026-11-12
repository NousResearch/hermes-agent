"""Retry can settle a Discard whose close reply was lost, without admitting a run."""

import pytest

from hermes_agent.gateway import hosted_room_driver as driver
from tests.gateway.test_session_group_peer_discard import (
    _never_admitted_deferred_turn, gateway as gateway_fixture, linked_peer as linked_peer_fixture,
)
from tests.gateway.test_session_group_peers import call, _deferred_unreceipted_peer_turn
from hermes_agent.tui_gateway.hosted_room_peer_http import PeerRunsHTTPError

gateway, linked_peer = gateway_fixture, linked_peer_fixture


@pytest.mark.asyncio
async def test_retry_after_lost_close_reply_cancels_the_same_attempt(gateway, monkeypatch, linked_peer):
    service, db, identity = await _never_admitted_deferred_turn(gateway, monkeypatch, linked_peer)
    client = service.peer_clients[('linked', 'reviewer')]
    request, admissions = client._request, []

    def lose_close_reply(path, **kwargs):
        result = request(path, **kwargs)
        if path == '/v1/room-members/runs/close':
            assert result['closed'] is True
            raise PeerRunsHTTPError('close reply lost', retryable=True)
        return result

    control = dict(room_id='linked', member_id='reviewer', task_id=identity.task_id, execution_generation=1)
    with monkeypatch.context() as lost:
        lost.setattr(client, '_request', lose_close_reply)
        assert await call(gateway.owner, 'groups.discard', **control) == 'peer_unreachable'
    assert driver.get_task(db, identity)['status'] == 'deferred'

    def record_retry(path, **kwargs):
        if path == '/v1/runs' and kwargs.get('method') == 'POST':
            admissions.append(kwargs['headers']['Idempotency-Key'])
        return request(path, **kwargs)

    monkeypatch.setattr(client, '_request', record_retry)
    retried = await call(gateway.owner, 'groups.retry', **control)
    assert isinstance(retried, dict) and retried['retried'], retried
    task = driver.get_task(db, identity)
    assert (task['status'], task['execution_generation']) == ('cancelled', 1)
    assert admissions and set(admissions) == {f'room:{identity.task_id}:1'}
    assert gateway.adapter._run_statuses == {}
    assert not service.status('linked')['pending_actions']
    assert (await call(gateway.owner, 'groups.discard', **control))['discarded']


@pytest.mark.asyncio
async def test_retry_conflict_with_an_admitted_run_keeps_the_turn(gateway, monkeypatch, linked_peer):
    service, db, identity = await _deferred_unreceipted_peer_turn(gateway, monkeypatch, linked_peer)
    client = service.peer_clients[('linked', 'reviewer')]
    request = client._request
    before = {k: dict(v) for k, v in gateway.adapter._run_statuses.items()}

    def conflicting_replay(path, **kwargs):
        if path == '/v1/runs' and kwargs.get('method') == 'POST':
            raise PeerRunsHTTPError('conflicting request', status_code=409,
                                    error_code='idempotency_key_conflict')
        return request(path, **kwargs)

    monkeypatch.setattr(client, '_request', conflicting_replay)
    assert await call(gateway.owner, 'groups.retry', room_id='linked', member_id='reviewer',
                      task_id=identity.task_id, execution_generation=1) == 'peer_unreachable'
    assert driver.get_task(db, identity)['status'] == 'deferred'
    assert gateway.adapter._run_statuses == before
