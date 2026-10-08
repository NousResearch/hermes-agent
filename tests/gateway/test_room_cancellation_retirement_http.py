"""Real HTTP retirement, permission and delayed-admission boundaries."""
import asyncio
import time
from unittest.mock import MagicMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway.hosted_room_peer import decode_room_grant, issue_room_grant
from gateway.platforms.api_server_run_authority import room_authority, room_run_scope
from tests.gateway.test_api_server_room_cancellation import _adapter, _app, _headers, _invitation


def app(adapter):
    result = _app(adapter)
    result.router.add_post('/v1/room-members/grants/revoke', adapter._handle_room_member_grant_revoke)
    return result


@pytest.mark.asyncio
async def test_mass_cancellation_disband_compacts_without_fabricating_absence(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path / 'runs.db')
    create = MagicMock(side_effect=AssertionError('a cancelled request executed'))
    monkeypatch.setattr(adapter, '_create_agent', create)
    try:
        async with TestClient(TestServer(app(adapter))) as cli:
            grant, body = await _invitation(cli)
            claims = decode_room_grant(adapter._room_grant_secret(), grant, permission='retire')
            headers = _headers(grant)
            for index in range(256):
                headers['Idempotency-Key'] = f'room:task-{index}:1'
                body['hosted_room_dispatch']['task_id'] = f'task-{index}'
                response = await cli.post('/v1/runs/stop', headers=headers, json=body)
                assert response.status == 200
                assert (await response.json())['admission_cancelled']
            store = adapter._run_idempotency_store
            assert store._conn.execute('SELECT COUNT(*) FROM run_idempotency').fetchone()[0] == 256
            # A stop-only bearer cannot upgrade its authority by setting a body flag.
            coords = {k: claims[k] for k in ('room_id', 'home_install_id', 'authority_gateway_id',
                'authority_epoch', 'member_id', 'target_install_id', 'target_profile', 'execution_policy_digest')}
            narrow = issue_room_grant(adapter._room_grant_secret(), grant_id='stop-only',
                                      **coords, permissions=('status', 'stop'), issued_at=time.time())
            denied = await cli.post('/v1/room-members/grants/revoke', headers=_headers(narrow),
                                    json={'retire_authority': True})
            assert denied.status == 403
            assert store._conn.execute('SELECT COUNT(*) FROM run_idempotency').fetchone()[0] == 256
            retired = await cli.post('/v1/room-members/grants/revoke', headers=_headers(grant),
                                     json={'retire_authority': True})
            assert retired.status == 200 and (await retired.json())['authority_retired']
            assert store._conn.execute('SELECT COUNT(*) FROM run_idempotency').fetchone()[0] == 0
            assert store._conn.execute('SELECT COUNT(*) FROM run_room_authorities').fetchone()[0] == 1
            for index in range(256):
                headers['Idempotency-Key'] = f'room:task-{index}:1'
                body['hosted_room_dispatch']['task_id'] = f'task-{index}'
                late = await cli.post('/v1/runs', headers=headers, json=body)
                assert late.status in {401, 403, 409}
            # Even an already-authorized request reaching the writer after retirement
            # gets history-retired, never a fabricated per-task cancellation receipt.
            assert store.reserve(room_run_scope(claims), 'room:task-0:1', 'f', 'late',
                {'status': 'queued'}, room_authority=room_authority(claims)) == ('authority_retired', None)
            # A fresh token cannot revive the same retired epoch.
            renewed = await cli.post('/v1/room-members/invitations',
                headers={'Authorization': 'Bearer test-room-key'},
                json={k: claims[k] for k in ('room_id', 'home_install_id', 'authority_gateway_id',
                                           'authority_epoch', 'member_id')})
            assert renewed.status == 400
            assert not create.called
    finally:
        adapter._run_idempotency_store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('successor_home', ['home', 'successor'])
@pytest.mark.parametrize('successor_member', ['member-1', 'member-2'])
async def test_successor_epoch_fences_admission_paused_before_writer(tmp_path, monkeypatch, successor_home, successor_member):
    adapter = _adapter(tmp_path / 'runs.db')
    entered, release = asyncio.Event(), asyncio.Event()
    history = adapter._conversation_history_for_session

    async def paused(*args, **kwargs):
        entered.set()
        await release.wait()
        return await history(*args, **kwargs)

    create = MagicMock(side_effect=AssertionError('retired request executed'))
    monkeypatch.setattr(adapter, '_create_agent', create)
    monkeypatch.setattr(adapter, '_conversation_history_for_session', paused)
    try:
        async with TestClient(TestServer(app(adapter))) as cli:
            grant, body = await _invitation(cli)
            if successor_member != 'member-1':
                extra = await cli.post('/v1/room-members/invitations', headers={'Authorization': 'Bearer test-room-key'},
                    json={'room_id': 'room-1', 'home_install_id': 'home', 'authority_gateway_id': 'home',
                          'authority_epoch': 1, 'member_id': successor_member})
                assert extra.status == 201
            pending = asyncio.create_task(cli.post('/v1/runs', headers=_headers(grant), json=body))
            await asyncio.wait_for(entered.wait(), 10)
            successor = await cli.post('/v1/room-members/invitations',
                headers={'Authorization': 'Bearer test-room-key'}, json={
                    'room_id': 'room-1', 'home_install_id': successor_home, 'authority_gateway_id': 'successor',
                    'authority_epoch': 2, 'member_id': successor_member, 'previous_authority': {
                        'home_install_id': 'home', 'authority_gateway_id': 'home', 'authority_epoch': 1}})
            assert successor.status == 201
            release.set()
            refused = await pending
            assert refused.status == 409
            assert (await refused.json())['error']['code'] == 'run_history_retired'
            assert not create.called
            assert adapter._run_idempotency_store._conn.execute('SELECT COUNT(*) FROM run_idempotency').fetchone()[0] == 0
            # The genuine successor can run new work; the fence is not a room-wide outage.
            if successor_member != 'member-1':
                successor = await cli.post('/v1/room-members/invitations',
                    headers={'Authorization': 'Bearer test-room-key'}, json={
                        'room_id': 'room-1', 'home_install_id': successor_home, 'authority_gateway_id': 'successor',
                        'authority_epoch': 2, 'member_id': 'member-1', 'previous_authority': {
                            'home_install_id': 'home', 'authority_gateway_id': 'home', 'authority_epoch': 1}})
                assert successor.status == 201
            invitation = await successor.json()
            agent = MagicMock()
            agent.run_conversation.return_value = {'final_response': 'new owner', 'messages': []}
            agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
            create.side_effect, create.return_value = None, agent
            body['hosted_room_dispatch'].update(home_install_id=successor_home, authority_epoch=2,
                                               authority_gateway_id='successor', task_id='new-task')
            headers = {**_headers(invitation['grant']), 'Idempotency-Key': 'room:new-task:1'}
            accepted = await cli.post('/v1/runs', headers=headers, json=body)
            assert accepted.status == 202, await accepted.json()
            await asyncio.wait_for(asyncio.gather(*adapter._active_run_tasks.values()), 10)
            assert agent.run_conversation.call_count == 1
    finally:
        release.set()
        adapter._run_idempotency_store.close()


@pytest.mark.asyncio
async def test_ordinary_revoke_and_reinvite_keeps_exact_same_epoch_cancellations(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path / 'runs.db')
    create = MagicMock(side_effect=AssertionError('cancelled request executed'))
    monkeypatch.setattr(adapter, '_create_agent', create)
    try:
        async with TestClient(TestServer(app(adapter))) as cli:
            grant, body = await _invitation(cli)
            cancelled = await cli.post('/v1/runs/stop', headers=_headers(grant), json=body)
            run_id = (await cancelled.json())['run_id']
            revoked = await cli.post('/v1/room-members/grants/revoke', headers=_headers(grant), json={})
            assert revoked.status == 200 and not (await revoked.json())['authority_retired']
            replacement, body = await _invitation(cli)
            stale_retirement = await cli.post('/v1/room-members/grants/revoke', headers=_headers(grant),
                                              json={'retire_authority': True})
            assert stale_retirement.status in {401, 403}
            replay = await cli.post('/v1/runs', headers=_headers(replacement), json=body)
            assert replay.status == 202
            result = await replay.json()
            assert result['run_id'] == run_id and result['status'] == 'cancelled'
            assert not create.called
    finally:
        adapter._run_idempotency_store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('member_id', ['member-1', 'another-member'])
async def test_unlinked_collision_leaves_original_paused_writer_authorized(tmp_path, monkeypatch, member_id):
    adapter = _adapter(tmp_path / 'runs.db')
    entered, release = asyncio.Event(), asyncio.Event()
    history = adapter._conversation_history_for_session
    async def pause(*args, **kwargs):
        entered.set()
        await release.wait()
        return await history(*args, **kwargs)
    agent = MagicMock()
    agent.run_conversation.return_value = {'final_response': 'original owner', 'messages': []}
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    monkeypatch.setattr(adapter, '_create_agent', MagicMock(return_value=agent))
    monkeypatch.setattr(adapter, '_conversation_history_for_session', pause)
    try:
        async with TestClient(TestServer(app(adapter))) as cli:
            grant, body = await _invitation(cli)
            pending = asyncio.create_task(cli.post('/v1/runs', headers=_headers(grant), json=body))
            await asyncio.wait_for(entered.wait(), 15)
            unrelated = await cli.post('/v1/room-members/invitations',
                headers={'Authorization': 'Bearer test-room-key'}, json={
                    'room_id': 'room-1', 'home_install_id': 'unrelated', 'authority_gateway_id': 'unrelated',
                    'authority_epoch': 2, 'member_id': member_id})
            assert unrelated.status == 400
            release.set()
            assert (await pending).status == 202
            await asyncio.wait_for(asyncio.gather(*adapter._active_run_tasks.values()), 15)
            assert agent.run_conversation.call_count == 1
    finally:
        release.set()
        adapter._run_idempotency_store.close()
