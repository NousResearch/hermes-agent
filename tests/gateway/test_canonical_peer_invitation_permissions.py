"""Real root invitation/refresh rights; no capable Files fixture or executor."""
import asyncio
import json
import threading
import time

import pytest

from gateway import hosted_room_peer as peer
from tests.gateway.test_canonical_peer_target_setup import target, invite, invitation, request  # noqa: F401


TEXT_RIGHTS = ['approve', 'dispatch', 'status', 'stop']


def receipts(target):
    return [json.loads(row[0]) for row in target.db._conn.execute(
        "SELECT value FROM state_meta WHERE key LIKE 'gateway.peer.invite.v1.%'")]


@pytest.mark.asyncio
async def test_real_native_and_http_text_rights_and_no_caller_selection(target):
    native = await invite(target)
    params = invitation()
    params.pop('request_id')
    response = await target.adapter._handle_room_member_invitation(request(params))
    assert response.status == 201, response.text
    http = json.loads(response.text)
    for issued in (native, http):
        assert issued['catalog']['text'] is True
        assert issued['catalog']['attachments'] is False
        claims = peer.decode_room_grant(target.adapter._room_grant_secret(), issued['grant'], permission='dispatch')
        assert claims['permissions'] == TEXT_RIGHTS
        assert claims['execution_policy_digest'] == issued['catalog']['execution_policy']['policy_digest']
        assert claims['target_install_id'] == issued['catalog']['installation_id']
        assert claims['target_profile'] == 'default'
        with pytest.raises(peer.HostedRoomGrantError, match='allow'):
            peer.decode_room_grant(target.adapter._room_grant_secret(), issued['grant'], permission='attachment.stage')
    receipt = receipts(target)[0]
    assert receipt['intent']['permissions'] == receipt['issue']['permissions'] == TEXT_RIGHTS
    for change in ({'permissions': ['attachment.stage']}, {'attachments': True}, {'catalog': native['catalog']}):
        response = await target.adapter._handle_room_member_invitation(request(params | change))
        assert response.status == 400, response.text
        response = await target.connection.dispatch(dict(id=2, method='groups.peer.invite', params=invitation() | change))
        assert 'error' in response
    assert await invite(target) == native


@pytest.mark.asyncio
@pytest.mark.parametrize('rights', [TEXT_RIGHTS, ['dispatch'], ['status'],
    ['attachment.stage', 'dispatch', 'status'],
    ['artifact.ack', 'artifact.read', 'dispatch', 'status'],
    ['artifact.read', 'status']])
async def test_refresh_preserves_existing_explicit_rights_and_hard_horizon(target, rights):
    from gateway.hosted_room_grant_state import grant_state_db_paths, reserve_grant_state
    issued = await invite(target)
    original = peer.decode_room_grant(target.adapter._room_grant_secret(), issued['grant'], permission='dispatch')
    fields = {name: original[name] for name in ('grant_id', 'room_id', 'home_install_id',
        'authority_gateway_id', 'authority_epoch', 'member_id', 'target_install_id',
        'target_profile', 'execution_policy_digest')}
    now = time.time()
    token = peer.issue_room_grant(target.adapter._room_grant_secret(), **fields,
        permissions=rights, issued_at=now, ttl_seconds=600, status_ttl_seconds=1200)
    claims = peer.decode_room_grant(target.adapter._room_grant_secret(), token, permission=rights[0])
    reserve_grant_state(grant_state_db_paths(target.home), claims=claims, expires_at=claims['status_expires_at'])
    response = await target.adapter._handle_room_member_grant_refresh(request({'ttl_seconds': 1800}, token=token))
    if 'dispatch' not in rights:
        assert response.status == 401, response.text
        return
    assert response.status == 200, response.text
    refreshed = peer.decode_room_grant(target.adapter._room_grant_secret(), json.loads(response.text)['grant'], permission='dispatch')
    assert refreshed['permissions'] == rights
    assert refreshed['status_expires_at'] == claims['status_expires_at']
    assert refreshed['expires_at'] == claims['status_expires_at']
    for right in ('attachment.stage', 'artifact.read', 'artifact.ack'):
        if right not in rights:
            with pytest.raises(peer.HostedRoomGrantError, match='allow'):
                peer.decode_room_grant(target.adapter._room_grant_secret(), json.loads(response.text)['grant'], permission=right)


@pytest.mark.asyncio
async def test_manifest_metadata_cannot_activate_files_or_silently_admit_text(target, monkeypatch):
    from gateway.platforms import api_server_runs
    from tests.gateway.test_canonical_peer_text_admission import dispatch
    issued = await invite(target)
    value = dispatch(issued).as_mapping() | {'attachment_manifest_digest': 'a' * 64}
    assert issued['catalog']['attachments'] is False
    executed = []

    async def inert(*args, **kwargs):
        executed.append(True)

    monkeypatch.setattr(api_server_runs, '_execute_run', inert)
    response = await target.adapter._handle_runs(request(
        {'hosted_room_dispatch': value}, token=issued['grant'], key='room:task-one:1'))
    await asyncio.sleep(0)
    assert response.status == 403, response.text
    assert json.loads(response.text)['error']['code'] == 'invalid_room_dispatch'
    assert not executed
    assert target.db._conn.execute('SELECT count(*) FROM sessions').fetchone()[0] == 0
    assert target.db._conn.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == 0


def change_endpoint(monkeypatch):
    # The YAML endpoint is restart-cached. Change the actual live override in
    # this test's private process, not a fake Files-ready catalog or a no-op edit.
    monkeypatch.setenv('HERMES_ROOM_LINK_URL', 'https://changed.example.test/hermes')


@pytest.mark.asyncio
@pytest.mark.parametrize('boundary', ['prepare', 'complete'])
async def test_native_rechecks_catalog_on_actual_owner_transaction(target, monkeypatch, boundary):
    entered, resume = threading.Event(), threading.Event()
    original = target.db._execute_write

    def write(fn, *args, **kwargs):
        if fn.__name__ == boundary:
            entered.set()
            assert resume.wait(10)
        return original(fn, *args, **kwargs)

    monkeypatch.setattr(target.db, '_execute_write', write)
    task = asyncio.create_task(target.connection.dispatch(dict(id=1, method='groups.peer.invite', params=invitation())))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        change_endpoint(monkeypatch)
    finally:
        resume.set()
    response = await asyncio.wait_for(task, 10)
    assert response.get('error', {}).get('message') == 'room_capability_catalog_changed', response
    saved = receipts(target)
    if boundary == 'prepare':
        assert saved == []
    else:
        assert len(saved) == 1 and saved[0]['status'] == 'pending'
        assert saved[0]['issue']['permissions'] == TEXT_RIGHTS


@pytest.mark.asyncio
async def test_http_rechecks_catalog_after_reservation_on_owner_transaction(target, monkeypatch):
    original = target.db._execute_write

    def write(fn, *args, **kwargs):
        if fn.__name__ == 'confirm':
            change_endpoint(monkeypatch)
        return original(fn, *args, **kwargs)

    monkeypatch.setattr(target.db, '_execute_write', write)
    params = invitation()
    params.pop('request_id')
    response = await target.adapter._handle_room_member_invitation(request(params))
    assert response.status == 400, response.text
    assert 'catalog changed' in json.loads(response.text)['error']['message']
