"""Root canonical target uses real bindings, policy, signer and dual grant stores."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp.test_utils import make_mocked_request

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.session import SessionStore
from gateway.session_authorities import SessionAuthorities
from gateway.session_authority import SessionAuthority
from gateway.session_controls import AuthorityConnection
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch


@pytest.fixture
def target(tmp_path, monkeypatch):
    home = tmp_path / '.hermes'
    home.mkdir()
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(home))
    (home / 'config.yaml').write_text('agent:\n  max_turns: 7\napprovals:\n  mode: manual\n')
    db = SessionDB(home / 'state.db')
    runner = SimpleNamespace(_draining=False, config=GatewayConfig(), adapters={},
        session_store=SessionStore(config=GatewayConfig(), sessions_dir=home / 'sessions'))
    authority = SessionAuthority(runner, profile_id=str(home), instance_id='owner', db=db,
        epoch=begin_runtime_epoch(db, instance_id='owner'))
    registry = runner.session_authorities = SessionAuthorities(home)
    registry.add(home, authority)
    runner.session_authority = authority
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={'key': 'synthetic-target-api-key'}))
    adapter.gateway_runner = runner
    runner.adapters[Platform.API_SERVER] = adapter
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    identity = dict(user_id='operator', provider='local', profile_id=str(home),
                    instance_id='owner', native_bootstrap=True)
    connection = AuthorityConnection(authority, AsyncMock(), identity, operator=True)
    yield SimpleNamespace(home=home, db=db, runner=runner, authority=authority,
                          adapter=adapter, connection=connection, identity=identity)
    adapter._run_idempotency_store._conn.close()
    db.close()


def invitation(**changes):
    return dict(request_id='invite-1', room_id='room-one', home_install_id='home-install',
        authority_gateway_id='home-gateway', authority_epoch=1, member_id='member-one',
        ttl_seconds=600, status_ttl_seconds=1200, **changes)


def request(body, *, token='', key='', path='/v1/runs'):
    req = make_mocked_request('POST', path, headers={
        'Authorization': 'HermesRoom ' + token if token else 'Bearer synthetic-target-api-key',
        'Idempotency-Key': key})
    req.json = AsyncMock(return_value=body)
    return req


async def invite(target, params=None):
    response = await target.connection.dispatch(dict(id=1, method='groups.peer.invite',
        params=invitation() if params is None else params))
    assert 'result' in response, response
    return response['result']


@pytest.mark.asyncio
async def test_native_invite_replay_real_catalog_dual_store_and_exact_revoke(target):
    from gateway import hosted_rooms as rooms
    from gateway.hosted_room_peer import decode_room_grant, issue_room_grant
    from gateway.hosted_room_grant_state import grant_state_db_paths
    first = await invite(target)
    assert await invite(target) == first
    paths = grant_state_db_paths(target.home)
    assert paths == (target.home / 'shared-state.db', target.home / 'state.db')
    claims = decode_room_grant(target.adapter._room_grant_secret(), first['grant'], permission='dispatch')
    assert all(rooms.peer_room_grant_is_current(path, claims=claims) for path in paths)
    assert first['catalog']['text'] is True and first['catalog']['attachments'] is False
    policy = first['catalog']['execution_policy']
    assert policy['max_iterations'] == 7 and policy['approval_mode'] == 'manual'
    assert 'bot_room' in policy['enabled_toolsets']
    cap = await target.adapter._handle_room_member_capabilities(request({}, token=first['grant']))
    assert cap.status == 200
    assert json.loads(cap.text)['catalog'] == first['catalog']
    changed = await target.connection.dispatch(dict(id=2, method='groups.peer.invite',
        params=invitation() | {'member_id': 'other'}))
    assert changed['error']['message'] == 'admission_conflict'
    # A replacement can reuse the grant ID; exact revoke must retain that token.
    signer = {key: claims[key] for key in ('grant_id', 'room_id', 'home_install_id',
        'authority_gateway_id', 'authority_epoch', 'member_id', 'target_install_id',
        'target_profile', 'execution_policy_digest', 'permissions')}
    replacement = issue_room_grant(target.adapter._room_grant_secret(), **signer,
        issued_at=claims['issued_at'] + 1, ttl_seconds=599, status_expires_at=claims['status_expires_at'])
    result = await target.connection.dispatch(dict(id=3, method='groups.peer.revoke_exact',
        params={'grant': first['grant']}))
    assert 'result' in result, result
    newer = decode_room_grant(target.adapter._room_grant_secret(), replacement, permission='status')
    assert all(rooms.room_grant_is_revoked(path, claims=claims) for path in paths)
    assert not any(rooms.room_grant_is_revoked(path, claims=newer) for path in paths)
    replay = await target.connection.dispatch(dict(id=4, method='groups.peer.invite', params=invitation()))
    assert 'error' in replay


@pytest.mark.asyncio
@pytest.mark.parametrize('denial', ['native', 'operator', 'control', 'profile', 'registry', 'store', 'approval'])
async def test_native_invite_denies_before_reservation(target, denial):
    from gateway import hosted_rooms as rooms
    if denial in {'native', 'operator', 'control'}:
        identity = dict(target.identity)
        if denial == 'native':
            identity['native_bootstrap'] = False
        if denial == 'control':
            identity['capabilities'] = ['session:read', 'session:operator']
        target.connection = AuthorityConnection(target.authority, AsyncMock(), identity, operator=denial != 'operator')
    elif denial == 'profile':
        target.connection.actor = type(target.connection.actor)('actor', 'foreign', frozenset({'session:operator', 'session:control'}), 't')
    elif denial == 'registry':
        target.runner.session_authorities._by_key.clear()
    elif denial == 'store':
        target.adapter._run_idempotency_store._db_path = None
    elif denial == 'approval':
        (target.home / 'config.yaml').write_text('approvals:\n  mode: off\n')
    response = await target.connection.dispatch(dict(id=1, method='groups.peer.invite', params=invitation()))
    assert 'error' in response, response
    assert not target.db._conn.execute("SELECT 1 FROM state_meta WHERE key LIKE 'gateway.peer.invite.%'").fetchone()
    assert not rooms.peer_room_is_reserved(target.home / 'shared-state.db', room_id='room-one', target_profile='default')


@pytest.mark.asyncio
async def test_receipt_pins_signer_and_pending_retry_never_remints(target, monkeypatch):
    from gateway.hosted_room_peer import decode_room_grant
    first = await invite(target)
    rows = target.db._conn.execute("SELECT key,value FROM state_meta WHERE key LIKE 'gateway.peer.invite.%'").fetchall()
    key, encoded = rows[0]
    receipt = json.loads(encoded)
    claims = decode_room_grant(target.adapter._room_grant_secret(), first['grant'], permission='status')
    assert receipt['token_sha256'] == claims['_token_sha256']
    with monkeypatch.context() as changed_signer:
        changed_signer.setattr(target.adapter, '_room_grant_secret', lambda: b'changed-synthetic-key' * 2)
        response = await target.connection.dispatch(dict(id=2, method='groups.peer.invite', params=invitation()))
        assert 'error' in response
    # A lost pending completion is not permission to mint/reserve on retry.
    receipt['status'] = 'pending'
    target.db._execute_write(lambda conn: conn.execute('UPDATE state_meta SET value=? WHERE key=?', (json.dumps(receipt), key)))
    response = await target.connection.dispatch(dict(id=2, method='groups.peer.invite', params=invitation()))
    assert response['error']['message'] == 'room_invitation_pending'
    assert json.loads(target.db._conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()[0]) == receipt


@pytest.mark.asyncio
async def test_http_invitation_keeps_strict_auth_schema_and_lifetime_contract(target):
    params = invitation()
    params.pop('request_id')
    path = '/v1/room-members/invitations'
    result = await target.adapter._handle_room_member_invitation(request(params, path=path))
    assert result.status == 201, result.text
    grant = json.loads(result.text)
    assert grant['catalog']['text'] is True
    assert grant['catalog']['execution_policy']['max_iterations'] == 7
    rejected = await target.adapter._handle_room_member_invitation(request(params, token=grant['grant'], path=path))
    assert rejected.status == 401
    for change in ({'request_id': 'native-only'}, {'ttl_seconds': '600'}, {'authority_epoch': True}):
        rejected = await target.adapter._handle_room_member_invitation(request(params | change, path=path))
        assert rejected.status == 400, rejected.text


@pytest.mark.asyncio
async def test_owner_partial_revoke_denies_observation_without_disabling_cleanup(target):
    from gateway import hosted_rooms
    from gateway.hosted_room_peer import decode_room_grant
    first = await invite(target)
    claims = decode_room_grant(target.adapter._room_grant_secret(), first['grant'], permission='status')
    hosted_rooms.revoke_room_grant_id(target.home / 'state.db', claims=claims, expires_at=claims['status_expires_at'])
    result = await target.adapter._handle_room_member_capabilities(request({}, token=first['grant']))
    assert result.status == 403, result.text
    # Cleanup must not depend on an execution-ready listener/catalog.
    target.runner.adapters.clear()
    (target.home / 'config.yaml').write_text('approvals:\n  mode: off\n')
    revoked = await target.connection.dispatch(dict(id=1, method='groups.peer.revoke', params={'grant': first['grant']}))
    assert revoked['result']['revoked'] is True


@pytest.mark.asyncio
async def test_native_invitation_consumes_existing_outbound_registration(target, monkeypatch):
    import asyncio
    from gateway import hosted_rooms, hosted_room_links
    from gateway.session_hosted_service import CanonicalHostedRoomService
    from tui_gateway import hosted_room_service
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient
    # Execution-only seam: never instantiate/start a coordinator or worker.
    monkeypatch.setattr(hosted_room_service, 'HostedRoomRuntime', lambda **kwargs: SimpleNamespace(
        status=lambda: {'running': True, 'stopping': False}, wakeup=lambda: None))
    service = CanonicalHostedRoomService(target.authority, None)
    target.authority.hosted_room_service = service
    installation = hosted_rooms.local_authority_gateway_id()
    params = invitation() | {'home_install_id': installation, 'authority_gateway_id': installation}
    first = await invite(target, params)
    service.authorize_room(target.connection.actor.subject, 'room-one', create=True)
    hosted_rooms.create_room(target.db.db_path, room_id='room-one', name='Room', authority_gateway_id=installation,
        members=[{'member_id': 'member-one', 'profile': 'default', 'handle': 'member-one', 'target': {
            'kind': 'peer', 'peer_id': installation, 'installation_id': installation, 'profile': 'default',
            'capability_digest': first['catalog']['catalog_digest']}}])
    def in_process_probe(client, *, grant):
        response = asyncio.run(target.adapter._handle_room_member_capabilities(request({}, token=grant)))
        assert response.status == 200, response.text
        return json.loads(response.text)
    monkeypatch.setattr(PeerRunsHTTPClient, 'probe', in_process_probe)
    result = await target.connection.dispatch(dict(id=2, method='groups.peer.register', params=dict(
        request_id='register-one', room_id='room-one', member_id='member-one', target_url=first['endpoint'],
        target_profile='default', grant=first['grant'], catalog=first['catalog'],
        cancellation_scope_id='cancel-one', trace_id='trace-one')))
    assert result.get('result', {}).get('registered') is True, result
    stored = hosted_room_links.load_room_link(target.db.db_path, room_id='room-one', member_id='member-one')
    assert stored.grant == first['grant'] and stored.catalog.as_mapping() == first['catalog']
    assert service.peer_routes[('room-one', 'member-one')].attachments is False
