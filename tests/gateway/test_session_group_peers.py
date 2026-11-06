"""Cross-gateway members of gateway-owned Group Chats: honest capability, operator-only grants,
pinned registration, and Disband that never leaves a grant behind."""
import asyncio
from dataclasses import replace
import json
import logging
import threading
import time
from types import SimpleNamespace

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestServer

from gateway import hosted_room_links, hosted_rooms
from gateway.config import Platform, PlatformConfig
from gateway.platforms import api_server_room_grants
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
from gateway.session_controls import AuthorityConnection
from gateway.session_group_peers import room_link
from gateway.session_hosted_service import CanonicalHostedRoomService
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch
from tui_gateway.contracts import groups_bot_relay as contract
from tui_gateway.contracts.registry import CANONICAL_METHODS


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    """One gateway that hosts the Group Chat and also serves its peer member over loopback."""
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.delenv('HERMES_ROOM_LINK_URL', raising=False)
    db = SessionDB(home / 'state.db')
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={'key': 'target-gateway-api-key'}))
    adapter._run_idempotency_store.close()
    adapter._run_idempotency_store = RunIdempotencyStore(str(tmp_path / 'runs.db'))
    authority = SimpleNamespace(profile_id=str(home), instance_id='owner', db=db, events={}, sessions={},
                                epoch=begin_runtime_epoch(db, instance_id='owner'),
                                runner=SimpleNamespace(adapters={Platform.API_SERVER: adapter}))
    service = authority.hosted_room_service = CanonicalHostedRoomService(authority, None)
    service.runtime.status = lambda: {'running': True, 'stopping': False}
    yield SimpleNamespace(home=home, db=db, adapter=adapter, authority=authority, service=service,
                          owner=AuthorityConnection(authority, object(), {'user_id': 'owner'}, operator=True))
    adapter._run_idempotency_store.close()
    db.close()


async def call(connection, method, **params):
    reply = await connection.dispatch({'id': 1, 'method': method, 'params': params})
    error = reply.get('error', {})
    return reply.get('result', error.get('data', {}).get('reason', error.get('message')))


async def serve(gateway, monkeypatch, port=None, *, stop=False):
    app = web.Application()
    for method, path, handler in api_server_room_grants._http_routes(gateway.adapter):
        app.router.add_route(method, path, handler)
    app.router.add_post('/v1/runs', gateway.adapter._handle_runs)
    app.router.add_get('/v1/runs/{run_id}', gateway.adapter._handle_get_run)
    if stop:
        app.router.add_post('/v1/runs/{run_id}/stop', gateway.adapter._handle_stop_run)
    server = TestServer(app, port=port)
    await server.start_server()
    url = str(server.make_url('')).rstrip('/')
    monkeypatch.setenv('HERMES_ROOM_LINK_URL', url)
    return server, url


async def linked_room(gateway, catalog, room_id='linked'):
    pinned = {'kind': 'peer', 'peer_id': 'target', 'installation_id': catalog['installation_id'],
              'profile': 'default', 'capability_digest': catalog['catalog_digest']}
    created = await call(gateway.owner, 'groups.create', room_id=room_id, name='Linked', members=[
        {'member_id': 'host', 'profile': 'default', 'handle': 'host'},
        {'member_id': 'reviewer', 'profile': 'default', 'handle': 'reviewer', 'target': pinned}])
    return created['room']


@pytest_asyncio.fixture
async def linked_peer(gateway, monkeypatch, request):
    server, url = await serve(gateway, monkeypatch, stop=getattr(request, 'param', False))
    try:
        catalog = room_link(gateway.authority)['catalog']
        room = await linked_room(gateway, catalog)
        yield server, url, catalog, room
    finally:
        await server.close()


async def invite(gateway, room, member_id='reviewer'):
    return await call(gateway.owner, 'groups.peer.invite', room_id=room['room_id'], member_id=member_id,
                      home_install_id=room['authority_gateway_id'],
                      authority_gateway_id=room['authority_gateway_id'],
                      authority_epoch=room['authority_epoch'])


def test_room_link_is_enabled_only_when_this_gateway_can_host_a_member(gateway, monkeypatch):
    authority, adapter = gateway.authority, gateway.adapter
    assert room_link(SimpleNamespace(profile_id=authority.profile_id)) == {
        'enabled': False, 'reason': 'api_server_required'}
    named = gateway.home / 'profiles' / 'helper'
    named.mkdir(parents=True)
    named_link = room_link(SimpleNamespace(profile_id=str(named), runner=authority.runner))
    assert named_link == {'enabled': False, 'reason': 'default_profile_required'}
    assert room_link(authority) == {'enabled': False, 'reason': 'endpoint_required'}
    monkeypatch.setenv('HERMES_ROOM_LINK_URL', 'http://127.0.0.1:9')
    durable = adapter._run_idempotency_store
    adapter._run_idempotency_store = RunIdempotencyStore(':memory:')
    assert room_link(authority) == {'enabled': False, 'reason': 'durable_run_storage_required'}
    adapter._run_idempotency_store.close()
    adapter._run_idempotency_store = durable
    (gateway.home / 'config.yaml').write_text('approvals:\n  mode: "off"\n')
    assert room_link(authority) == {'enabled': False, 'reason': 'execution_policy_unsupported'}
    (gateway.home / 'config.yaml').write_text('approvals:\n  mode: manual\n')
    link = room_link(authority)
    assert link['enabled'] and link['profile'] == 'default', link
    assert link['endpoint'] == link['catalog']['endpoint'] == {
        'available': True, 'url': 'http://127.0.0.1:9', 'transport_security': 'loopback'}
    assert link['catalog']['text'] and not link['catalog']['attachments']

    capabilities = asyncio.run(call(gateway.owner, 'groups.capabilities'))
    assert capabilities['room_link'] == link
    # Reading capabilities never mints the gateway's RoomLink secret; issuing a grant does.
    assert not (gateway.home / '.room-link-grant-secret').exists()
    assert {'groups.peer.register', 'groups.peer.invite', 'groups.peer.revoke'} <= set(capabilities['methods'])
    contract.GroupsCapabilitiesResult.model_validate(capabilities)
    monkeypatch.delenv('HERMES_ROOM_LINK_URL')
    contract.GroupsCapabilitiesResult.model_validate(asyncio.run(call(gateway.owner, 'groups.capabilities')))


@pytest.mark.asyncio
async def test_home_without_room_link_can_register_and_run_a_lent_member(gateway, monkeypatch, linked_peer, tmp_path):
    """The home owns the group chat without lending bots; a separate gateway lends its member."""
    from gateway.platforms import api_server_runs
    from gateway.run import _profile_runtime_scope

    async def complete(adapter, run, **kwargs):
        adapter._set_run_status(run.run_id, 'completed', output='peer reply', completed=True)

    monkeypatch.setattr(api_server_runs, '_execute_run', complete)
    _server, url, catalog, _room = linked_peer
    lender = await call(gateway.owner, 'groups.capabilities')
    assert lender['room_link']['enabled'] is True and 'groups.peer.invite' in lender['methods']
    assert lender['room_link']['catalog'] == catalog

    home = tmp_path / 'owner'
    home.mkdir()
    with SessionDB(home / 'state.db') as db:
        authority = SimpleNamespace(profile_id=str(home), instance_id='home', db=db, events={}, sessions={},
                                    epoch=begin_runtime_epoch(db, instance_id='home'))
        with _profile_runtime_scope(home):
            service = authority.hosted_room_service = CanonicalHostedRoomService(authority, None)
        service.runtime.status = lambda: {'running': True, 'stopping': False}
        owner = AuthorityConnection(authority, object(), {'user_id': 'owner'}, operator=True)
        capabilities = await call(owner, 'groups.capabilities')
        assert capabilities['driver'] is True and capabilities['persistent_process'] is True
        assert 'groups.peer.register' in capabilities['methods']
        assert capabilities['room_link']['enabled'] is False

        room = await linked_room(SimpleNamespace(owner=owner), catalog)
        grant = (await invite(gateway, room))['grant']
        registered = await call(owner, 'groups.peer.register', room_id='linked', member_id='reviewer',
                                target_url=url, target_profile='default', grant=grant, catalog=catalog)
        assert isinstance(registered, dict) and registered['registered'], registered
        assert (await call(owner, 'groups.send', room_id='linked', event_id='ask',
                           payload={'text': '@reviewer review this', 'thread_id': 'thread'}))['accepted']

        def drive():
            with _profile_runtime_scope(home):
                binding, = service.bindings()
                service.runtime._run_room_once(binding)
                service.prepare_room(binding)

        await asyncio.to_thread(drive)
        log = await call(owner, 'groups.log', room_id='linked')
        replies = [event for event in log['events'] if event['kind'] == 'message.member']
        assert [(reply['actor']['id'], reply['payload']['text']) for reply in replies] == [('reviewer', 'peer reply')]


def test_peer_methods_enforce_operator_control_and_exact_params(gateway, monkeypatch):
    authority = gateway.authority
    member = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    reader = AuthorityConnection(authority, object(), {'user_id': 'owner', 'capabilities': ['session:read']})
    identity = dict(room_id='linked', home_install_id='install:home', authority_gateway_id='install:home',
                    authority_epoch=1, member_id='reviewer')

    async def probe():
        assert await call(member, 'groups.peer.invite', **identity) == 'permission_denied'
        assert await call(member, 'groups.peer.revoke', grant='grant') == 'permission_denied'
        assert await call(reader, 'groups.peer.register', room_id='linked') == 'permission_denied'
        assert await call(gateway.owner, 'groups.peer.invite', **identity) == 'room_link_unavailable'
        monkeypatch.setenv('HERMES_ROOM_LINK_URL', 'http://127.0.0.1:9')
        # Grant ids are the target's to choose.
        for params in ({**identity, 'grant_id': 'chosen'}, {**identity, 'status_ttl_seconds': None},
                       {**identity, 'authority_epoch': '1'}, {**identity, 'authority_epoch': True},
                       {**identity, 'authority_epoch': 0}, {**identity, 'authority_epoch': 2**63},
                       {**identity, 'member_id': ''}, {**identity, 'room_id': 'r' * 129},
                       {**identity, 'ttl_seconds': True}, {**identity, 'ttl_seconds': 30},
                       {**identity, 'ttl_seconds': float('nan')}, {**identity, 'ttl_seconds': float('inf')}):
            assert await call(gateway.owner, 'groups.peer.invite', **params) == 'invalid_params', params
        assert await call(gateway.owner, 'groups.peer.revoke', grant=7) == 'invalid_params'
        # The declared keys are exactly the accepted ones: an explicit null lifetime is refused by both.
        assert await call(gateway.owner, 'groups.peer.invite', **identity, ttl_seconds=None) == 'invalid_params'
        with pytest.raises(ValueError):
            CANONICAL_METHODS['groups.peer.invite'].params.model_validate({**identity, 'ttl_seconds': None})
        assert await call(gateway.owner, 'groups.peer.revoke', grant='not.a-grant') == 'invalid_room_grant'
        assert await call(gateway.owner, 'groups.peer.revoke', grant='g', room_id='linked') == 'invalid_params'
        link = room_link(authority)
        room = await linked_room(gateway, link['catalog'])
        base = dict(room_id=room['room_id'], member_id='reviewer', grant='grant', catalog=link['catalog'],
                    target_url='http://127.0.0.1:9', target_profile='default')
        CANONICAL_METHODS['groups.peer.register'].params.model_validate(base)
        # Route identity is the home's to derive, so a client cannot choose it.
        for params in ({**base, 'target_url': 'http://peer.example.test'}, {**base, 'catalog': {}},
                       {**base, 'grant': ''}, {k: v for k, v in base.items() if k != 'target_profile'},
                       {**base, 'trace_id': 'chosen'}, {**base, 'cancellation_scope_id': 'chosen'}):
            assert await call(gateway.owner, 'groups.peer.register', **params) == 'invalid_params', params
        stranger = AuthorityConnection(authority, object(), {'user_id': 'stranger'})
        assert await call(stranger, 'groups.peer.register', **base) == 'permission_denied'
        gateway.service.runtime.status = lambda: {'running': False}
        assert await call(gateway.owner, 'groups.peer.register', **base) == 'runtime_coordination_required'
    asyncio.run(probe())


@pytest.mark.asyncio
async def test_register_binds_only_the_pinned_peer_after_a_live_scoped_probe(gateway, monkeypatch, linked_peer):
    from gateway.hosted_room_peer import (
        catalog_mapping, decode_room_grant, gateway_room_grant_secret, issue_room_grant)
    _server, url, catalog, room = linked_peer
    invitation = await invite(gateway, room)
    assert invitation['catalog'] == catalog and invitation['endpoint'] == catalog['endpoint']
    CANONICAL_METHODS['groups.peer.invite'].result.model_validate(invitation)
    claims = decode_room_grant(gateway_room_grant_secret(), invitation['grant'], permission='status')
    assert claims['permissions'] == ['approve', 'dispatch', 'status', 'stop']
    register = dict(room_id='linked', member_id='reviewer', target_url=url, target_profile='default',
                    grant=invitation['grant'], catalog=catalog)
    other_member = (await invite(gateway, room, member_id='host'))['grant']
    with_files = catalog_mapping(installation_id=catalog['installation_id'], persistent_process=True,
                                 attachments=True, target_profile='default', endpoint=catalog['endpoint'])
    for change, reason in [({'member_id': 'host'}, 'peer_target_mismatch'),
                           ({'target_profile': 'helper'}, 'peer_target_mismatch'),
                           ({'catalog': with_files}, 'peer_target_mismatch'),
                           ({'grant': other_member}, 'peer_target_mismatch')]:
        reply = await gateway.owner.dispatch({'id': 1, 'method': 'groups.peer.register',
                                              'params': register | change})
        assert reply['error']['message'] == reason, (change, reply)
        assert invitation['grant'] not in json.dumps(reply)
    await linked_room(gateway, with_files, room_id='files')
    assert await call(gateway.owner, 'groups.peer.register', **register | {
        'room_id': 'files', 'catalog': with_files}) == 'peer_target_unsupported'
    assert gateway.service.peer_routes == {}

    registered = await call(gateway.owner, 'groups.peer.register', **register)
    assert registered == {'registered': True, 'mode': 'direct', 'transport_security': 'loopback',
                          'target_install_id': catalog['installation_id'], 'target_profile': 'default'}
    CANONICAL_METHODS['groups.peer.register'].result.model_validate(registered)
    stored, = hosted_room_links.load_room_links(gateway.db.db_path)
    assert (stored.room_id, stored.member_id, stored.grant, stored.target_url) == (
        'linked', 'reviewer', invitation['grant'], url)
    assert gateway.service.peer_routes[('linked', 'reviewer')].trace_id == stored.trace_id

    # Only the target's own operator can revoke, and only grants it issued itself.
    foreign = issue_room_grant(gateway_room_grant_secret(), grant_id='foreign', room_id='linked',
                               home_install_id=room['authority_gateway_id'],
                               authority_gateway_id=room['authority_gateway_id'], authority_epoch=1,
                               member_id='reviewer', target_install_id='install:another',
                               target_profile='default')
    assert await call(gateway.owner, 'groups.peer.revoke', grant=foreign) == 'permission_denied'
    revoked = await call(gateway.owner, 'groups.peer.revoke', grant=invitation['grant'])
    assert revoked == {'revoked': True}
    CANONICAL_METHODS['groups.peer.revoke'].result.model_validate(revoked)
    assert await call(gateway.owner, 'groups.peer.register', **register) == 'room_reauthorization_required'


@pytest.mark.asyncio
async def test_an_invitation_can_keep_its_route_renewable_past_its_first_dispatch_lifetime(gateway, monkeypatch, linked_peer):
    """Renewal extends dispatch only within the operator's requested status horizon."""
    from gateway.hosted_room_peer import decode_room_grant, gateway_room_grant_secret
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient
    _server, url, catalog, room = linked_peer
    assert 'peer_renewal_horizon' in (await call(gateway.owner, 'groups.capabilities'))['features']
    identity = dict(room_id=room['room_id'], member_id='reviewer', home_install_id=room['authority_gateway_id'],
                    authority_gateway_id=room['authority_gateway_id'], authority_epoch=room['authority_epoch'])
    client, secret, month = PeerRunsHTTPClient(base_url=url, api_key=''), gateway_room_grant_secret(), 30 * 86400

    async def renewal(**lifetimes):
        invited = await call(gateway.owner, 'groups.peer.invite', **identity, **lifetimes)
        CANONICAL_METHODS['groups.peer.invite'].params.model_validate({**identity, **lifetimes})
        refreshed = await asyncio.to_thread(
            client.refresh_grant, grant=invited['grant'], capability_digest=catalog['catalog_digest'],
            execution_policy_digest=catalog['execution_policy']['policy_digest'])
        return [decode_room_grant(secret, grant, permission='status')
                for grant in (invited['grant'], refreshed['grant'])]

    first, renewed = await renewal(ttl_seconds=60)
    assert first['status_expires_at'] == first['expires_at']
    assert renewed['expires_at'] == pytest.approx(first['expires_at'])
    first, renewed = await renewal(ttl_seconds=60, status_ttl_seconds=month)
    assert first['status_expires_at'] == pytest.approx(first['issued_at'] + month)
    assert renewed['expires_at'] > first['expires_at'] + 23 * 3600
    assert renewed['status_expires_at'] == first['status_expires_at']
    for horizon in (True, '3600', 30, month + 1, float('nan'), float('inf'), None, 10**400):
        assert await call(gateway.owner, 'groups.peer.invite', **identity, ttl_seconds=60,
                          status_ttl_seconds=horizon) == 'invalid_params', horizon
    assert await call(gateway.owner, 'groups.peer.invite', **identity, ttl_seconds=10**400) == 'invalid_params'
    with pytest.raises(ValueError):  # absent means ttl_seconds; null is refused, as for ttl_seconds
        CANONICAL_METHODS['groups.peer.invite'].params.model_validate({**identity, 'status_ttl_seconds': None})
    horizon_schema = CANONICAL_METHODS['groups.peer.invite'].params.model_json_schema()['properties']['status_ttl_seconds']
    assert 'default' not in horizon_schema and horizon_schema['type'] == 'number', horizon_schema


@pytest.mark.asyncio
async def test_disband_owes_a_renewable_invitation_until_its_status_horizon(gateway, monkeypatch, linked_peer):
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient
    _server, url, catalog, room = linked_peer
    invitation = await call(
        gateway.owner, 'groups.peer.invite', room_id=room['room_id'], member_id='reviewer',
        home_install_id=room['authority_gateway_id'], authority_gateway_id=room['authority_gateway_id'],
        authority_epoch=room['authority_epoch'], ttl_seconds=60, status_ttl_seconds=30 * 86400)
    original = invitation['grant']
    renewed = await asyncio.to_thread(
        PeerRunsHTTPClient(base_url=url, api_key='').refresh_grant, grant=original,
        capability_digest=catalog['catalog_digest'],
        execution_policy_digest=catalog['execution_policy']['policy_digest'])
    registered = await call(gateway.owner, 'groups.peer.register', room_id=room['room_id'],
                            member_id='reviewer', target_url=url, target_profile='default',
                            grant=renewed['grant'], catalog=catalog)
    assert registered['registered'], registered
    # Disband revokes at the member gateway first: its scope revocation covers the original grant
    # and every renewal of it, so nothing is left owed.
    assert 'tombstone' in await call(gateway.owner, 'groups.disband', room_id=room['room_id'])
    assert hosted_room_links.load_room_links(gateway.db.db_path) == ()
    assert _owed(gateway.db.db_path) == []
    for grant in (original, renewed['grant']):
        assert await _grant_error(url, grant) == 'room_reauthorization_required'


@pytest.mark.asyncio
async def test_reregistered_route_replays_accepted_work_as_the_same_target_run(gateway, monkeypatch, linked_peer):
    from gateway.platforms import api_server_runs
    from tui_gateway.hosted_room_driver import HostedRoomBinding
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError
    from tui_gateway.hosted_room_peer_transport import build_member_dispatch

    async def inert(adapter, run, **kwargs):
        pass
    monkeypatch.setattr(api_server_runs, '_execute_run', inert)
    _server, url, catalog, room = linked_peer
    binding = HostedRoomBinding('linked', room['authority_gateway_id'], room['authority_epoch'])
    key = ('linked', 'reviewer')

    async def register():
        grant = (await invite(gateway, room))['grant']
        result = await call(gateway.owner, 'groups.peer.register', room_id='linked', member_id='reviewer',
                            target_url=url, target_profile='default', grant=grant, catalog=catalog)
        assert result['registered'], result
        return gateway.service.peer_routes[key], gateway.service.peer_clients[key]

    def dispatch(route):
        return build_member_dispatch(
            binding=binding, route=route, room_id='linked', task_id='dtask:accepted',
            target_profile='default', execution_generation=1, source_event_seq=1, prompt='review',
            trace_id=route.trace_id).as_mapping()

    first, client = await register()
    accepted = await asyncio.to_thread(client.dispatch, dispatch=dispatch(first), grant=first.grant)
    # The acceptance reply is lost before the home records it.
    with gateway.db._lock:
        gateway.db._conn.execute('DELETE FROM hosted_room_remote_runs')
        gateway.db._conn.commit()
    second, client = await register()
    assert second.grant != first.grant
    assert (second.trace_id, second.cancellation_scope_id) == (first.trace_id, first.cancellation_scope_id)
    recovered = await asyncio.to_thread(client.recover_dispatch, dispatch=dispatch(second), grant=second.grant)
    assert recovered['run_id'] == accepted['run_id'] and recovered['replayed'], recovered

    # A route minted with a fresh identity would turn that replay into a conflicting request,
    # which never resolves to the accepted run.
    drifted = replace(second, trace_id='trace-' + 'f' * 32)
    fresh = PeerRunsHTTPClient(base_url=url, api_key='')
    with pytest.raises(PeerRunsHTTPError) as refused:
        await asyncio.to_thread(fresh.recover_dispatch, dispatch=dispatch(drifted), grant=second.grant)
    assert refused.value.status_code == 409 and refused.value.ambiguous


def _owed(db_path):
    """The revokes this home still owes, as (room, member, attempts); the grant itself is not read."""
    import sqlite3
    with sqlite3.connect(db_path) as conn:
        return conn.execute('SELECT room_id, member_id, attempts FROM hosted_room_owed_revokes '
                            'ORDER BY room_id, member_id').fetchall()


@pytest.mark.parametrize('obligation', ['route', 'held', 'owed', 'expired', 'none'])
@pytest.mark.parametrize('room_id', ['linked', ' linked '])
def test_disband_without_a_service_requires_coordination_for_peer_routes(tmp_path, monkeypatch, obligation, room_id):
    from gateway.hosted_room_peer import GatewayRoomCatalog, catalog_mapping
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    with SessionDB(tmp_path / 'state.db') as db:
        authority = SimpleNamespace(profile_id=str(tmp_path), instance_id='owner', db=db, events={})
        owner = AuthorityConnection(authority, object(), {'user_id': 'owner'})
        endpoint = {'available': True, 'url': 'http://127.0.0.1:9', 'transport_security': 'loopback'}
        catalog = catalog_mapping(installation_id='install:target', persistent_process=True,
                                  target_profile='default', endpoint=endpoint)

        async def probe():
            pinned = {'kind': 'peer', 'peer_id': 'target', 'installation_id': 'install:target',
                      'profile': 'default', 'capability_digest': catalog['catalog_digest']}
            await call(owner, 'groups.create', room_id='linked', name='Linked', members=[
                {'member_id': 'host', 'profile': 'default', 'handle': 'host'},
                {'member_id': 'reviewer', 'profile': 'default', 'handle': 'reviewer', 'target': pinned}])
            if obligation == 'route':
                hosted_room_links.save_room_link(db.db_path, hosted_room_links.make_stored_link(
                    room_id='linked', member_id='reviewer', target_url=endpoint['url'], target_profile='default',
                    grant='stored-grant', catalog=GatewayRoomCatalog.from_mapping(catalog),
                    cancellation_scope_id='cancel-linked', trace_id='trace-linked'))
            if obligation in {'held', 'owed', 'expired'}:
                with hosted_rooms._transaction(db.db_path, immediate=True) as conn:
                    hosted_rooms._owe_revoke_locked(conn, room_id='linked', member_id='reviewer',
                        target_url=endpoint['url'], grant='retained-grant',
                        expires_at=time.time() + (-1 if obligation == 'expired' else 3600),
                        now=time.time() - 10, held=obligation != 'owed')
            owed = _owed(db.db_path)
            links = hosted_room_links.load_room_links(db.db_path)
            result = await call(owner, 'groups.disband', room_id=room_id)
            if obligation in {'route', 'held', 'owed'}:
                assert result == 'runtime_coordination_required'
                assert hosted_rooms.room_state(db.db_path, room_id='linked').get('disbanded_at') is None
            else:
                assert 'tombstone' in result
            assert hosted_room_links.load_room_links(db.db_path) == links
            if obligation != 'expired':
                assert _owed(db.db_path) == owed
        asyncio.run(probe())


@pytest.mark.asyncio
async def test_disband_replay_without_service_preserves_later_owed_debt(gateway, linked_peer):
    _, url, _, _ = linked_peer
    first = (await call(gateway.owner, 'groups.disband', room_id='linked'))['tombstone']
    hosted_rooms.owe_room_grant_revoke(gateway.db.db_path, room_id='linked', member_id='reviewer',
        target_url=url, grant='late-grant', expires_at=time.time() + 3600)
    debt = _owed(gateway.db.db_path)
    assert debt
    gateway.authority.hosted_room_service = None
    replay = await call(gateway.owner, 'groups.disband', room_id=' linked ')
    assert isinstance(replay, dict) and 'tombstone' in replay, replay
    assert replay['tombstone'] == {**first, 'idempotent': True, 'event': {**first['event'], 'idempotent': True}}
    assert _owed(gateway.db.db_path) == debt


@pytest.mark.asyncio
async def test_disband_keeps_the_room_while_its_member_gateway_is_down_and_revokes_once_it_answers(
        gateway, monkeypatch, linked_peer, caplog):
    """The member's gateway goes down while its room is live, after a re-seat replaced a longer-lived
    grant with a shorter one. Disband revokes at the member gateway first, so it refuses and keeps
    the room and its route; once that gateway answers, Disband ends the room and its scope
    revocation covers both grants. No grant appears in an answer, error or log."""
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient
    caplog.set_level(logging.DEBUG)
    server, url, catalog, room = linked_peer
    restarted, db = None, gateway.db.db_path
    try:
        long_lived, _ = await _seat(gateway, room, url, catalog, ttl_seconds=86400)
        grant, registered = await _seat(gateway, room, url, catalog)  # a re-seat with a one-hour grant
        assert registered['registered'], registered
        port = server.port
        await server.close()

        refused = await gateway.owner.dispatch({'id': 1, 'method': 'groups.disband', 'params': {'room_id': 'linked'}})
        assert 'error' in refused, refused
        assert hosted_rooms.room_state(db, room_id='linked').get('disbanded_at') is None
        assert ('linked', 'reviewer') in gateway.service.peer_routes
        assert len(hosted_room_links.load_room_links(db)) == 1

        restarted, _ = await serve(gateway, monkeypatch, port=port)
        disbanded = await gateway.owner.dispatch({'id': 2, 'method': 'groups.disband', 'params': {'room_id': 'linked'}})
        assert 'tombstone' in disbanded['result'], disbanded
        assert gateway.service.peer_routes == {} and hosted_room_links.load_room_links(db) == ()
        assert _owed(db) == []  # the held row for the replaced grant goes with the scope revocation
        for revoked in (long_lived, grant):
            assert await _grant_error(url, revoked) == 'room_reauthorization_required'
            assert not any(revoked in text for text in (json.dumps(refused), json.dumps(disbanded), caplog.text))
    finally:
        if restarted is not None:
            await restarted.close()


def test_an_owed_revoke_settles_on_unknown_or_revoked_and_is_dropped_at_its_grant_expiry(gateway):
    """A member gateway's answer that the grant is unknown or already revoked settles the revoke; any
    other refusal keeps it owed, with backoff. Past the grant's own expiry it is dropped unsent: that
    gateway refuses the grant by then anyway."""
    from gateway.hosted_room_peer import issue_room_grant, room_grant_hard_expiry
    from tui_gateway.hosted_room_driver import HostedRoomRuntime
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
    service, db, now = gateway.service, gateway.db.db_path, [time.time()]
    service.runtime.clock = lambda: now[0]

    class Member:
        def __init__(self, error=None):
            self.error, self.calls = error, 0

        def revoke_grant(self, *, grant):
            self.calls += 1
            if self.error is not None:
                raise self.error
            return {'revoked': True}

    members = {
        'unknown': Member(PeerRunsHTTPError('rejected', status_code=401, error_code='invalid_room_grant')),
        'revoked': Member(PeerRunsHTTPError('revoked', status_code=403, error_code='room_reauthorization_required')),
        'refused': Member(PeerRunsHTTPError('refused', status_code=403, error_code='permission_denied')),
        'expiring': Member()}
    grants = {member_id: issue_room_grant(
        b's' * 32, grant_id=f'grant-{member_id}', room_id='ended', home_install_id='install:home',
        authority_gateway_id='install:home', authority_epoch=1, member_id=member_id,
        target_install_id='install:target', target_profile='default', issued_at=now[0], ttl_seconds=600,
        status_ttl_seconds=600 if member_id == 'expiring' else 3600) for member_id in members}
    assert room_grant_hard_expiry(grants['expiring']) == now[0] + 600
    assert room_grant_hard_expiry('not-a-grant', now=now[0]) == now[0] + 30 * 24 * 60 * 60
    for member_id, client in members.items():
        service.owe_grant_revoke(room_id='ended', member_id=member_id, target_url='http://127.0.0.1:9',
                                 grant=grants[member_id], client=client)
    assert [row[1] for row in _owed(db)] == ['expiring', 'refused', 'revoked', 'unknown']

    now[0] += 601
    assert sum(service.retry_owed_revokes() for _ in range(3)) == 2
    assert {member_id: client.calls for member_id, client in members.items()} == {
        'unknown': 1, 'revoked': 1, 'refused': 1, 'expiring': 0}
    assert _owed(db) == [('ended', 'refused', 1)]
    error = str(HostedRoomRuntime.status(service.runtime)['last_error'])
    assert 'refused' in error and not any(grant in error for grant in grants.values())
    # Backoff doubles from the first retry; the refused revoke is tried again once due.
    now[0] += 4
    assert service.retry_owed_revokes() == 0 and members['refused'].calls == 1
    now[0] += 1
    assert service.retry_owed_revokes() == 0 and members['refused'].calls == 2
    assert _owed(db) == [('ended', 'refused', 2)]


@pytest.mark.asyncio
async def test_disband_and_registration_never_leave_a_route_behind(gateway, monkeypatch, linked_peer):
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient
    _server, url, catalog, room = linked_peer
    probe = PeerRunsHTTPClient.probe

    # Disband completes while the probe is in flight: the route is never published.
    def probe_then_disband(client, *, grant):
        result = probe(client, grant=grant)
        asyncio.run(call(gateway.owner, 'groups.disband', room_id='linked'))
        return result
    monkeypatch.setattr(PeerRunsHTTPClient, 'probe', probe_then_disband)
    early_grant, early = await _seat(gateway, room, url, catalog)
    assert early == 'invalid_params'
    monkeypatch.setattr(PeerRunsHTTPClient, 'probe', probe)
    # Its gateway saw that grant during the probe: the home owes its revoke and pays it.
    assert _owed(gateway.db.db_path) == [('linked', 'reviewer', 0)]
    assert await asyncio.to_thread(gateway.service.retry_owed_revokes) == 1
    assert await _grant_error(url, early_grant) == 'room_reauthorization_required'


@pytest.mark.asyncio
@pytest.mark.parametrize("stopping", [False, True], ids=["running", "stopping"])
async def test_disband_waits_for_a_registration_even_while_the_driver_stops(gateway, monkeypatch, linked_peer, stopping):
    """The room driver stopping must not take Disband off the registration lock: a route published
    meanwhile is either refused by the room's tombstone or dropped by Disband with its revoke owed,
    never left behind."""
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient
    probe = PeerRunsHTTPClient.probe
    _server, url, catalog, room = linked_peer
    grant = (await invite(gateway, room))['grant']
    race, save, running = {}, gateway.service._save_link, gateway.service.runtime.status

    contended, lock = threading.Event(), threading.Lock()

    class PublicationLock:
        def __enter__(self):
            if not lock.acquire(blocking=False):
                contended.set()
                assert lock.acquire(timeout=10)

        def __exit__(self, *args):
            lock.release()

    def disband():
        try:
            race['disband'] = asyncio.run(call(gateway.owner, 'groups.disband', room_id='linked'))
        finally:
            contended.set()  # An unlocked Disband must also release the test's barrier.

    def stop_driver_then_save(**link):
        if stopping:
            gateway.service.runtime.status = lambda: {'running': False, 'stopping': True}
        race['thread'] = threading.Thread(target=disband)
        race['thread'].start()
        assert contended.wait(10)
        race['during_save'] = race.get('disband')
        return save(**link)
    monkeypatch.setattr(gateway.service, 'peer_route_lock', PublicationLock())
    monkeypatch.setattr(gateway.service, '_save_link', stop_driver_then_save)
    registered = await call(gateway.owner, 'groups.peer.register', room_id='linked', member_id='reviewer',
                            target_url=url, target_profile='default', grant=grant, catalog=catalog)
    await asyncio.to_thread(race['thread'].join, 30)

    assert registered['registered'], registered
    assert race['during_save'] is None, race
    # The route was published first; Disband then revokes it at the member gateway and drops it.
    assert 'tombstone' in race['disband'], race
    assert hosted_room_links.load_room_links(gateway.db.db_path) == () and gateway.service.peer_routes == {}
    assert _owed(gateway.db.db_path) == []
    assert await _grant_error(url, grant) == 'room_reauthorization_required'
    gateway.service.runtime.status = running

    # An invitation still in flight when Disband returned lands once the room has ended and its
    # stored grant's revoke was paid. That grant is owed too, at the pinned endpoint only, unprobed.
    probes = []
    monkeypatch.setattr(PeerRunsHTTPClient, 'probe', lambda client, *, grant: probes.append(grant) or {})
    after, refused = await _seat(gateway, room, url, catalog)
    assert refused == 'invalid_params', refused  # the room has ended
    _, refused = await _seat(gateway, room, 'http://127.0.0.1:9', catalog)
    assert refused == 'peer_target_mismatch' and probes == []
    assert _owed(gateway.db.db_path) == [('linked', 'reviewer', 0)]
    monkeypatch.setattr(PeerRunsHTTPClient, 'probe', probe)
    assert await _grant_error(url, after) is None
    assert await asyncio.to_thread(gateway.service.retry_owed_revokes) == 1
    assert await _grant_error(url, after) == 'room_reauthorization_required'


async def _deferred_unreceipted_peer_turn(gateway, monkeypatch, linked_peer):
    """A member's gateway admitted a turn, the reply was lost, the home restarted before it saved
    the receipt, and the turn was deferred while the member stayed unreachable."""
    from gateway import hosted_room_driver as driver
    from gateway.platforms import api_server_runs
    from tui_gateway.hosted_room_peer_transport import build_member_dispatch

    async def inert(adapter, run, **kwargs):
        pass
    monkeypatch.setattr(api_server_runs, '_execute_run', inert)
    _server, url, catalog, room = linked_peer
    grant = (await invite(gateway, room))['grant']
    assert (await call(gateway.owner, 'groups.peer.register', room_id='linked', member_id='reviewer',
                       target_url=url, target_profile='default', grant=grant, catalog=catalog))['registered']
    assert (await call(gateway.owner, 'groups.send', room_id='linked', event_id='ask',
                       payload={'text': '@reviewer review this', 'thread_id': 'thread'}))['accepted']
    db, service = gateway.db.db_path, gateway.service
    task, = driver.list_tasks(db, room_id='linked', status='queued')
    identity, binding = task['identity'], service.bindings()[0]
    route, client = service.peer_routes[('linked', 'reviewer')], service.peer_clients[('linked', 'reviewer')]

    def earlier():
        return time.time() - 30
    crashed = driver.acquire_lease(db, room_id='linked', gateway_id=binding.gateway_id,
                                   authority_epoch=binding.authority_epoch, process_generation='crashed',
                                   ttl_seconds=1, clock=earlier)
    driver.start_task(db, identity, crashed, expected_cancel_generation=0, clock=earlier)
    await asyncio.to_thread(client.dispatch, grant=route.grant, dispatch=build_member_dispatch(
        binding=binding, route=route, room_id='linked', task_id=identity.task_id, target_profile='default',
        execution_generation=1, source_event_seq=task['payload']['source_event_seq'],
        prompt=task['payload']['prompt'], trace_id=route.trace_id).as_mapping())
    with gateway.db._lock:  # the reply is lost and the home restarts before it saves the receipt
        gateway.db._conn.execute('DELETE FROM hosted_room_remote_runs')
        gateway.db._conn.commit()
    service = gateway.authority.hosted_room_service = CanonicalHostedRoomService(gateway.authority, None)
    idle = service.runtime.status
    service.runtime.status = lambda: {**idle(), 'running': True, 'stopping': False}
    lease = service.runtime._ensure_lease(binding)
    driver.recover_room(db, lease, clock=time.time)
    recovered = driver.get_task(db, identity)
    service.runtime._fenced(driver.defer_indeterminate_task, None, recovered, lease, reason='member_unavailable')

    return service, db, identity


@pytest.mark.asyncio
async def test_a_deferred_peer_turn_offers_retry_on_its_own_generation(gateway, monkeypatch, linked_peer, tmp_path):
    """A deferred peer turn without a receipt offers Retry, which asks for that exact generation
    and never sends another; Discard is offered too (P2-3), and refused once Retry is watching
    the run again."""
    import sqlite3
    _service, _db, identity = await _deferred_unreceipted_peer_turn(gateway, monkeypatch, linked_peer)
    actions = (await call(gateway.owner, 'groups.state', room_id='linked'))['driver_status']['pending_actions']
    assert [(a['kind'], a['member_id'], a['task_id'], a['execution_generation']) for a in actions] == [
        ('retry', 'reviewer', identity.task_id, 1), ('discard', 'reviewer', identity.task_id, 1)], actions
    retried = await call(gateway.owner, 'groups.retry', room_id='linked', member_id='reviewer',
                         task_id=identity.task_id, execution_generation=1)
    assert isinstance(retried, dict) and retried['retried'], retried
    assert retried['task']['execution_generation'] == 1, retried
    # The member still runs generation 1, so the turn is watched again, at that generation.
    assert retried['task']['status'] == 'indeterminate', retried
    with sqlite3.connect(tmp_path / 'runs.db') as runs:
        keys = [k for (k,) in runs.execute('SELECT idempotency_key FROM run_idempotency')]
    assert keys == [f'room:{identity.task_id}:1'], keys
    assert 'discard' not in {a['kind'] for a in
                             (await call(gateway.owner, 'groups.state', room_id='linked'))['driver_status']['pending_actions']}
    assert await call(gateway.owner, 'groups.discard', room_id='linked', member_id='reviewer',
                      task_id=identity.task_id, execution_generation=1) == 'unsupported_operation'


@pytest.mark.asyncio
async def test_discard_keeps_a_deferred_peer_turn_while_its_member_is_down(gateway, monkeypatch, linked_peer):
    """Without a receipt nothing proves what the member's gateway did with the turn, so while it is
    unreachable Discard keeps the turn, as Disband does, and says why."""
    from gateway import hosted_room_driver as driver
    server = linked_peer[0]
    service, db, identity = await _deferred_unreceipted_peer_turn(gateway, monkeypatch, linked_peer)
    await server.close()
    assert await call(gateway.owner, 'groups.discard', room_id='linked', member_id='reviewer',
                      task_id=identity.task_id, execution_generation=1) == 'peer_unreachable'
    assert driver.get_task(db, identity)['status'] == 'deferred'
    with pytest.raises(RuntimeError, match='still stopping'):
        await asyncio.to_thread(service.stop_room, 'linked', cancel_id='d', require_acknowledged=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('linked_peer', [True], indirect=True)
async def test_discard_of_an_admitted_peer_turn_stops_its_run_before_disband(gateway, monkeypatch, linked_peer):
    """The member's gateway admitted the turn (its run is queued there) and answers. Discard finds
    that run under the turn's own key and settles the turn only once the member confirms it stopped,
    so Disband never tombstones the room while the member's run still executes."""
    from gateway import hosted_room_driver as driver
    _service, db, identity = await _deferred_unreceipted_peer_turn(gateway, monkeypatch, linked_peer)
    adapter = gateway.adapter
    run_id, = list(adapter._run_statuses)
    assert adapter._run_statuses[run_id]['status'] == 'queued'
    control = dict(room_id='linked', member_id='reviewer', task_id=identity.task_id, execution_generation=1)
    # The member's gateway is asked to stop that run; until it confirms, the turn holds.
    assert await call(gateway.owner, 'groups.discard', **control) == 'session_busy'
    assert adapter._run_statuses[run_id]['status'] == 'stopping'
    assert driver.get_task(db, identity)['status'] == 'deferred'
    assert await call(gateway.owner, 'groups.disband', room_id='linked') == 'room_stop_pending'
    adapter._set_run_status(run_id, 'cancelled')  # the member's run ends
    discarded = await call(gateway.owner, 'groups.discard', **control)
    assert isinstance(discarded, dict) and discarded['discarded'], discarded
    assert driver.get_task(db, identity)['status'] == 'cancelled'
    assert 'tombstone' in await call(gateway.owner, 'groups.disband', room_id='linked')


@pytest.mark.asyncio
async def test_register_sends_the_grant_only_to_the_pinned_endpoint(gateway, monkeypatch, linked_peer):
    """The pinned catalog's digest covers the endpoint its gateway advertises. Any other URL is
    refused before the grant leaves the home, so a host that repeats that gateway's answers is
    never bound in its place and never sees its grant."""
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient
    server, url, catalog, room = linked_peer
    seen, answer = [], {}

    async def echo(request):
        seen.append(request.headers.get('Authorization'))
        return web.json_response(answer)
    echo_app = web.Application()
    echo_app.router.add_route('*', '/{tail:.*}', echo)
    echo_server = TestServer(echo_app)
    await echo_server.start_server()
    try:
        grant = (await invite(gateway, room))['grant']
        answer.update(await asyncio.to_thread(PeerRunsHTTPClient(base_url=url, api_key='').probe, grant=grant))
        register = dict(room_id='linked', member_id='reviewer', target_profile='default', grant=grant, catalog=catalog)

        echo_url = str(echo_server.make_url('')).rstrip('/')
        assert await call(gateway.owner, 'groups.peer.register', **register, target_url=echo_url) == 'peer_target_mismatch'
        assert seen == [] and gateway.service.peer_routes == {}

        assert (await call(gateway.owner, 'groups.peer.register', **register, target_url=url))['registered']
        await server.close()
        assert await call(gateway.owner, 'groups.peer.register', **register, target_url=url) == 'peer_unreachable'
    finally:
        await echo_server.close()


async def _seat(gateway, room, url, catalog, **invite_params):
    """Invite on the member gateway and register that grant on the home; return the grant."""
    grant = (await call(gateway.owner, 'groups.peer.invite', room_id=room['room_id'], member_id='reviewer',
                        home_install_id=room['authority_gateway_id'],
                        authority_gateway_id=room['authority_gateway_id'],
                        authority_epoch=room['authority_epoch'], **invite_params))['grant']
    registered = await call(gateway.owner, 'groups.peer.register', room_id=room['room_id'], member_id='reviewer',
                            target_url=url, target_profile='default', grant=grant, catalog=catalog)
    return grant, registered


async def _grant_error(url, grant):
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError
    try:
        await asyncio.to_thread(PeerRunsHTTPClient(base_url=url, api_key='').probe, grant=grant)
    except PeerRunsHTTPError as exc:
        return exc.error_code
    return None


def test_paying_owed_revokes_holds_the_room_supervisor_only_briefly(gateway):
    """A member gateway that accepts the connection and never answers holds the worker cycle that
    pays its revoke. Each cycle pays one owed revoke, briefly, and its next attempt is timed from
    when that attempt gave up, so the room supervisor is never held for long."""
    import socket
    import sqlite3
    db, service = gateway.db.db_path, gateway.service
    held, now = [], [time.time()]
    service.runtime.clock = lambda: now[0]
    with socket.create_server(('127.0.0.1', 0)) as silent:
        silent.settimeout(30)
        port = silent.getsockname()[1]
        def accept():
            held.append(silent.accept())
            now[0] += 100.0  # Completion must schedule backoff from after the attempt began.
        worker = threading.Thread(target=accept, daemon=True)
        worker.start()
        for member_id in ('first', 'second'):
            service.owe_grant_revoke(room_id='ended', member_id=member_id, target_url=f'http://127.0.0.1:{port}',
                                     grant=f'grant-for-{member_id}')
        now[0] = time.time()
        started = time.monotonic()
        assert service.retry_owed_revokes() == 0
        elapsed = time.monotonic() - started
        worker.join(10)
        assert not worker.is_alive()
    assert elapsed < 4, elapsed
    with sqlite3.connect(db) as conn:
        rows = conn.execute('SELECT member_id, attempts, next_attempt_at FROM hosted_room_owed_revokes '
                            'ORDER BY member_id').fetchall()
    attempted = [row for row in rows if row[1] == 1]
    assert len(attempted) == 1 and len(rows) == 2, rows
    assert attempted[0][2] == now[0] + 5.0, attempted
    for connection, _ in held:
        connection.close()


@pytest.mark.asyncio
async def test_a_refresh_of_the_previous_grant_never_writes_over_a_re_seat(gateway, monkeypatch, linked_peer):
    """A turn renews the route's grant while an open room re-seats the member. The renewal belongs
    to the grant it renewed: once the route holds the re-seated grant, the renewal does not replace it."""
    from tui_gateway.hosted_room_driver import HostedRoomBinding
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient
    from tui_gateway.hosted_room_peer_transport import build_member_dispatch
    _server, url, catalog, room = linked_peer
    expiring, registered = await _seat(gateway, room, url, catalog, ttl_seconds=120)
    assert registered['registered'], registered
    service, key, reseated = gateway.service, ('linked', 'reviewer'), []
    refresh = PeerRunsHTTPClient.refresh_grant

    def refresh_while_reseating(client, **kwargs):
        renewed = refresh(client, **kwargs)
        grant, result = asyncio.run(_seat(gateway, room, url, catalog))
        assert result['registered'], result
        reseated.append(grant)
        return renewed
    monkeypatch.setattr(PeerRunsHTTPClient, 'refresh_grant', refresh_while_reseating)
    binding = HostedRoomBinding('linked', room['authority_gateway_id'], room['authority_epoch'])
    route = service.peer_routes[key]
    transport = service._resolve_member_transport(
        binding, {'payload': {'target_member_id': 'reviewer', 'source_event_seq': 1}})
    dispatch = build_member_dispatch(
        binding=binding, route=route, room_id='linked', task_id='dtask:renew', target_profile='default',
        execution_generation=1, source_event_seq=1, prompt='review', trace_id=route.trace_id).as_mapping()
    sent = await asyncio.to_thread(transport.client._refresh_grant, {'dispatch': dispatch, 'grant': expiring})
    assert sent['grant'] not in (expiring, *reseated)  # this turn still goes out on its renewal
    stored, = hosted_room_links.load_room_links(gateway.db.db_path)
    assert service.peer_routes[key].grant == stored.grant == reseated[0]
