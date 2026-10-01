"""Canonical cross-gateway members: honest capability, operator-only grants, pinned registration."""
import asyncio
from dataclasses import replace
import json
import threading
import time
from types import SimpleNamespace

import pytest
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
    return reply.get('result', reply.get('error', {}).get('message'))


async def serve(gateway, monkeypatch):
    app = web.Application()
    for method, path, handler in api_server_room_grants._http_routes(gateway.adapter):
        app.router.add_route(method, path, handler)
    app.router.add_post('/v1/runs', gateway.adapter._handle_runs)
    server = TestServer(app)
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
    assert capabilities['methods'][-3:] == ['groups.peer.register', 'groups.peer.invite', 'groups.peer.revoke']
    contract.GroupsCapabilitiesResult.model_validate(capabilities)
    monkeypatch.delenv('HERMES_ROOM_LINK_URL')
    contract.GroupsCapabilitiesResult.model_validate(asyncio.run(call(gateway.owner, 'groups.capabilities')))


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
        # Grant ids and lifetimes beyond ``ttl_seconds`` are the target's to choose.
        for params in ({**identity, 'grant_id': 'chosen'}, {**identity, 'status_ttl_seconds': 86400},
                       {**identity, 'authority_epoch': '1'}, {**identity, 'authority_epoch': True},
                       {**identity, 'authority_epoch': 0}, {**identity, 'authority_epoch': 2**63},
                       {**identity, 'member_id': ''}, {**identity, 'room_id': 'r' * 129},
                       {**identity, 'ttl_seconds': True}, {**identity, 'ttl_seconds': 30},
                       {**identity, 'ttl_seconds': float('nan')}, {**identity, 'ttl_seconds': float('inf')}):
            assert await call(gateway.owner, 'groups.peer.invite', **params) == 'invalid_params', params
        assert await call(gateway.owner, 'groups.peer.revoke', grant=7) == 'invalid_params'
        assert await call(gateway.owner, 'groups.peer.revoke', grant='not.a-grant') == 'invalid_room_grant'
        assert await call(gateway.owner, 'groups.peer.revoke', grant='g', room_id='linked') == 'invalid_params'
        link = room_link(authority)
        room = await linked_room(gateway, link['catalog'])
        base = dict(room_id=room['room_id'], member_id='reviewer', grant='grant', catalog=link['catalog'],
                    target_url='http://127.0.0.1:9', target_profile='default')
        contract.GroupsPeerRegisterParams.model_validate(base)
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
async def test_register_binds_only_the_pinned_peer_after_a_live_scoped_probe(gateway, monkeypatch):
    from gateway.hosted_room_peer import (
        catalog_mapping, decode_room_grant, gateway_room_grant_secret, issue_room_grant)
    server, url = await serve(gateway, monkeypatch)
    try:
        catalog = room_link(gateway.authority)['catalog']
        room = await linked_room(gateway, catalog)
        invitation = await invite(gateway, room)
        assert invitation['catalog'] == catalog and invitation['endpoint'] == catalog['endpoint']
        contract.GroupsPeerInviteResult.model_validate(invitation)
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
                               ({'grant': other_member}, 'peer_target_mismatch'),
                               ({'target_url': 'http://127.0.0.1:9'}, 'peer_unreachable')]:
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
        contract.GroupsPeerRegisterResult.model_validate(registered)
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
        contract.GroupsPeerRevokeResult.model_validate(revoked)
        assert await call(gateway.owner, 'groups.peer.register', **register) == 'room_reauthorization_required'
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_reregistered_route_replays_accepted_work_as_the_same_target_run(gateway, monkeypatch):
    from gateway.platforms import api_server_runs
    from tui_gateway.hosted_room_driver import HostedRoomBinding
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError
    from tui_gateway.hosted_room_peer_transport import build_member_dispatch

    async def inert(adapter, run, **kwargs):
        pass
    monkeypatch.setattr(api_server_runs, '_execute_run', inert)
    server, url = await serve(gateway, monkeypatch)
    try:
        catalog = room_link(gateway.authority)['catalog']
        room = await linked_room(gateway, catalog)
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

        # A route minted with a fresh identity would turn that replay into a new request.
        drifted = replace(second, trace_id='trace-' + 'f' * 32)
        fresh = PeerRunsHTTPClient(base_url=url, api_key='')
        with pytest.raises(PeerRunsHTTPError) as refused:
            await asyncio.to_thread(fresh.recover_dispatch, dispatch=dispatch(drifted), grant=second.grant)
        assert refused.value.status_code == 409 and refused.value.not_admitted
    finally:
        await server.close()


def test_disband_waits_for_the_driver_while_a_peer_route_is_stored(tmp_path, monkeypatch):
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
            hosted_room_links.save_room_link(db.db_path, hosted_room_links.make_stored_link(
                room_id='linked', member_id='reviewer', target_url=endpoint['url'], target_profile='default',
                grant='stored-grant', catalog=GatewayRoomCatalog.from_mapping(catalog),
                cancellation_scope_id='cancel-linked', trace_id='trace-linked'))
            assert await call(owner, 'groups.disband', room_id='linked') == 'runtime_coordination_required'
            hosted_rooms.delete_room_link_records(db.db_path, room_id='linked')
            assert 'tombstone' in await call(owner, 'groups.disband', room_id='linked')
        asyncio.run(probe())


@pytest.mark.asyncio
async def test_disband_and_registration_never_leave_a_route_behind(gateway, monkeypatch):
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError
    server, url = await serve(gateway, monkeypatch)
    probe = PeerRunsHTTPClient.probe
    try:
        catalog = room_link(gateway.authority)['catalog']

        async def register(room_id):
            room = await linked_room(gateway, catalog, room_id=room_id)
            grant = (await invite(gateway, room))['grant']
            result = await call(gateway.owner, 'groups.peer.register', room_id=room_id, member_id='reviewer',
                                target_url=url, target_profile='default', grant=grant, catalog=catalog)
            return grant, result

        # Disband completes while the probe is in flight: the route is never published.
        def probe_then_disband(client, *, grant):
            result = probe(client, grant=grant)
            asyncio.run(call(gateway.owner, 'groups.disband', room_id='early'))
            return result
        monkeypatch.setattr(PeerRunsHTTPClient, 'probe', probe_then_disband)
        assert (await register('early'))[1] == 'invalid_params'
        monkeypatch.setattr(PeerRunsHTTPClient, 'probe', probe)

        # Disband starts while the route is being published: it waits, then revokes the new grant.
        race, save = {}, gateway.service._save_link

        def disband_late():
            race['disband'] = asyncio.run(call(gateway.owner, 'groups.disband', room_id='late'))

        def save_then_race(**link):
            race['thread'] = threading.Thread(target=disband_late)
            race['thread'].start()
            time.sleep(.3)  # Room for Disband to overtake, were registration not holding it back.
            return save(**link)
        monkeypatch.setattr(gateway.service, '_save_link', save_then_race)
        grant, registered = await register('late')
        assert registered['registered'], registered
        await asyncio.to_thread(race['thread'].join, 30)
        assert 'tombstone' in race['disband'], race

        assert gateway.service.peer_routes == {} and hosted_room_links.load_room_links(gateway.db.db_path) == ()
        with pytest.raises(PeerRunsHTTPError) as revoked:
            await asyncio.to_thread(PeerRunsHTTPClient(base_url=url, api_key='').probe, grant=grant)
        assert revoked.value.error_code == 'room_reauthorization_required'
    finally:
        await server.close()
