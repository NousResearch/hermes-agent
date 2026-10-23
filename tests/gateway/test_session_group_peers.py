"""Cross-gateway members of gateway-owned Group Chats: honest capability, operator-only grants,
pinned registration, and Disband that never leaves a grant behind."""
import pytest_asyncio
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
    return reply.get('result', reply.get('error', {}).get('message'))


async def serve(gateway, monkeypatch):
    app = web.Application()
    for method, path, handler in api_server_room_grants._http_routes(gateway.adapter):
        app.router.add_route(method, path, handler)
    app.router.add_post('/v1/runs', gateway.adapter._handle_runs)
    app.router.add_get('/v1/runs/{run_id}', gateway.adapter._handle_get_run)
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


@pytest_asyncio.fixture
async def linked_peer(gateway, monkeypatch):
    server, url = await serve(gateway, monkeypatch)
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
    assert {'groups.peer.register', 'groups.peer.invite', 'groups.peer.revoke'} <= set(capabilities['methods'])
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
async def test_disband_and_registration_never_leave_a_route_behind(gateway, monkeypatch, linked_peer):
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError
    _server, url, catalog, room = linked_peer
    probe = PeerRunsHTTPClient.probe

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
    early_grant, early = await register('early')
    assert early == 'invalid_params'
    monkeypatch.setattr(PeerRunsHTTPClient, 'probe', probe)
    # Its gateway saw that grant during the probe: the home revokes it rather than drop it.
    with pytest.raises(PeerRunsHTTPError) as dropped:
        await asyncio.to_thread(PeerRunsHTTPClient(base_url=url, api_key='').probe, grant=early_grant)
    assert dropped.value.error_code == 'room_reauthorization_required'

    # Disband starts while the route is being published: it waits, then revokes the new grant.
    race, save = {}, gateway.service._save_link

    contended, lock = threading.Event(), threading.Lock()

    class PublicationLock:
        def __enter__(self):
            if not lock.acquire(blocking=False):
                contended.set()
                assert lock.acquire(timeout=10)

        def __exit__(self, *args):
            lock.release()

    def disband_late():
        try:
            race['disband'] = asyncio.run(call(gateway.owner, 'groups.disband', room_id='late'))
        finally:
            contended.set()

    def save_then_race(**link):
        race['thread'] = threading.Thread(target=disband_late)
        race['thread'].start()
        assert contended.wait(10)
        return save(**link)
    monkeypatch.setattr(gateway.service, 'peer_route_lock', PublicationLock())
    monkeypatch.setattr(gateway.service, '_save_link', save_then_race)
    grant, registered = await register('late')
    assert registered['registered'], registered
    await asyncio.to_thread(race['thread'].join, 30)
    assert 'tombstone' in race['disband'], race

    assert gateway.service.peer_routes == {} and hosted_room_links.load_room_links(gateway.db.db_path) == ()
    with pytest.raises(PeerRunsHTTPError) as revoked:
        await asyncio.to_thread(PeerRunsHTTPClient(base_url=url, api_key='').probe, grant=grant)
    assert revoked.value.error_code == 'room_reauthorization_required'


@pytest.mark.asyncio
async def test_disband_waits_for_a_registration_even_while_the_driver_stops(gateway, monkeypatch, linked_peer):
    """The room driver stopping must not take Disband off the registration lock: a route published
    meanwhile is either refused by the room's tombstone or stops Disband, never left behind."""
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient
    _server, url, catalog, room = linked_peer
    room = await linked_room(gateway, catalog, room_id='stopping')
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
            race['disband'] = asyncio.run(call(gateway.owner, 'groups.disband', room_id='stopping'))
        finally:
            contended.set()

    def stop_driver_then_save(**link):
        gateway.service.runtime.status = lambda: {'running': False, 'stopping': True}
        race['thread'] = threading.Thread(target=disband)
        race['thread'].start()
        assert contended.wait(10)
        race['during_save'] = race.get('disband')
        return save(**link)
    monkeypatch.setattr(gateway.service, 'peer_route_lock', PublicationLock())
    monkeypatch.setattr(gateway.service, '_save_link', stop_driver_then_save)
    registered = await call(gateway.owner, 'groups.peer.register', room_id='stopping', member_id='reviewer',
                            target_url=url, target_profile='default', grant=grant, catalog=catalog)
    await asyncio.to_thread(race['thread'].join, 30)

    assert registered['registered'], registered
    assert race['during_save'] is None, race
    # The route was published first, and no driver is running to revoke its grant: Disband refuses.
    assert race['disband'] == 'runtime_coordination_required', race
    assert hosted_room_links.load_room_links(gateway.db.db_path)
    assert (await asyncio.to_thread(PeerRunsHTTPClient(base_url=url, api_key='').probe, grant=grant))['room_id']

    # With the driver back, Disband revokes the grant and ends the room.
    gateway.service.runtime.status = running
    assert 'tombstone' in await call(gateway.owner, 'groups.disband', room_id='stopping')
    assert hosted_room_links.load_room_links(gateway.db.db_path) == ()


@pytest.mark.asyncio
async def test_a_deferred_peer_turn_offers_retry_on_its_own_generation(gateway, monkeypatch, linked_peer, tmp_path):
    """A member's gateway admitted a turn, the reply was lost, the home restarted, and the turn was
    deferred while the member stayed unreachable. Retry is offered for it, asks for that exact
    generation, and never sends another; Discard of a member's uncertain turn stays unoffered."""
    import sqlite3
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

    actions = (await call(gateway.owner, 'groups.state', room_id='linked'))['driver_status']['pending_actions']
    assert [(a['kind'], a['member_id'], a['task_id'], a['execution_generation']) for a in actions] == [
        ('retry', 'reviewer', identity.task_id, 1)], actions
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
