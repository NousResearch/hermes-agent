"""Passive copies on the canonical surface: opt-in invitations, the operator's copy read, the home's publisher."""
import asyncio
import threading
from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from gateway import hosted_room_peer as peer
from gateway import hosted_room_replicas as replicas
from gateway import hosted_room_replication as publisher
from gateway import hosted_rooms as rooms
from gateway.config import Platform, PlatformConfig
from gateway.hosted_rooms_common import table_exists
from gateway.platforms import api_server_room_grants, api_server_runs
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
from gateway.session_controls import AuthorityConnection
from gateway.session_hosted_service import CanonicalHostedRoomService
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch
from tests.gateway.fixtures.passive_copy import HOME, MEMBERS, append, catalog, member
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient
from tui_gateway.hosted_room_peer_transport import PeerMemberRoute

IDENTITY = dict(room_id='room', home_install_id=HOME, authority_gateway_id=HOME, authority_epoch=1,
                member_id='reviewer')


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    """A canonical default-profile gateway with its API server, as a participant or a home."""
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HERMES_ROOM_LINK_URL', 'http://127.0.0.1:9')
    db = SessionDB(home / 'state.db')
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={'key': 'participant-api-key'}))
    adapter._run_idempotency_store.close()
    adapter._run_idempotency_store = RunIdempotencyStore(str(tmp_path / 'runs.db'))
    authority = SimpleNamespace(profile_id=str(home), instance_id='owner', db=db, events={}, sessions={},
                                epoch=begin_runtime_epoch(db, instance_id='owner'),
                                runner=SimpleNamespace(adapters={Platform.API_SERVER: adapter}))
    service = authority.hosted_room_service = CanonicalHostedRoomService(authority, None)
    service.runtime.status = lambda: {'running': True, 'stopping': False}
    yield SimpleNamespace(home=home, db=db, adapter=adapter, authority=authority, service=service,
                          owner=AuthorityConnection(authority, object(), {'user_id': 'owner'}, operator=True))
    assert service.replication.stop(timeout=5)
    adapter._run_idempotency_store.close()
    db.close()


async def call(connection, method, **params):
    reply = await connection.dispatch({'id': 1, 'method': method, 'params': params})
    return reply.get('result', reply.get('error', {}).get('message'))


def permissions(grant):
    return peer.decode_room_grant(peer.gateway_room_grant_secret(), grant, permission='status')['permissions']


@pytest.mark.asyncio
async def test_the_canonical_invitation_adds_a_copy_only_when_asked(gateway):
    ordinary = await call(gateway.owner, 'groups.peer.invite', **IDENTITY)
    assert permissions(ordinary['grant']) == ['approve', 'dispatch', 'status', 'stop']
    copying = await call(gateway.owner, 'groups.peer.invite', **IDENTITY, replication=True)
    assert permissions(copying['grant']) == ['approve', 'dispatch', 'replicate', 'status', 'stop']
    evidence = await call(gateway.owner, 'groups.peer.invite', **IDENTITY, replication=True, work_records=True)
    assert permissions(evidence['grant']) == ['approve', 'dispatch', 'replicate', 'status', 'stop', 'work_records']
    passive = await call(gateway.owner, 'groups.peer.invite', **IDENTITY, replication=True, passive_only=True)
    assert permissions(passive['grant']) == ['replicate', 'status']
    passive = await call(gateway.owner, 'groups.peer.invite', **IDENTITY, replication=True, work_records=True,
                         passive_only=True)
    assert permissions(passive['grant']) == ['replicate', 'status', 'work_records']
    for flags in ({'replication': 'yes'}, {'passive_only': True}, {'replication': True, 'passive_only': 1},
                  {'work_records': True}, {'replication': True, 'work_records': 'yes'}):
        assert await call(gateway.owner, 'groups.peer.invite', **IDENTITY, **flags) == 'invalid_params', flags
    member_only = AuthorityConnection(gateway.authority, object(), {'user_id': 'owner'})
    assert await call(member_only, 'groups.peer.invite', **IDENTITY, replication=True) == 'permission_denied'


@pytest.mark.asyncio
async def test_a_passive_only_participant_can_be_copied_to_but_never_runs_work(gateway, tmp_path):
    gateway.adapter.gateway_runner = SimpleNamespace(session_authority=gateway.authority)
    source = tmp_path / 'home.db'
    rooms.create_room(source, room_id='room', name='Workshop', members=[
        MEMBERS[0], {**member('reviewer'), 'target': {**member('reviewer')['target'],
                                                      'installation_id': rooms.local_authority_gateway_id()}}],
        authority_gateway_id=HOME)
    append(source, 'hello')
    grant = (await call(gateway.owner, 'groups.peer.invite', **IDENTITY, replication=True,
                        passive_only=True))['grant']
    app = web.Application()
    for method, path, handler in (*api_server_room_grants._http_routes(gateway.adapter),
                                  *api_server_runs._http_routes(gateway.adapter)):
        app.router.add_route(method, path, handler)
    server = TestServer(app)
    await server.start_server()
    try:
        client = PeerRunsHTTPClient(base_url=str(server.make_url('')).rstrip('/'), api_key='', timeout_seconds=5)
        room = rooms.room_state(source, room_id='room')
        await asyncio.to_thread(client.replicate_page, grant=grant, room_id='room', room_name='Workshop',
                                members=room['members'], page=rooms.read_events(source, room_id='room'))
        copy = await call(gateway.owner, 'groups.replica_state', room_id='room')
        assert (copy['last_seq'], copy['safety_status']) == (1, 'passive')
        assert copy['work_records']['availability'] == 'not_retained' and not copy['work_records']['source_loss_safe']
        async with aiohttp.ClientSession() as http:
            for path, payload in [('/v1/runs', {'input': 'must not run'}),
                                  ('/v1/runs/retained-run/approval', {'decision': 'allow'}),
                                  ('/v1/runs/retained-run/stop', {}),
                                  ('/v1/room-members/grants/refresh', {})]:
                async with http.post(server.make_url(path), json=payload,
                                     headers={'Authorization': f'HermesRoom {grant}'}) as denied:
                    assert denied.status == 401, (path, await denied.text())
                    assert (await denied.json())['error']['code'] == 'invalid_room_grant'
        assert not gateway.adapter._active_run_tasks and not gateway.adapter._run_statuses
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_the_copy_is_an_operator_read_on_the_default_profile(gateway, tmp_path):
    reader = AuthorityConnection(gateway.authority, object(), {'user_id': 'owner', 'capabilities': ['session:read']})
    member_only = AuthorityConnection(gateway.authority, object(), {'user_id': 'owner'})
    assert await call(reader, 'groups.replica_state', room_id='room') == 'permission_denied'
    assert await call(member_only, 'groups.replica_state', room_id='room') == 'permission_denied'
    assert await call(gateway.owner, 'groups.replica_state', room_id='room') == 'not_found'
    assert await call(gateway.owner, 'groups.replica_state', room_id='room', extra=1) == 'invalid_params'
    named = gateway.home / 'profiles' / 'helper'
    named.mkdir(parents=True)
    named_db = SessionDB(named / 'state.db')
    try:
        helper = SimpleNamespace(profile_id=str(named), instance_id='owner', db=named_db, events={}, sessions={},
                                 epoch=begin_runtime_epoch(named_db, instance_id='owner'), runner=None)
        operator = AuthorityConnection(helper, object(), {'user_id': 'owner'}, operator=True)
        assert await call(operator, 'groups.replica_state', room_id='room') == 'default_profile_required'
    finally:
        named_db.close()


def _participant_room(gateway):
    created = asyncio.run(call(gateway.owner, 'groups.create', room_id='room', name='Workshop', members=[
        MEMBERS[0], member('reviewer', target='install:participant')]))
    assert created['room']['room_id'] == 'room', created


def _registration(gateway, *, grant_id, permissions=('approve', 'dispatch', 'status', 'stop', 'replicate')):
    """What ``groups.peer.register`` hands the service for a member on a participant gateway."""
    participant, local = catalog('install:participant'), rooms.local_authority_gateway_id()
    grant = peer.issue_room_grant(
        b'p' * 32, grant_id=grant_id, room_id='room', home_install_id=local, authority_gateway_id=local,
        authority_epoch=1, member_id='reviewer', target_install_id='install:participant', target_profile='default',
        permissions=permissions, execution_policy_digest=participant.execution_policy.policy_digest, ttl_seconds=600)
    route = PeerMemberRoute(
        home_install_id=local, member_id='reviewer', target_install_id='install:participant',
        target_profile='default', capability_digest=participant.catalog_digest,
        execution_policy_digest=participant.execution_policy.policy_digest,
        cancellation_scope_id='cancel-room', trace_id='trace-room', grant=grant)
    client = PeerRunsHTTPClient(base_url='http://127.0.0.1:9', api_key='', receipt_db_path=gateway.service.db_path)
    client.revoke_grant_exact = lambda **kwargs: {'revoked': True}
    return dict(room_id='room', member_id='reviewer', route=route, client=client,
                target_url='http://127.0.0.1:9', catalog=participant)


def _quiet_runtime(service):
    """Keep these lifecycle tests independent of the execution driver's work loop."""
    service.runtime = SimpleNamespace(
        start=lambda: None, stop=lambda **_: True, wakeup=lambda: None,
        status=lambda: {'running': True, 'stopping': False, 'blocked_rooms': []})


def test_an_idle_home_runs_no_publisher_until_a_copy_route_is_registered(gateway):
    service = gateway.service
    _quiet_runtime(service)
    _participant_room(gateway)
    service.start()
    assert service.replication.status()['workers'] == 0
    service.register_peer_route(**_registration(gateway, grant_id='member', permissions=('dispatch', 'status')))
    assert service.replication.status()['workers'] == 0
    with gateway.db._read_ctx() as conn:
        assert not table_exists(conn, publisher.ROUTES_TABLE)
    service.register_peer_route(**_registration(gateway, grant_id='member-with-copy'))
    assert service.replication.status()['workers'] == publisher.WORKERS
    state = asyncio.run(call(gateway.owner, 'groups.state', room_id='room'))['driver_status']['replication']
    assert (state['mode'], state['source_loss_safe']) == ('passive_async_copy', False)
    assert service.stop(timeout=5)
    assert service.replication.status()['workers'] == 0


def test_a_restarted_home_resumes_its_publisher_from_stored_routes(gateway):
    _quiet_runtime(gateway.service)
    _participant_room(gateway)
    gateway.service.register_peer_route(**_registration(gateway, grant_id='member-with-copy'))
    restarted = CanonicalHostedRoomService(gateway.authority, None)
    _quiet_runtime(restarted)
    try:
        restarted.start()
        assert restarted.replication.status()['workers'] == publisher.WORKERS
    finally:
        assert restarted.stop(timeout=5)


def test_copying_progresses_while_the_room_policy_lock_is_held(gateway, monkeypatch):
    service = gateway.service
    _quiet_runtime(service)
    _participant_room(gateway)
    received = threading.Event()

    def participant(request, *, timeout, **kwargs):
        received.set()
        raise TimeoutError('participant offline')

    monkeypatch.setattr('tui_gateway.hosted_room_peer_http._open_roomlink_url', participant)
    with service._policy_lock:
        service.register_peer_route(**_registration(gateway, grant_id='member-with-copy'))
        service.start()
        try:
            assert received.wait(5)
        finally:
            assert service.stop(timeout=5)


@pytest.mark.parametrize('failed_index', [0, 1])
def test_a_partial_worker_start_failure_keeps_controls_and_cleanup_working(gateway, monkeypatch, failed_index):
    service = gateway.service
    _quiet_runtime(service)
    _participant_room(gateway)
    service.register_peer_route(**_registration(gateway, grant_id='member-with-copy'))
    original_start = threading.Thread.start

    def fail_one(thread):
        if thread.name == f'hosted-room-replication-{failed_index}':
            raise RuntimeError("can't start new thread")
        return original_start(thread)

    with monkeypatch.context() as scoped:
        scoped.setattr(threading.Thread, 'start', fail_one)
        service.start()
        assert service.replication.status()['error'] == 'publisher_start_failed'
        assert asyncio.run(call(gateway.owner, 'groups.state', room_id='room'))['room']['room_id'] == 'room'
        assert service.stop(timeout=2)
    assert not any(thread.is_alive() for thread in service.replication._threads)
    service.start()
    try:
        assert service.replication.status()['error'] is None
        assert service.replication.status()['workers'] == publisher.WORKERS
    finally:
        assert service.stop(timeout=2)
