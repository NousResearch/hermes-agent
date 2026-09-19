"""Canonical outbound setup: real dispatcher/SQL, inert network and runtime."""
import asyncio
import copy
import hashlib
import json
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from gateway import hosted_room_links as links, hosted_rooms as rooms
from gateway.hosted_room_peer import GatewayRoomCatalog, catalog_mapping, issue_room_grant
from gateway.session_contract import Principal
from gateway.session_controls import AuthorityConnection
from gateway.session_hosted_service import CanonicalHostedRoomService
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError


@pytest.fixture
def setup_owner(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(rooms, 'local_authority_gateway_id', lambda: 'home')
    db = SessionDB(tmp_path / 'state.db')
    authority = SimpleNamespace(db=db, profile_id=str(tmp_path),
        epoch=begin_runtime_epoch(db, instance_id='test'))
    service = CanonicalHostedRoomService(authority, None)
    # No service/worker is started; this fixture supplies only readiness and wakeup.
    service.runtime = SimpleNamespace(status=lambda: {'running': True, 'stopping': False}, wakeup=lambda: None)
    authority.hosted_room_service = service
    service.authorize_room('alice', 'room', create=True)
    catalog = catalog_mapping(installation_id='peer', target_profile='ops', persistent_process=True)
    rooms.create_room(db.db_path, room_id='room', name='Room', authority_gateway_id='home', members=[
        {'member_id': 'remote', 'profile': 'ops', 'handle': 'remote', 'target': {
            'kind': 'peer', 'peer_id': 'peer', 'installation_id': 'peer', 'profile': 'ops',
            'capability_digest': catalog['catalog_digest']}}])
    grant = issue_room_grant(b'x' * 32, grant_id='supplied', room_id='room', home_install_id='home',
        authority_gateway_id='home', authority_epoch=1, member_id='remote', target_install_id='peer',
        target_profile='ops', execution_policy_digest=catalog['execution_policy']['policy_digest'])
    params = dict(request_id='setup-1', room_id='room', member_id='remote', target_url='https://peer.example',
        target_profile='ops', grant=grant, catalog=catalog, cancellation_scope_id='cancel', trace_id='trace')
    connection = object.__new__(AuthorityConnection)
    connection.authority = authority
    connection.actor = Principal('alice', str(tmp_path), frozenset({'session:control', 'session:read'}), 'transport')
    probe = dict(room_id='room', home_install_id='home', authority_gateway_id='home', authority_epoch=1,
        member_id='remote', target_profile='ops', catalog=catalog)
    calls = []
    def network(client, *, grant):
        calls.append(('probe', client.base_url, grant))
        return copy.deepcopy(probe)
    monkeypatch.setattr(PeerRunsHTTPClient, 'probe', network)
    state = SimpleNamespace(db=db, authority=authority, service=service, connection=connection,
        params=params, probe=probe, calls=calls, network=network)
    try:
        yield state
    finally:
        db.close()


async def register(s, **changes):
    return await s.connection.dispatch({'id': 'rpc', 'method': 'groups.peer.register', 'params': {**s.params, **changes}})


def stored(s):
    return links.load_room_link(s.db.db_path, room_id='room', member_id='remote')


def sql(s, query, values=()):
    return s.db._execute_write(lambda conn: conn.execute(query, values))


@pytest.mark.asyncio
@pytest.mark.parametrize('drift', [None, 'payload', 'actor', 'new_owner', 'profile', 'member', 'epoch', 'route'])
async def test_completed_registration_replay_requires_exact_binding(setup_owner, drift):
    s = setup_owner
    result = await register(s)
    assert result.get('result', {}).get('registered') is True, result
    route = stored(s)
    assert route.grant == s.params['grant']
    assert route.target_url == s.params['target_url']
    assert route.catalog == GatewayRoomCatalog.from_mapping(s.params['catalog'])
    changes = {}
    if drift == 'payload':
        changes['trace_id'] = 'different'
    elif drift == 'actor':
        s.connection.actor = replace(s.connection.actor, subject='bob')
    elif drift == 'new_owner':
        s.connection.actor = replace(s.connection.actor, subject='bob')
        sql(s, "UPDATE state_meta SET value='bob' WHERE key='gateway.hosted.owner.v1:room'")
    elif drift == 'profile':
        s.connection.actor = replace(s.connection.actor, profile_id=str(s.db.db_path) + '.foreign')
    elif drift == 'member':
        changes['member_id'] = 'other'
    elif drift == 'epoch':
        s.authority.epoch = begin_runtime_epoch(s.db, instance_id='replacement')
    elif drift == 'route':
        links.save_room_link(s.db.db_path, replace(route, target_url='https://winner.example'))
    replay = await register(s, **changes)
    assert ('error' in replay) is (drift is not None), replay
    if drift is None:
        assert replay['result'] == result['result']
    assert len(s.calls) == 1
    assert stored(s) == (replace(route, target_url='https://winner.example') if drift == 'route' else route)
    capabilities = await s.connection.dispatch({'method': 'groups.capabilities'}) if drift != 'profile' else None
    if capabilities and 'result' in capabilities:
        assert capabilities['result']['room_link']['enabled'] is False
        assert 'groups.peer.register' in capabilities['result']['methods']
        for method in ('groups.peer.invite', 'groups.peer.revoke', 'groups.peer.revoke_exact'):
            denied = await s.connection.dispatch({'method': method, 'params': {}})
            assert denied['error']['message'] == 'permission_denied'


@pytest.mark.asyncio
@pytest.mark.parametrize('deny', ['capability', 'actor', 'profile', 'missing', 'stopping', 'fence', 'quarantine',
    'unsupported', 'member', 'request_empty', 'request_large'])
async def test_registration_preflight_refuses_without_probe(setup_owner, deny):
    s = setup_owner
    changes = {}
    if deny == 'capability':
        s.connection.actor = replace(s.connection.actor, capabilities=frozenset({'session:read'}))
    elif deny == 'actor':
        s.connection.actor = replace(s.connection.actor, subject='bob')
    elif deny == 'profile':
        changes['profile'] = 'foreign'
    elif deny == 'missing':
        s.authority.hosted_room_service = None
    elif deny == 'stopping':
        s.service.runtime.status = lambda: {'running': True, 'stopping': True}
    elif deny == 'fence':
        s.service.begin_room_disband('room')
    elif deny == 'quarantine':
        sql(s, 'INSERT INTO hosted_room_quarantine(room_id,reason,detected_at) VALUES(?,?,?)', ('room','held',time.time()))
    elif deny == 'unsupported':
        changes['catalog'] = catalog_mapping(installation_id='peer', target_profile='ops', persistent_process=True, text=False)
    elif deny == 'member':
        changes['member_id'] = 'absent'
    else:
        changes['request_id'] = '' if deny == 'request_empty' else 'x' * 257
    result = await register(s, **changes)
    assert 'error' in result, result
    assert not s.calls
    assert stored(s) is None


@pytest.mark.asyncio
@pytest.mark.parametrize('drift', ['owner', 'runtime_epoch', 'room_epoch', 'member', 'fence', 'quarantine',
    'coordinator', 'probe_scope', 'probe_catalog', 'winner'])
async def test_registration_rechecks_authority_at_publication(setup_owner, monkeypatch, drift):
    s = setup_owner
    winner = None
    def network(client, *, grant):
        nonlocal winner
        result = s.network(client, grant=grant)
        if drift == 'owner':
            sql(s, "UPDATE state_meta SET value='bob' WHERE key='gateway.hosted.owner.v1:room'")
        elif drift == 'runtime_epoch':
            begin_runtime_epoch(s.db, instance_id='replacement')
        elif drift == 'room_epoch':
            sql(s, 'UPDATE hosted_rooms SET authority_epoch=2 WHERE room_id=?', ('room',))
        elif drift == 'member':
            sql(s, 'UPDATE hosted_rooms SET members_json=? WHERE room_id=?', ('[]', 'room'))
        elif drift == 'fence':
            s.service.begin_room_disband('room')
        elif drift == 'quarantine':
            sql(s, 'INSERT INTO hosted_room_quarantine(room_id,reason,detected_at) VALUES(?,?,?)', ('room','held',time.time()))
        elif drift == 'coordinator':
            s.authority.hosted_room_service = None
        elif drift == 'probe_scope':
            result['authority_epoch'] = True
        elif drift == 'probe_catalog':
            result['catalog'] = catalog_mapping(installation_id='other', target_profile='ops', persistent_process=True)
        else:
            winner = links.make_stored_link(room_id='room', member_id='remote', target_url='https://winner.example',
                target_profile='ops', grant=grant, catalog=GatewayRoomCatalog.from_mapping(s.params['catalog']),
                cancellation_scope_id='winner', trace_id='winner')
            links.save_room_link(s.db.db_path, winner)
        return result
    monkeypatch.setattr(PeerRunsHTTPClient, 'probe', network)
    result = await register(s)
    assert 'error' in result, result
    assert stored(s) == winner
    assert len(s.calls) == 1


@pytest.mark.asyncio
async def test_same_request_is_reserved_before_probe_and_failure_stays_pending(setup_owner, monkeypatch):
    s = setup_owner
    entered, release = threading.Event(), threading.Event()
    def network(client, *, grant):
        s.calls.append(('probe',))
        entered.set()
        assert release.wait(5)
        raise PeerRunsHTTPError('sensitive raw body ' + grant)
    monkeypatch.setattr(PeerRunsHTTPClient, 'probe', network)
    first = asyncio.create_task(register(s))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        second = await register(s)
        assert second.get('error', {}).get('message') == 'peer_setup_pending', second
    finally:
        release.set()
    result = await first
    assert result.get('error', {}).get('message') == 'peer_setup_pending', result
    assert (await register(s))['error']['message'] == 'peer_setup_pending'
    assert len(s.calls) == 1
    assert stored(s) is None
    assert s.params['grant'] not in json.dumps(result)


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['retire', 'save', 'lost_response', None])
async def test_replacement_preserves_original_credential_until_ack(setup_owner, monkeypatch, failure):
    s = setup_owner
    old = links.make_stored_link(room_id='room', member_id='remote', target_url='https://old.example',
        target_profile='ops', grant='old-exact-bearer', catalog=GatewayRoomCatalog.from_mapping(s.params['catalog']),
        cancellation_scope_id='old', trace_id='old')
    links.save_room_link(s.db.db_path, old)
    events = []
    def retire(client, *, grant):
        events.append((client.base_url, grant))
        assert stored(s) == old
        if failure == 'retire':
            raise PeerRunsHTTPError('unreachable')
        return {'revoked': True}
    monkeypatch.setattr(PeerRunsHTTPClient, 'revoke_grant_exact', retire)
    original = links.save_room_link
    def save(*args, **kwargs):
        assert events == [('https://old.example', old.grant)]
        if failure == 'save':
            raise OSError('disk failure')
        original(*args, **kwargs)
        if failure == 'lost_response':
            raise OSError('lost return after commit')
    monkeypatch.setattr(links, 'save_room_link', save)
    result = await register(s, expected_grant_sha256=hashlib.sha256(old.grant.encode()).hexdigest())
    assert events == [('https://old.example', old.grant)]
    if failure in {'retire', 'save'}:
        assert result.get('error', {}).get('message') == 'peer_setup_pending', result
        assert stored(s) == old
        assert (await register(s, expected_grant_sha256=hashlib.sha256(old.grant.encode()).hexdigest()))['error']['message'] == 'peer_setup_pending'
    else:
        assert stored(s).grant == s.params['grant']
        replay = await register(s, expected_grant_sha256=hashlib.sha256(old.grant.encode()).hexdigest())
        assert replay.get('result', {}).get('registered') is True, replay
    assert len(events) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('drift', ['owner', 'epoch', 'fence', 'quarantine', 'member', 'winner', 'receipt', 'expired'])
async def test_route_writer_revalidates_after_retirement_ack(setup_owner, monkeypatch, drift):
    from gateway import session_group_setup
    s = setup_owner
    old = links.make_stored_link(room_id='room', member_id='remote', target_url='https://old.example',
        target_profile='ops', grant='old-bearer', catalog=GatewayRoomCatalog.from_mapping(s.params['catalog']),
        cancellation_scope_id='old', trace_id='old')
    links.save_room_link(s.db.db_path, old)
    winner = replace(old, grant='winner-bearer', target_url='https://winner.example')
    retired = []
    def request(client, path, **kwargs):
        assert path == '/v1/room-members/grants/revoke-exact'
        retired.append((client.base_url, kwargs['room_grant']))
        return {'revoked': True}
    monkeypatch.setattr(PeerRunsHTTPClient, '_request', request)
    original = links.save_room_link
    def raced_save(*args, **kwargs):
        assert retired == [('https://old.example', old.grant)]
        mutations = {
            'owner': ("UPDATE state_meta SET value='bob' WHERE key='gateway.hosted.owner.v1:room'", ()),
            'epoch': ('UPDATE runtime_epoch SET epoch=epoch+1', ()),
            'member': ('UPDATE hosted_rooms SET members_json=? WHERE room_id=?', ('[]', 'room')),
            'receipt': ("UPDATE state_meta SET value='{}' WHERE key LIKE 'gateway.hosted.peer.setup.v1:%'", ()),
            'quarantine': ('INSERT INTO hosted_room_quarantine(room_id,reason,detected_at) VALUES(?,?,?)',
                           ('room', 'held', time.time())),
        }
        if drift in mutations:
            sql(s, *mutations[drift])
        elif drift == 'fence':
            s.service.begin_room_disband('room')
        elif drift == 'winner':
            original(s.db.db_path, winner)
        else:
            monkeypatch.setattr(session_group_setup, 'time', SimpleNamespace(time=lambda: time.time() + 86400))
        original(*args, **kwargs)
    monkeypatch.setattr(links, 'save_room_link', raced_save)
    result = await register(s)
    assert 'error' in result, result
    assert stored(s) == (winner if drift == 'winner' else old)
    assert retired == [('https://old.example', old.grant)]
    with s.db._read_ctx() as conn:
        receipts = [json.loads(row[0]) for row in conn.execute(
            "SELECT value FROM state_meta WHERE key LIKE 'gateway.hosted.peer.setup.v1:%'")]
    assert receipts and all(r.get('state') != 'completed' for r in receipts)
