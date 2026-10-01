"""A member keeps working past its grant's lifetime: the home renews it before it expires.

The home is a real canonical service driven by manual room cycles; the member's gateway is a
real API server adapter whose grant and run handlers answer in process. Time is a test clock.
"""
import asyncio
from dataclasses import replace
import json
import sqlite3
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import hosted_room_driver as driver, hosted_room_links as links, hosted_rooms
from gateway.hosted_room_peer import (
    GatewayRoomCatalog, decode_room_grant, gateway_room_grant_secret, issue_room_grant, room_grant_token_digest)
from gateway.platforms import api_server, api_server_room_grants
from gateway.run import _profile_runtime_scope
from gateway.session_authorities import SessionAuthorities
from gateway.session_authority import SessionAuthority
from gateway.session_hosted_service import CanonicalHostedRoomService
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch
from tests.tui_gateway.test_hosted_room_driver_runtime import FakeSessionRPC
from tui_gateway.hosted_room_driver import HostedRoomRuntime, room_session_title
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError
from tui_gateway.hosted_room_peer_transport import PeerMemberRoute

KEY = ('room', 'reviewer')
_HANDLERS = {
    '/v1/room-members/grants/refresh': '_handle_room_member_grant_refresh',
    '/v1/room-members/capabilities': '_handle_room_member_capabilities',
    '/v1/room-members/grants/revoke': '_handle_room_member_grant_revoke',
    '/v1/room-members/grants/revoke-exact': '_handle_room_member_grant_revoke_exact',
}


class Member(PeerRunsHTTPClient):
    """The member's gateway: the real room-member handlers, called in process."""

    def __init__(self, adapter, clock):
        super().__init__(base_url='https://member.example.test', api_key='', clock=lambda: clock[0])
        self.adapter, self.time = adapter, clock
        self.offline, self.on_refresh, self.runs = False, None, None
        self.offline_seconds = 0.0  # how long an unanswered request takes, within its timeout
        self.refreshes, self.issued, self.failures = [], [], []

    def _request(self, path, *, method='GET', body=None, headers=None, room_grant=None):
        assert self.timeout_seconds <= 30
        if path.endswith('/refresh'):
            self.refreshes.append(self.time[0])
        if self.offline:
            self.time[0] += min(self.timeout_seconds, self.offline_seconds)
            raise PeerRunsHTTPError('member offline', retryable=True, not_admitted=method == 'POST')
        if path.startswith('/v1/runs'):
            return self.runs(path, method=method, body=body, room_grant=room_grant)

        async def call():
            self.adapter._read_json_body = AsyncMock(return_value=(body or {}, None))
            request = SimpleNamespace(headers={'Authorization': f'HermesRoom {room_grant}'})
            return await getattr(api_server_room_grants, _HANDLERS[path])(
                self.adapter, request, _openai_error=api_server._openai_error,
                _api_request_profile=api_server._api_request_profile)
        response = asyncio.run(call())
        payload = json.loads(response.text)
        if response.status >= 400:
            self.failures.append((path, payload['error']['code']))
            raise PeerRunsHTTPError('member refused', status_code=response.status,
                                    error_code=payload['error']['code'])
        if path.endswith('/refresh'):
            self.issued.append(payload['grant'])
            if self.on_refresh:
                callback, self.on_refresh = self.on_refresh, None
                callback()
        return payload


@pytest.fixture
def renewal(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))

    def forbidden(*args, **kwargs):
        raise AssertionError('runtime threads are forbidden here')
    monkeypatch.setattr(HostedRoomRuntime, 'start', forbidden)
    clock = [time.time()]
    monkeypatch.setattr(time, 'time', lambda: clock[0])
    adapter = api_server.APIServerAdapter.__new__(api_server.APIServerAdapter)
    adapter._profile_scope = lambda _profile: _profile_runtime_scope(home)  # the member's own config
    member = Member(adapter, clock)
    with SessionDB(home / 'state.db') as db:
        runner = SimpleNamespace(_draining=False, session_authorities=SessionAuthorities(home))
        authority = SessionAuthority(runner, db=db, profile_id=str(home), instance_id='fixture',
                                     epoch=begin_runtime_epoch(db, instance_id='fixture'))
        runner.session_authorities.add(home, authority)
        service = CanonicalHostedRoomService(authority, None)
        authority.hosted_room_service = service
        service.local_profiles = lambda: ('default',)
        service.runtime.clock = lambda: clock[0]
        service.runtime._thread = SimpleNamespace(is_alive=lambda: True)
        service.authorize_room('alice', 'room', create=True)
        gateway = hosted_rooms.local_authority_gateway_id()
        invitation = api_server_room_grants._issue_invitation(adapter, {
            'room_id': 'room', 'home_install_id': gateway, 'authority_gateway_id': gateway,
            'authority_epoch': 1, 'member_id': 'reviewer', 'ttl_seconds': 3600,
            'status_ttl_seconds': 30 * 24 * 3600}, 'default')
        catalog = GatewayRoomCatalog.from_mapping(invitation['catalog'])
        pin = {'kind': 'peer', 'peer_id': 'member', 'installation_id': catalog.installation_id,
               'profile': 'default', 'capability_digest': catalog.catalog_digest}
        service.create_room(room_id='room', name='Renewal', members=[
            {'member_id': 'host', 'profile': 'default', 'handle': 'host'},
            {'member_id': 'reviewer', 'profile': 'default', 'handle': 'reviewer', 'target': pin}])
        route = PeerMemberRoute(home_install_id=gateway, member_id='reviewer',
                                target_install_id=catalog.installation_id, target_profile='default',
                                capability_digest=catalog.catalog_digest,
                                execution_policy_digest=catalog.execution_policy.policy_digest,
                                cancellation_scope_id='cancel-room', trace_id='trace-room',
                                grant=invitation['grant'])
        service.register_peer_route(room_id='room', member_id='reviewer', route=route, client=member,
                                    target_url=member.base_url, catalog=catalog)
        secret = gateway_room_grant_secret()
        yield SimpleNamespace(
            service=service, member=member, clock=clock, secret=secret, home=home, db=db, route=route,
            catalog=catalog, adapter=adapter, authority=authority, old=invitation['grant'],
            claims=decode_room_grant(secret, invitation['grant'], permission='dispatch'))
        service.runtime._thread = None


def cycle(r, n=1):
    for _ in range(n):
        r.service.runtime._run_cycle()


def stored(r, member='reviewer'):
    return next(l for l in links.load_room_links(r.service.db_path) if (l.room_id, l.member_id) == ('room', member))


def retired(r, grant):
    return hosted_rooms.room_grant_is_revoked(
        hosted_rooms.default_db_path(), claims=decode_room_grant(r.secret, grant, permission='status'),
        token_sha256=room_grant_token_digest(grant))


def test_idle_rooms_renew_on_a_bounded_cadence_and_back_off_while_offline(renewal):
    r = renewal
    with sqlite3.connect(r.service.db_path) as db:  # an unreadable route elsewhere in the room is skipped
        row = dict(zip([c[0] for c in db.execute('SELECT * FROM hosted_room_links').description],
                       db.execute('SELECT * FROM hosted_room_links').fetchone()))
        db.execute('INSERT INTO hosted_room_links VALUES (?,?,?,?,?,?,?,?,?,?,?)', (
            'room', 'host', 'http://invalid.example.test', *[row[c] for c in (
                'target_profile', 'grant', 'catalog_json', 'cancellation_scope_id', 'trace_id',
                'transport_security', 'status', 'updated_at')]))
    cycle(r)
    assert not r.member.refreshes  # a fresh one-hour grant is not due yet
    previous = r.old
    for _ in range(3):
        r.clock[0] += 3300  # no turns, no model: only the clock moves past each grant's lifetime
        cycle(r)
        current = r.service.peer_routes[KEY].grant
        claims = decode_room_grant(r.secret, current, permission='dispatch')
        assert claims['expires_at'] == r.clock[0] + 3600, r.member.failures
        assert claims['status_expires_at'] == r.claims['status_expires_at']
        assert claims['permissions'] == r.claims['permissions']
        assert retired(r, previous) and not retired(r, current)
        previous = current
        calls = len(r.member.refreshes)
        cycle(r)
        assert len(r.member.refreshes) == calls
    assert len(r.member.refreshes) == 3
    assert not driver.list_tasks(r.service.db_path, room_id='room')
    r.clock[0] += 3300
    r.member.offline = True
    cycle(r)
    calls = len(r.member.refreshes)
    r.clock[0] += 29
    cycle(r)
    assert len(r.member.refreshes) == calls
    r.clock[0] += 1
    cycle(r)
    assert len(r.member.refreshes) == calls + 1
    r.member.offline = False
    r.clock[0] += 60
    cycle(r)
    assert len(r.member.refreshes) == calls + 2
    assert r.service._route_statuses('room')[0]['status'] == 'ready'


@pytest.mark.parametrize('change', ['expired', 'revoked', 'policy', 'disband', 'epoch', 'lease_lost'])
def test_renewal_never_crosses_an_authority_or_grant_fence(renewal, change):
    r = renewal
    r.clock[0] += 3300
    if change == 'expired':
        r.clock[0] = r.claims['expires_at'] + 1
    elif change == 'revoked':
        hosted_rooms.revoke_room_grant_scope(hosted_rooms.default_db_path(), claims=r.claims,
                                             expires_at=r.claims['status_expires_at'])
    elif change == 'policy':
        (r.home / 'config.yaml').write_text('agent:\n  max_turns: 7\n', encoding='utf-8')
    elif change == 'disband':
        def disband():
            r.service.revoke_room_routes('room')
            hosted_rooms.disband_room(r.service.db_path, room_id='room',
                                      expected_gateway_id=r.route.home_install_id, expected_epoch=1)
        r.member.on_refresh = disband
    elif change == 'epoch':
        def change_epoch():
            with sqlite3.connect(r.service.db_path) as db:
                db.execute("UPDATE hosted_rooms SET authority_epoch=2 WHERE room_id='room'")
        r.member.on_refresh = change_epoch
    else:
        r.member.on_refresh = lambda: driver.release_lease(
            r.service.db_path, r.service.runtime._leases['room'], clock=lambda: r.clock[0])
    cycle(r)
    if change == 'disband':
        assert not [l for l in links.load_room_links(r.service.db_path) if l.room_id == 'room']
    else:
        assert stored(r).grant == r.old
    if change in {'expired', 'revoked', 'policy'}:
        assert not r.member.issued
        assert stored(r).status == 'needs_reauthorization'
        calls = len(r.member.refreshes)
        r.clock[0] += 120
        cycle(r)
        assert len(r.member.refreshes) == calls  # no retries until the member is invited again
    else:
        # The renewal that could not be published was retired, never left live.
        assert retired(r, r.member.issued[0])
        if change == 'lease_lost':
            assert stored(r).status == 'ready'
            r.clock[0] += 60
            cycle(r)
            assert r.service.peer_routes[KEY].grant not in {r.old, r.member.issued[0]}


def runs_handler(r, *, started, finish_after, reads, admissions, rejected, on_read=None):
    target = r.adapter

    def runs(path, *, method, body, room_grant):
        permission = 'dispatch' if method == 'POST' else 'stop' if path.endswith('/stop') else 'status'
        if on_read is not None and method == 'GET':
            on_read()
        error = target._check_run_auth(SimpleNamespace(
            headers={'Authorization': f'HermesRoom {room_grant}'}, path=path, method=method), permission=permission)
        if error is not None:
            rejected.append(permission)
            raise PeerRunsHTTPError('member refused a retired grant', status_code=error.status,
                                    error_code=json.loads(error.text)['error']['code'],
                                    not_admitted=method == 'POST')
        if method == 'POST':
            admissions.append(body['hosted_room_dispatch']['task_id'])
        else:
            reads.append(room_grant)
        complete = r.clock[0] - started >= finish_after
        return {'run_id': 'peer-run', 'status': 'completed' if complete else 'running',
                'output': 'Healthy peer result' if complete else ''}
    return runs


@pytest.mark.parametrize('maintenance', ['disabled', 'poll', 'racing_read'])
def test_an_active_peer_turn_settles_once_across_its_grants_renewal(renewal, monkeypatch, maintenance):
    r = renewal
    r.clock[0] = r.claims['expires_at'] - 360
    started = r.clock[0]
    reads, admissions, rejected = [], [], []
    raced = []
    binding = r.service.bindings()[0]

    def race():
        if maintenance == 'racing_read' and not raced and r.clock[0] - started >= 60:
            raced.append(True)
            r.service._peer_renewal_scans.clear()
            r.service._maintain_peer_grants(binding, r.service.runtime._leases['room'])
    r.member.runs = runs_handler(r, started=started, finish_after=400, reads=reads, admissions=admissions,
                                 rejected=rejected, on_read=race)
    if maintenance != 'poll':
        r.service.runtime.maintain_leased_room = None
    r.service.send(room_id='room', event_id='active', payload={
        'text': '@reviewer Complete one healthy remote task', 'thread_id': 'active-thread'})
    (task,) = driver.list_tasks(r.service.db_path, room_id='room', status='queued')

    def tick(_timeout=None):
        r.clock[0] += 5
        assert r.clock[0] - started <= 450
        return False
    monkeypatch.setattr(r.service.runtime._wake, 'wait', tick)
    cycle(r)
    assert driver.get_task(r.service.db_path, task['identity'])['status'] == 'settled'
    cycle(r)
    events = hosted_rooms.read_events(r.service.db_path, room_id='room')['events']
    visible = [e for e in events if e['kind'] == 'message.member' and e['payload']['text'] == 'Healthy peer result']
    assert len(visible) == 1 and len(admissions) == 1
    assert len([e for e in events if e['kind'] == 'turn.settled']) == 1
    assert stored(r).status == 'ready'
    assert rejected == (['status'] if maintenance == 'racing_read' else [])
    if maintenance != 'disabled':
        assert stored(r).grant != r.old and reads[-1] == stored(r).grant
        assert retired(r, r.old)
    assert r.member.timeout_seconds == 30


def local_turn(r, *, complete):
    rpc = FakeSessionRPC(auto_complete=complete)
    r.service.member_rpcs[('room', 'host', 'default', 'alice', str(r.home))] = rpc
    r.service.send(room_id='room', event_id='local', payload={
        'text': '@host Complete one local turn', 'thread_id': 'local-thread'})
    (task,) = driver.list_tasks(r.service.db_path, room_id='room', status='queued')
    return task['identity'], rpc


@pytest.mark.parametrize('work', ['long', 'queued', 'stopping'])
def test_renewal_runs_during_active_work_but_never_ahead_of_stop_or_new_work(renewal, monkeypatch, work):
    r = renewal
    r.clock[0] = r.claims['expires_at'] - (360 if work == 'long' else 300)
    started = r.clock[0]
    identity, rpc = local_turn(r, complete=work != 'long')
    runtime, binding = r.service.runtime, r.service.bindings()[0]
    expected = 'cancelled' if work == 'stopping' else 'settled'
    if work == 'long':
        def poll(_timeout=None):
            r.clock[0] += 5
            if r.clock[0] - started >= 400:
                rpc.complete(identity.task_id, content='Healthy local result')
            return False
        monkeypatch.setattr(runtime._wake, 'wait', poll)
    else:
        if work == 'stopping':
            lease = runtime._ensure_lease(binding)
            attempt = driver.start_task(r.service.db_path, identity, lease, expected_cancel_generation=0,
                                        clock=lambda: r.clock[0])
            sid = rpc.add_session(profile='default', title=room_session_title('room'), active=True,
                                  task_id=identity.task_id)
            rpc.states[sid]['execution_generation'] = attempt.execution_generation
            driver.begin_task_cancel(r.service.db_path, identity, cancel_id='pending-stop',
                                     expected_cancel_generation=0, clock=lambda: r.clock[0])
        r.member.offline, r.member.offline_seconds = True, 30.0

        def offline_after_the_turn(path, **kwargs):
            assert driver.get_task(r.service.db_path, identity)['status'] == expected
            return Member._request(r.member, path, **kwargs)
        monkeypatch.setattr(r.member, '_request', offline_after_the_turn)
    cycle(r)
    assert driver.get_task(r.service.db_path, identity)['status'] == expected
    if work == 'long':
        assert r.clock[0] - started >= 400 and r.member.refreshes
        assert stored(r).status == 'ready'
        assert decode_room_grant(r.secret, stored(r).grant, permission='dispatch')['status_expires_at'] == \
            r.claims['status_expires_at']
    else:
        assert r.member.refreshes and r.clock[0] - started <= 2.001  # bounded by the cycle's budget
    assert r.member.timeout_seconds == 30


def test_scans_and_requests_stay_throttled_when_the_horizon_prevents_extension(renewal, monkeypatch):
    r = renewal
    scans = []
    original = links.load_room_links_tolerant

    def counted(db_path):
        scans.append(1)
        return original(db_path)
    monkeypatch.setattr(links, 'load_room_links_tolerant', counted)
    cycle(r)
    assert len(scans) == 1
    cycle(r, 10)
    assert len(scans) == 1 and not r.member.refreshes
    hard = r.claims['status_expires_at']
    r.clock[0] = hard - 180
    scope = {k: r.claims[k] for k in ('room_id', 'home_install_id', 'authority_gateway_id', 'authority_epoch',
                                      'member_id', 'target_install_id', 'target_profile',
                                      'execution_policy_digest', 'permissions')}
    near = issue_room_grant(r.secret, grant_id='near-horizon', issued_at=r.clock[0], ttl_seconds=180,
                            status_expires_at=hard, **scope)
    r.service.register_peer_route(room_id='room', member_id='reviewer', route=replace(r.route, grant=near),
                                  client=r.member, target_url=r.member.base_url, catalog=r.catalog)
    for remaining in (180, 120, 60):
        r.clock[0] = hard - remaining
        cycle(r)
        count, scan_count = len(r.member.refreshes), len(scans)
        cycle(r)
        assert len(r.member.refreshes) == count and len(scans) == scan_count
        claims = decode_room_grant(r.secret, stored(r).grant, permission='dispatch')
        assert claims['expires_at'] == claims['status_expires_at'] == hard
    issued = len(r.member.issued)
    r.clock[0] = hard
    cycle(r)
    assert stored(r).status == 'needs_reauthorization' and len(r.member.issued) == issued


def test_the_cycle_budget_defers_unvisited_routes_without_starving_them(renewal, monkeypatch):
    r = renewal
    for member in ('second', 'third'):
        r.service.register_peer_route(room_id='room', member_id=member, route=replace(r.route, member_id=member),
                                      client=r.member, target_url=r.member.base_url, catalog=r.catalog)
    r.clock[0] = r.claims['expires_at'] - 300
    calls = []
    r.member.offline, r.member.offline_seconds = True, 30.0
    original = Member._request

    def counted(path, **kwargs):
        calls.append(r.clock[0])
        return original(r.member, path, **kwargs)
    monkeypatch.setattr(r.member, '_request', counted)
    start = r.clock[0]
    cycle(r)
    assert len(calls) == 2 and r.clock[0] - start <= 2
    assert ('room', 'second') not in r.service._peer_renewals
    assert ('room', 'third') not in r.service._peer_renewals
    for elapsed, member in ((5, 'second'), (10, 'third')):
        r.clock[0] = start + elapsed
        cycle(r)
        assert r.clock[0] - (start + elapsed) <= 2
        assert ('room', member) in r.service._peer_renewals
    assert len(calls) == 6


@pytest.mark.asyncio
async def test_a_refresh_racing_a_revocation_never_returns_a_live_grant(renewal, monkeypatch):
    r = renewal
    from gateway import hosted_room_peer
    real_issue = hosted_room_peer.issue_room_grant

    def revoke_then_issue(*args, **kwargs):
        # Disband's revocation lands after the refresh's first check, before it mints.
        hosted_rooms.revoke_room_grant_scope(hosted_rooms.default_db_path(), claims=r.claims,
                                             expires_at=r.claims['status_expires_at'])
        r.clock[0] += 0.5
        return real_issue(*args, **kwargs)
    monkeypatch.setattr(hosted_room_peer, 'issue_room_grant', revoke_then_issue)
    r.adapter._read_json_body = AsyncMock(return_value=({'ttl_seconds': 3600}, None))
    response = await api_server_room_grants._handle_room_member_grant_refresh(
        r.adapter, SimpleNamespace(headers={'Authorization': f'HermesRoom {r.old}'}),
        _openai_error=api_server._openai_error, _api_request_profile=api_server._api_request_profile)
    assert response.status == 403
    assert 'grant' not in json.loads(response.text)


@pytest.mark.parametrize('change', ['horizon', 'rights', 'scope'])
def test_a_renewal_that_widens_or_moves_its_grant_is_refused(renewal, monkeypatch, change):
    r = renewal
    r.clock[0] += 3300
    scope = {k: r.claims[k] for k in ('room_id', 'home_install_id', 'authority_gateway_id', 'authority_epoch',
                                      'member_id', 'target_install_id', 'target_profile', 'execution_policy_digest')}
    permissions, horizon = r.claims['permissions'], r.claims['status_expires_at']
    if change == 'horizon':
        horizon += 1000
    elif change == 'rights':
        permissions = [p for p in permissions if p != 'stop']
    else:
        scope['room_id'] = 'another-room'
        hosted_rooms.reserve_peer_room(hosted_rooms.default_db_path(), claims={**scope, 'expires_at': horizon},
                                       expires_at=horizon)
    crafted = issue_room_grant(r.secret, grant_id='crafted', issued_at=r.clock[0], ttl_seconds=3600,
                               status_expires_at=horizon, permissions=permissions, **scope)
    original = r.member._request

    def crafted_renewal(path, **kwargs):
        if path.endswith('/refresh'):  # a member gateway answering a renewal with another grant
            r.member.refreshes.append(r.clock[0])
            return {'grant': crafted}
        return original(path, **kwargs)
    monkeypatch.setattr(r.member, '_request', crafted_renewal)
    cycle(r)
    assert r.member.refreshes
    assert stored(r).grant == r.old and stored(r).status == 'needs_reauthorization'
    assert retired(r, crafted)  # refused, and retired rather than left live


def test_upkeep_that_fails_never_decides_the_turn_it_runs_beside(renewal, monkeypatch):
    r = renewal
    started = r.clock[0]
    identity, rpc = local_turn(r, complete=False)

    def poll(_timeout=None):
        r.clock[0] += 5
        if r.clock[0] - started >= 30:
            rpc.complete(identity.task_id, content='Healthy local result')
        return False
    monkeypatch.setattr(r.service.runtime._wake, 'wait', poll)

    def failing(binding, lease):
        raise RuntimeError('renewal store unavailable')
    r.service.runtime.maintain_leased_room = failing
    cycle(r)
    assert driver.get_task(r.service.db_path, identity)['status'] == 'settled'
    assert 'upkeep failed' in r.service.runtime.status()['last_error']


def test_a_newly_registered_grant_is_scheduled_afresh(renewal):
    r = renewal
    cycle(r)  # a scan that found nothing due schedules the next one a minute out
    assert r.service._peer_renewal_scans['room'] >= r.clock[0] + 60
    scope = {k: r.claims[k] for k in ('room_id', 'home_install_id', 'authority_gateway_id', 'authority_epoch',
                                      'member_id', 'target_install_id', 'target_profile',
                                      'execution_policy_digest', 'permissions')}
    short = issue_room_grant(r.secret, grant_id='short', issued_at=r.clock[0], ttl_seconds=60,
                             status_expires_at=r.claims['status_expires_at'], **scope)
    hosted_rooms.reserve_peer_room(hosted_rooms.default_db_path(), claims=decode_room_grant(
        r.secret, short, permission='status'), expires_at=r.claims['status_expires_at'])
    r.service.register_peer_route(room_id='room', member_id='reviewer', route=replace(r.route, grant=short),
                                  client=r.member, target_url=r.member.base_url, catalog=r.catalog)
    r.clock[0] += 1
    cycle(r)
    assert stored(r).grant not in {r.old, short} and retired(r, short)
