"""A replaced grant is retired at once; attempts stay on their route across the replacement.

The member's gateway is a real API server adapter behind an aiohttp TestServer: its grant,
capability, refresh and run handlers answer every request the home sends.
"""
import asyncio
from dataclasses import replace
import json
import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest

from gateway import hosted_room_links, hosted_rooms
from gateway.hosted_room_peer import (
    HostedRoomGrantError, decode_room_grant, gateway_room_grant_secret, issue_room_grant)
from gateway.platforms import api_server_room_grants
from gateway.session_group_peer_routes import CanonicalPeerClient, before_sending, set_route_status
from tests.gateway.test_session_group_peers import call, gateway, invite, linked_room  # noqa: F401
from gateway.session_group_peers import room_link
from tui_gateway.hosted_room_driver import HostedRoomBinding
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError
from tui_gateway.hosted_room_peer_transport import build_member_dispatch

KEY = ('linked', 'reviewer')


async def serve(gateway, monkeypatch):
    """The member's gateway: its room-member and run routes over loopback."""
    from aiohttp import web
    from aiohttp.test_utils import TestServer
    from gateway.platforms import api_server_runs
    app = web.Application()
    for method, path, handler in (api_server_room_grants._http_routes(gateway.adapter)
                                  + api_server_runs._http_routes(gateway.adapter)):
        app.router.add_route(method, path, handler)
    server = TestServer(app)
    await server.start_server()
    url = str(server.make_url('')).rstrip('/')
    monkeypatch.setenv('HERMES_ROOM_LINK_URL', url)
    return server, url


def capabilities(url, grant):
    """The member's gateway's answer to a grant: (HTTP status, error code or None)."""
    try:
        PeerRunsHTTPClient(base_url=url, api_key='').probe(grant=grant)
        return 200, None
    except PeerRunsHTTPError as exc:
        return exc.status_code, exc.error_code


def respelled(grant):
    """The same signed grant in another base64 spelling (a padded signature)."""
    payload, signature = grant.split('.')
    assert len(signature) % 4
    return payload + '.' + signature + '=' * (-len(signature) % 4)


def denylisted():
    with sqlite3.connect(hosted_rooms.default_db_path()) as db:
        return db.execute('SELECT COUNT(*) FROM hosted_room_revoked_grant_tokens').fetchone()[0]


async def joined(gateway, monkeypatch, **invitation):
    """A served member gateway and a room whose peer member is registered: (url, room, catalog, grant)."""
    server, url = await serve(gateway, monkeypatch)
    catalog = room_link(gateway.authority)['catalog']
    room = await linked_room(gateway, catalog)
    grant = (await invite(gateway, room))['grant']
    registered = await call(gateway.owner, 'groups.peer.register', room_id='linked', member_id='reviewer',
                            target_url=url, target_profile='default', grant=grant, catalog=catalog)
    assert registered['registered'], registered
    return server, url, room, catalog, grant


async def reregister(gateway, room, url, catalog):
    grant = (await invite(gateway, room))['grant']
    return grant, await call(gateway.owner, 'groups.peer.register', room_id='linked', member_id='reviewer',
                             target_url=url, target_profile='default', grant=grant, catalog=catalog)


@pytest.mark.asyncio
async def test_exact_revoke_retires_one_grant_and_leaves_its_scope_usable(gateway, monkeypatch):
    server, url, room, catalog, grant = await joined(gateway, monkeypatch)
    try:
        sibling = (await invite(gateway, room))['grant']  # same room, member and target
        client = PeerRunsHTTPClient(base_url=url, api_key='')
        assert await asyncio.to_thread(client.revoke_grant_exact, grant=grant) == {
            'object': 'hermes.room_member.grant.revocation', 'revoked': True}
        assert await asyncio.to_thread(capabilities, url, grant) == (403, 'room_reauthorization_required')
        assert await asyncio.to_thread(capabilities, url, sibling) == (200, None)
        # Idempotent: a retry after a lost response is acknowledged again.
        assert (await asyncio.to_thread(client.revoke_grant_exact, grant=grant))['revoked'] is True
        # One grant, whatever its spelling: revoking one spelling retires every other.
        assert await asyncio.to_thread(capabilities, url, respelled(sibling)) == (200, None)
        await asyncio.to_thread(client.revoke_grant_exact, grant=respelled(sibling))
        assert await asyncio.to_thread(capabilities, url, sibling) == (403, 'room_reauthorization_required')
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_exact_revoke_refuses_anything_but_a_signed_grant_for_this_gateway(gateway, monkeypatch):
    server, url, room, catalog, grant = await joined(gateway, monkeypatch)
    secret = gateway_room_grant_secret()
    claims = decode_room_grant(secret, grant, permission='status')
    scope = {k: claims[k] for k in ('room_id', 'home_install_id', 'authority_gateway_id', 'authority_epoch',
                                    'member_id', 'target_install_id', 'target_profile', 'execution_policy_digest')}
    try:
        client = PeerRunsHTTPClient(base_url=url, api_key='')
        before = await asyncio.to_thread(denylisted)
        tampered = grant[:-2] + ('AA' if grant[-2:] != 'AA' else 'BB')
        refused = [
            tampered,
            issue_room_grant(secret, grant_id='other-profile', **{**scope, 'target_profile': 'helper'}),
            issue_room_grant(secret, grant_id='other-install', **{**scope, 'target_install_id': 'install:other'}),
            issue_room_grant(secret, grant_id='dispatch-only', **scope, permissions=('dispatch',)),
            issue_room_grant(b'x' * 32, grant_id='foreign-secret', **scope),
        ]
        for token in refused:
            with pytest.raises(PeerRunsHTTPError) as caught:
                await asyncio.to_thread(client.revoke_grant_exact, grant=token)
            assert caught.value.status_code == 401, token
        with pytest.raises(PeerRunsHTTPError) as caught:
            await asyncio.to_thread(client._request, '/v1/room-members/grants/revoke-exact', method='POST',
                                    body={'grant': grant}, room_grant=grant)
        assert caught.value.status_code == 400
        assert await asyncio.to_thread(denylisted) == before
        assert await asyncio.to_thread(capabilities, url, grant) == (200, None)
        # A grant past its whole lifetime can still be retired: nothing is left to do.
        expired = issue_room_grant(secret, grant_id='expired', **scope, issued_at=time.time() - 7200,
                                   ttl_seconds=60, status_ttl_seconds=60)
        assert (await asyncio.to_thread(client.revoke_grant_exact, grant=expired))['revoked'] is True
        assert await asyncio.to_thread(denylisted) == before
        # A store that cannot record the revocation never acknowledges it.
        def unavailable(*args, **kwargs):
            raise OSError('disk full')
        monkeypatch.setattr(hosted_rooms, 'revoke_room_grant_token', unavailable)
        with pytest.raises(PeerRunsHTTPError) as caught:
            await asyncio.to_thread(client.revoke_grant_exact, grant=grant)
        assert (caught.value.status_code, caught.value.error_code) == (503, 'room_grant_revocation_unavailable')
        assert caught.value.retryable
        assert await asyncio.to_thread(capabilities, url, grant) == (200, None)
    finally:
        await server.close()


def test_an_expired_grant_is_valid_only_for_its_own_revocation(gateway):
    secret = gateway_room_grant_secret()
    expired = issue_room_grant(secret, grant_id='expired', room_id='linked', home_install_id='home',
                               authority_gateway_id='home', authority_epoch=1, member_id='reviewer',
                               target_install_id='target', target_profile='default', execution_policy_digest='0' * 64,
                               issued_at=time.time() - 7200, ttl_seconds=60, status_ttl_seconds=60)
    for permission in ('status', 'dispatch'):
        with pytest.raises(HostedRoomGrantError):
            decode_room_grant(secret, expired, permission=permission)
    assert decode_room_grant(secret, expired, permission='status', allow_expired_for_revocation=True)
    for permission in ('dispatch', 'stop', 'approve'):
        with pytest.raises(HostedRoomGrantError):
            decode_room_grant(secret, expired, permission=permission, allow_expired_for_revocation=True)


@pytest.mark.parametrize('method', ['revoke_grant', 'revoke_grant_exact'])
@pytest.mark.parametrize('response', [{}, {'revoked': False}, {'revoked': 'true'}])
def test_a_revocation_counts_only_when_acknowledged(monkeypatch, method, response):
    client = PeerRunsHTTPClient(base_url='http://127.0.0.1:9', api_key='')
    monkeypatch.setattr(client, '_request', lambda *a, **k: response)
    with pytest.raises(PeerRunsHTTPError) as caught:
        getattr(client, method)(grant='grant')
    assert caught.value.retryable
    monkeypatch.setattr(client, '_request', lambda *a, **k: {'revoked': True})
    assert getattr(client, method)(grant='grant') == {'revoked': True}


@pytest.mark.asyncio
async def test_reregistration_retires_the_replaced_grant_at_once(gateway, monkeypatch):
    server, url, room, catalog, first = await joined(gateway, monkeypatch)
    try:
        second, registered = await reregister(gateway, room, url, catalog)
        assert registered['registered'], registered
        assert await asyncio.to_thread(capabilities, url, first) == (403, 'room_reauthorization_required')
        assert await asyncio.to_thread(capabilities, url, second) == (200, None)
        stored, = hosted_room_links.load_room_links(gateway.db.db_path)
        assert stored.grant == second == gateway.service.peer_routes[KEY].grant
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_a_grant_that_cannot_be_retired_keeps_the_old_route(gateway, monkeypatch):
    server, url, room, catalog, first = await joined(gateway, monkeypatch)
    real = api_server_room_grants._handle_room_member_grant_revoke_exact
    try:
        async def unavailable(self, request, **kwargs):
            return api_server_room_grants._json_error(
                kwargs['_openai_error'], 'busy', code='room_grant_revocation_unavailable', status=503)
        monkeypatch.setattr(api_server_room_grants, '_handle_room_member_grant_revoke_exact', unavailable)
        second, refused = await reregister(gateway, room, url, catalog)
        assert refused == 'peer_unreachable'
        stored, = hosted_room_links.load_room_links(gateway.db.db_path)
        assert stored.grant == first == gateway.service.peer_routes[KEY].grant
        assert await asyncio.to_thread(capabilities, url, first) == (200, None)
        monkeypatch.setattr(api_server_room_grants, '_handle_room_member_grant_revoke_exact', real)
        third, registered = await reregister(gateway, room, url, catalog)
        assert registered['registered'], registered
        assert await asyncio.to_thread(capabilities, url, first) == (403, 'room_reauthorization_required')
        assert hosted_room_links.load_room_links(gateway.db.db_path)[0].grant == third
    finally:
        await server.close()


def attempt(gateway, room, *, task_id='dtask:route'):
    """One attempt's client and dispatch, as the driver resolves them."""
    binding = HostedRoomBinding('linked', room['authority_gateway_id'], room['authority_epoch'])
    route, client = gateway.service.peer_routes[KEY], gateway.service.peer_clients[KEY]
    tracked = gateway.service._track_peer_client(binding, KEY, route, client)
    dispatch = build_member_dispatch(
        binding=binding, route=route, room_id='linked', task_id=task_id, target_profile='default',
        execution_generation=1, source_event_seq=1, prompt='review', trace_id=route.trace_id).as_mapping()
    return tracked, route, dispatch


async def runs(gateway):
    with sqlite3.connect(gateway.adapter._run_idempotency_store._db_path) as db:
        return db.execute('SELECT idempotency_key FROM run_idempotency').fetchall()


@pytest.fixture
def inert_runs(monkeypatch):
    from gateway.platforms import api_server_runs

    async def inert(adapter, run, **kwargs):
        pass
    monkeypatch.setattr(api_server_runs, '_execute_run', inert)


@pytest.mark.asyncio
async def test_a_stale_attempt_sends_no_new_work_with_a_replaced_grant(gateway, monkeypatch, inert_runs):
    server, url, room, catalog, first = await joined(gateway, monkeypatch)
    try:
        tracked, route, dispatch = attempt(gateway, room)
        await reregister(gateway, room, url, catalog)
        for method, phase in (('dispatch', 'dispatch_not_attempted'), ('recover_dispatch', 'ambiguous')):
            with pytest.raises(RuntimeError) as caught:
                await asyncio.to_thread(getattr(tracked, method), dispatch=dispatch, grant=route.grant)
            assert 'route changed before admission' in str(caught.value)
            assert getattr(caught.value, phase) is True and caught.value.not_admitted is False
        with pytest.raises(RuntimeError, match='route changed before admission'):
            await asyncio.to_thread(tracked.probe, grant=route.grant)
        assert await runs(gateway) == []
        assert gateway.service._route_statuses('linked')[0]['status'] == 'ready'
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_accepted_work_is_read_and_stopped_with_its_routes_current_grant(gateway, monkeypatch, inert_runs):
    server, url, room, catalog, first = await joined(gateway, monkeypatch)
    try:
        tracked, route, dispatch = attempt(gateway, room)
        accepted = await asyncio.to_thread(tracked.dispatch, dispatch=dispatch, grant=route.grant)
        second, _ = await reregister(gateway, room, url, catalog)
        sent = []
        raw = tracked._client
        real_request = raw._request

        def recording(path, **kwargs):
            sent.append((path, kwargs.get('room_grant')))
            return real_request(path, **kwargs)
        monkeypatch.setattr(raw, '_request', recording)
        status = await asyncio.to_thread(tracked.status, room_id='linked', profile='default',
                                         session_id=accepted['session_id'], grant=route.grant)
        assert status['active'] and sent == [(f"/v1/runs/{accepted['run_id']}", second)]
        stopped = await asyncio.to_thread(tracked.stop_receipt, task_id=dispatch['task_id'],
                                          execution_generation=1, grant=route.grant)
        assert stopped is not None and sent[-1][1] == second
        # Exact cleanup always names the grant it retires, never the current one.
        await asyncio.to_thread(tracked.revoke_grant_exact, grant=first)
        assert sent[-1] == ('/v1/room-members/grants/revoke-exact', first)
        assert await asyncio.to_thread(capabilities, url, second) == (200, None)
    finally:
        await server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['trace', 'url', 'membership', 'epoch', 'removed', 'reauthorization'])
async def test_an_observer_refuses_a_route_that_changed_under_it(gateway, monkeypatch, inert_runs, change):
    server, url, room, catalog, first = await joined(gateway, monkeypatch)
    try:
        tracked, route, dispatch = attempt(gateway, room)
        accepted = await asyncio.to_thread(tracked.dispatch, dispatch=dispatch, grant=route.grant)
        await reregister(gateway, room, url, catalog)
        service = gateway.service
        if change == 'trace':
            service.peer_routes[KEY] = replace(service.peer_routes[KEY], trace_id='trace-other')
        elif change == 'url':
            service.peer_clients[KEY] = PeerRunsHTTPClient(base_url='http://127.0.0.1:9', api_key='')
        elif change == 'membership':
            members = service._room('linked')['members']
            members[1]['handle'] = 'renamed'
            with sqlite3.connect(gateway.db.db_path) as db:
                db.execute('UPDATE hosted_rooms SET members_json=? WHERE room_id=?', (json.dumps(members), 'linked'))
        elif change == 'epoch':
            hosted_rooms.claim_authority(gateway.db.db_path, room_id='linked',
                                         expected_gateway_id=room['authority_gateway_id'], expected_epoch=1,
                                         new_gateway_id=room['authority_gateway_id'], event_id='reclaim')
        elif change == 'removed':
            service.peer_routes.pop(KEY)
        else:
            service._peer_route_status[KEY] = 'needs_reauthorization'
        sent = []
        monkeypatch.setattr(tracked._client, '_request', lambda path, **kwargs: sent.append(path))
        with pytest.raises(RuntimeError, match='observer'):
            await asyncio.to_thread(tracked.status, room_id='linked', profile='default',
                                    session_id=accepted['session_id'], grant=route.grant)
        assert sent == []
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_a_read_that_races_a_replacement_retries_once_with_the_new_grant(gateway, monkeypatch, inert_runs):
    server, url, room, catalog, first = await joined(gateway, monkeypatch)
    try:
        tracked, route, dispatch = attempt(gateway, room)
        accepted = await asyncio.to_thread(tracked.dispatch, dispatch=dispatch, grant=route.grant)
        second = (await invite(gateway, room))['grant']
        raw, sent = tracked._client, []
        raw.clock = lambda: 100.0  # the status poll's backoff window is still open for the retry
        real_request = raw._request
        loop = asyncio.get_running_loop()

        def racing(path, **kwargs):
            if path.startswith('/v1/runs/') and not sent:
                # Re-registration lands between choosing the grant and sending the read.
                asyncio.run_coroutine_threadsafe(call(
                    gateway.owner, 'groups.peer.register', room_id='linked', member_id='reviewer',
                    target_url=url, target_profile='default', grant=second, catalog=catalog), loop).result(30)
            sent.append(kwargs.get('room_grant'))
            return real_request(path, **kwargs)
        monkeypatch.setattr(raw, '_request', racing)
        status = await asyncio.to_thread(tracked.status, room_id='linked', profile='default',
                                         session_id=accepted['session_id'], grant=route.grant)
        assert status['active'] and sent[0] == first and sent[-1] == second
        # The refusal of the retired grant did not relabel the route that replaced it.
        assert gateway.service._route_statuses('linked')[0]['status'] == 'ready'
    finally:
        await server.close()


def test_a_health_report_applies_only_while_its_grant_is_current(gateway):
    from gateway.hosted_room_peer import GatewayRoomCatalog, catalog_mapping
    from tui_gateway.hosted_room_peer_transport import PeerMemberRoute
    service = gateway.service
    catalog = GatewayRoomCatalog.from_mapping(catalog_mapping(
        installation_id='install:target', persistent_process=True, target_profile='default'))
    route = PeerMemberRoute(home_install_id='home', member_id='reviewer', target_install_id='install:target',
                            target_profile='default', capability_digest=catalog.catalog_digest,
                            execution_policy_digest=catalog.execution_policy.policy_digest,
                            cancellation_scope_id='cancel-linked', trace_id='trace-linked', grant='current')
    service._save_link(room_id='linked', member_id='reviewer', target_url='http://127.0.0.1:9',
                       target_profile='default', grant='current', catalog=catalog,
                       cancellation_scope_id='cancel-linked', trace_id='trace-linked')
    service._publish_route(KEY, route, object())
    set_route_status(service, KEY, 'needs_reauthorization', 'retired')
    assert service._route_statuses()[0]['status'] == 'ready'
    # Persisted but not yet published: the stored grant decides.
    service._save_link(room_id='linked', member_id='reviewer', target_url='http://127.0.0.1:9',
                       target_profile='default', grant='newer', catalog=catalog,
                       cancellation_scope_id='cancel-linked', trace_id='trace-linked')
    set_route_status(service, KEY, 'unavailable', 'current')
    assert service._route_statuses()[0]['status'] == 'ready'
    assert hosted_room_links.load_room_links(gateway.db.db_path)[0].status == 'ready'
    service._publish_route(KEY, replace(route, grant='newer'))
    set_route_status(service, KEY, 'unavailable', 'newer')
    assert service._route_statuses()[0]['status'] == 'unavailable'
    assert hosted_room_links.load_room_links(gateway.db.db_path)[0].status == 'unavailable'


@pytest.mark.asyncio
async def test_a_renewal_before_dispatch_is_published_and_retires_the_old_grant(gateway, monkeypatch, inert_runs):
    server, url = await serve(gateway, monkeypatch)
    try:
        catalog = room_link(gateway.authority)['catalog']
        room = await linked_room(gateway, catalog)
        owner = PeerRunsHTTPClient(base_url=url, api_key='target-gateway-api-key')
        # A one-minute grant under a longer status horizon: due for renewal at once.
        short = (await asyncio.to_thread(
            owner.issue_invitation, room_id='linked', home_install_id=room['authority_gateway_id'],
            authority_gateway_id=room['authority_gateway_id'], authority_epoch=1, member_id='reviewer',
            grant_id='short', ttl_seconds=60, status_ttl_seconds=3600))['grant']
        registered = await call(gateway.owner, 'groups.peer.register', room_id='linked', member_id='reviewer',
                                target_url=url, target_profile='default', grant=short, catalog=catalog)
        assert registered['registered'], registered
        tracked, route, dispatch = attempt(gateway, room)
        refreshes = []
        real_refresh = tracked._client.refresh_grant
        monkeypatch.setattr(tracked._client, 'refresh_grant', lambda **kw: refreshes.append(1) or real_refresh(**kw))
        accepted = await asyncio.to_thread(tracked.dispatch, dispatch=dispatch, grant=route.grant)
        renewed = gateway.service.peer_routes[KEY].grant
        assert renewed != short and refreshes == [1] and accepted['run_id']
        assert hosted_room_links.load_room_links(gateway.db.db_path)[0].grant == renewed
        assert await asyncio.to_thread(capabilities, url, short) == (403, 'room_reauthorization_required')
        # The attempt keeps using its own published renewal: no second refresh.
        _, _, other = attempt(gateway, room, task_id='dtask:second')
        await asyncio.to_thread(tracked.dispatch, dispatch=other, grant=route.grant)
        assert refreshes == [1] and len(await runs(gateway)) == 2
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_a_renewal_that_loses_to_a_reregistration_is_retired(gateway, monkeypatch, inert_runs):
    server, url = await serve(gateway, monkeypatch)
    try:
        catalog = room_link(gateway.authority)['catalog']
        room = await linked_room(gateway, catalog)
        owner = PeerRunsHTTPClient(base_url=url, api_key='target-gateway-api-key')
        short = (await asyncio.to_thread(
            owner.issue_invitation, room_id='linked', home_install_id=room['authority_gateway_id'],
            authority_gateway_id=room['authority_gateway_id'], authority_epoch=1, member_id='reviewer',
            grant_id='short', ttl_seconds=60, status_ttl_seconds=3600))['grant']
        await call(gateway.owner, 'groups.peer.register', room_id='linked', member_id='reviewer',
                   target_url=url, target_profile='default', grant=short, catalog=catalog)
        tracked, route, dispatch = attempt(gateway, room)
        winner = (await invite(gateway, room))['grant']
        loop, issued = asyncio.get_running_loop(), []
        real_refresh = tracked._client.refresh_grant

        def refresh_then_lose(**kwargs):
            refreshed = real_refresh(**kwargs)
            issued.append(refreshed['grant'])
            asyncio.run_coroutine_threadsafe(call(
                gateway.owner, 'groups.peer.register', room_id='linked', member_id='reviewer',
                target_url=url, target_profile='default', grant=winner, catalog=catalog), loop).result(30)
            return refreshed
        monkeypatch.setattr(tracked._client, 'refresh_grant', refresh_then_lose)
        with pytest.raises(Exception) as caught:
            await asyncio.to_thread(tracked.dispatch, dispatch=dispatch, grant=route.grant)
        assert caught.value.dispatch_not_attempted is True and not caught.value.not_admitted
        assert hosted_room_links.load_room_links(gateway.db.db_path)[0].grant == winner
        assert await asyncio.to_thread(capabilities, url, issued[0]) == (403, 'room_reauthorization_required')
        assert await asyncio.to_thread(capabilities, url, winner) == (200, None)
        assert await runs(gateway) == []
    finally:
        await server.close()


class _Refreshing:
    """A member client whose refresh fails in a chosen way (no network)."""

    base_url = 'http://127.0.0.1:9'

    def __init__(self, failure=None, refreshed=None, admission=None):
        self.failure, self.refreshed, self.admission = failure, refreshed, admission
        self.calls, self.retired = [], []

    catalog = None

    def refresh_grant(self, **kwargs):
        if self.failure:
            raise self.failure
        return {'grant': self.refreshed, **({'catalog': self.catalog} if self.catalog else {})}

    def dispatch(self, **kwargs):
        self.calls.append('dispatch')
        if self.admission:
            raise self.admission
        return {'status': 'accepted'}

    def recover_dispatch(self, **kwargs):
        self.calls.append('recover_dispatch')
        return {'status': 'accepted'}

    def revoke_grant_exact(self, *, grant):
        self.retired.append(grant)
        return {'revoked': True}


def phase_case(tmp_path, monkeypatch, client, *, save_fails=False):
    """A canonical attempt client over an expired grant, on a minimal service (no I/O)."""
    from gateway.hosted_room_peer import GatewayRoomCatalog, catalog_mapping
    from tui_gateway.hosted_room_peer_transport import PeerMemberRoute
    catalog = GatewayRoomCatalog.from_mapping(catalog_mapping(
        installation_id='install:target', persistent_process=True, target_profile='default'))
    scope = dict(room_id='linked', home_install_id='home', authority_gateway_id='home', authority_epoch=1,
                 member_id='reviewer', target_install_id='install:target', target_profile='default',
                 execution_policy_digest=catalog.execution_policy.policy_digest)
    now = time.time()
    old = issue_room_grant(b's' * 32, grant_id='old', issued_at=now - 3700, ttl_seconds=3600,
                           status_expires_at=now + 10000, **scope)
    new = issue_room_grant(b's' * 32, grant_id='new', issued_at=now, ttl_seconds=3600,
                           status_expires_at=now + 10000, **scope)
    route = PeerMemberRoute(home_install_id='home', member_id='reviewer', target_install_id='install:target',
                            target_profile='default', capability_digest=catalog.catalog_digest,
                            execution_policy_digest=catalog.execution_policy.policy_digest,
                            cancellation_scope_id='cancel-linked', trace_id='trace-linked', grant=old)
    stored = hosted_room_links.make_stored_link(
        room_id='linked', member_id='reviewer', target_url=client.base_url, target_profile='default', grant=old,
        catalog=catalog, cancellation_scope_id='cancel-linked', trace_id='trace-linked')
    hosted_room_links.save_room_link(tmp_path / 'state.db', stored)
    statuses = []

    def save(**link):
        if save_fails:
            raise hosted_rooms.HostedRoomError('the route store is unavailable')
        hosted_room_links.save_room_link(tmp_path / 'state.db', hosted_room_links.make_stored_link(
            **{k: v for k, v in link.items() if k != 'authorize'}))
    service = SimpleNamespace(
        _policy_lock=threading.RLock(), peer_route_lock=threading.RLock(), db_path=tmp_path / 'state.db',
        peer_routes={KEY: route}, peer_clients={KEY: client}, _peer_route_status={KEY: 'ready'},
        _room=lambda room_id: {'authority_gateway_id': 'home', 'authority_epoch': 1, 'members': []},
        _save_link=save, _publish_route=lambda key, route, client=None: service.peer_routes.update({key: route}),
        runtime=SimpleNamespace(wakeup=lambda: None))
    monkeypatch.setattr('gateway.session_group_peer_routes.set_route_status',
                        lambda svc, key, status, grant: statuses.append(status))
    binding = HostedRoomBinding('linked', 'home', 1)
    tracked = CanonicalPeerClient(service, binding, KEY, route, client)
    dispatch = build_member_dispatch(binding=binding, route=route, room_id='linked', task_id='task-1',
                                     target_profile='default', execution_generation=1, source_event_seq=1,
                                     prompt='review', trace_id=route.trace_id).as_mapping()
    return SimpleNamespace(tracked=tracked, old=old, new=new, dispatch=dispatch, statuses=statuses)


@pytest.mark.parametrize('method', ['dispatch', 'recover_dispatch'])
@pytest.mark.parametrize('kind', ['auth', 'network', 'missing-grant', 'persistence'])
def test_a_failure_before_sending_keeps_its_phase(tmp_path, monkeypatch, method, kind):
    failure = (PeerRunsHTTPError('expired grant', status_code=401, error_code='invalid_room_grant')
               if kind == 'auth' else RuntimeError('refresh unavailable') if kind == 'network' else None)
    client = _Refreshing(failure=failure)
    case = phase_case(tmp_path, monkeypatch, client, save_fails=kind == 'persistence')
    client.refreshed = '' if kind == 'missing-grant' else case.new
    with pytest.raises(Exception) as caught:
        getattr(case.tracked, method)(dispatch=case.dispatch, grant=case.old)
    error = caught.value
    assert error.not_admitted is False
    assert getattr(error, 'dispatch_not_attempted', False) is (method == 'dispatch')
    if method == 'recover_dispatch':
        assert error.ambiguous is True
    assert client.calls == []
    if failure:
        assert error.__cause__ is failure and type(error) is type(failure)
    if kind == 'auth':
        assert error.needs_reauthorization and case.statuses == ['needs_reauthorization']
    if kind == 'persistence':
        assert client.retired == [case.new]  # the unpublished renewal is retired, not left live


def test_a_renewal_with_a_changed_policy_is_refused_and_retired(tmp_path, monkeypatch):
    from gateway.hosted_room_execution_policy import execution_policy_mapping
    from gateway.hosted_room_peer import catalog_mapping
    client = _Refreshing()
    case = phase_case(tmp_path, monkeypatch, client)
    client.refreshed = case.new
    client.catalog = catalog_mapping(
        installation_id='install:target', persistent_process=True, target_profile='default',
        execution_policy=execution_policy_mapping(target_profile='default', config={'agent': {'max_turns': 7}}))
    with pytest.raises(PeerRunsHTTPError) as caught:
        case.tracked.dispatch(dispatch=case.dispatch, grant=case.old)
    assert caught.value.needs_reauthorization and caught.value.dispatch_not_attempted is True
    assert case.statuses == ['needs_reauthorization']
    assert client.retired == [case.new] and client.calls == []


@pytest.mark.parametrize('ambiguous,not_admitted', [(True, False), (False, True)])
def test_a_failure_of_the_send_itself_keeps_its_classification(tmp_path, monkeypatch, ambiguous, not_admitted):
    failure = PeerRunsHTTPError('admission response', ambiguous=ambiguous, not_admitted=not_admitted,
                                status_code=409, error_code='conflict')
    client = _Refreshing(admission=failure)
    case = phase_case(tmp_path, monkeypatch, client)
    client.refreshed = case.new
    with pytest.raises(PeerRunsHTTPError) as caught:
        case.tracked.dispatch(dispatch=case.dispatch, grant=case.old)
    assert caught.value is failure and not hasattr(failure, 'dispatch_not_attempted')
    assert client.calls == ['dispatch']


def test_phase_evidence_never_mutates_an_exception_it_did_not_create(tmp_path, monkeypatch):
    prior = PeerRunsHTTPError('prior uncertain request', ambiguous=True)
    before = dict(prior.__dict__)
    case = phase_case(tmp_path, monkeypatch, _Refreshing(failure=prior))
    with pytest.raises(PeerRunsHTTPError) as caught:
        case.tracked.dispatch(dispatch=case.dispatch, grant=case.old)
    assert caught.value is not prior and caught.value.__cause__ is prior
    assert caught.value.dispatch_not_attempted is True
    assert prior.__dict__ == before


@pytest.mark.parametrize('method', ['dispatch', 'recover_dispatch'])
def test_read_only_exception_traits_become_a_controlled_phase_error(tmp_path, monkeypatch, method):
    class FixedError(RuntimeError):
        status_code = 401
        error_code = 'invalid_room_grant'
        needs_reauthorization = True

        @property
        def not_admitted(self):
            return False
    original = FixedError('refresh refused')
    case = phase_case(tmp_path, monkeypatch, _Refreshing(failure=original))
    with pytest.raises(PeerRunsHTTPError) as caught:
        getattr(case.tracked, method)(dispatch=case.dispatch, grant=case.old)
    assert caught.value.__cause__ is original
    assert caught.value.not_admitted is False
    assert caught.value.dispatch_not_attempted is (method == 'dispatch')
    assert caught.value.ambiguous is (method == 'recover_dispatch')
    assert caught.value.needs_reauthorization


def test_only_dispatch_and_replay_carry_phase_evidence():
    for method in ('probe', 'history', 'stop'):
        failure = RuntimeError('refused before sending')
        with pytest.raises(RuntimeError) as caught:
            with before_sending(method):
                raise failure
        assert caught.value is failure and not hasattr(failure, 'dispatch_not_attempted')


def test_a_respelled_grant_has_the_same_identity():
    from gateway.hosted_room_peer import room_grant_token_digest
    grant = issue_room_grant(b's' * 32, grant_id='g', room_id='r', home_install_id='h', authority_gateway_id='h',
                             authority_epoch=1, member_id='m', target_install_id='t', target_profile='default',
                             execution_policy_digest='0' * 64)
    assert room_grant_token_digest(respelled(grant)) == room_grant_token_digest(grant)
    other = issue_room_grant(b's' * 32, grant_id='g2', room_id='r', home_install_id='h', authority_gateway_id='h',
                             authority_epoch=1, member_id='m', target_install_id='t', target_profile='default',
                             execution_policy_digest='0' * 64)
    assert room_grant_token_digest(other) != room_grant_token_digest(grant)
