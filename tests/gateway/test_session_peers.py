"""Peer RPCs use the authenticated profile, real HTTP mirrors, and compensating RoomLinks."""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from gateway.hosted_room_peer import catalog_mapping
from gateway.hosted_room_execution_policy import execution_policy_mapping
from gateway.session_authority import SessionAuthority
from gateway.session_controls import AuthorityConnection
from gateway.session_hosted_service import CanonicalHostedRoomService
from gateway.run import _profile_runtime_scope
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch


@pytest.fixture
def peer_server():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, data, status=200):
            raw = json.dumps(data).encode()
            self.send_response(status)
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            self.server.calls.append((self.path, self.headers.get('Authorization'), None))
            if self.path == self.server.redirect_path:
                self.send_response(302)
                self.send_header('Location', self.server.redirect_target)
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            if self.headers.get('Authorization') == 'HermesRoom scoped-grant':
                if self.server.fail_probe:
                    return self.respond({'error': 'secret response'}, 503)
                return self.respond({**self.server.scope, 'target_profile': 'research',
                                     'catalog': self.server.catalog})
            if self.headers.get('Authorization') != 'Bearer ' + 'k' * 32:
                return self.respond({'error': 'credential detail'}, 401)
            if self.path.endswith('/v1/capabilities'):
                if self.server.fail_capabilities:
                    return self.respond({'error': 'private peer error'}, 503)
                return self.respond({'bot': {'profile': self.server.bot_profile, 'room_link': self.server.catalog},
                                     'features': {'runs_idempotency': {
                                         'supported': self.server.supported, 'durable': self.server.durable}}})
            if '/api/sessions?' in self.path:
                return self.respond({'data': [{'id': 'hidden-chat', 'title': 'Bot Chat', 'hidden': True}]})
            return self.respond({})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            self.server.calls.append((self.path, self.headers.get('Authorization'), body))
            if self.path == self.server.redirect_path:
                self.send_response(302)
                self.send_header('Location', self.server.redirect_target)
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            if self.path.endswith('/invitations'):
                if self.server.fail_invitation:
                    return self.respond({'error': 'private peer error'}, 503)
                self.server.scope = {k: body[k] for k in (
                    'room_id', 'home_install_id', 'authority_gateway_id', 'authority_epoch', 'member_id')}
                return self.respond({'grant': 'scoped-grant', 'target_profile': 'research',
                                     'catalog': self.server.catalog})
            if self.path.endswith('/revoke'):
                self.server.revoked.append(self.headers.get('Authorization'))
                return self.respond({'revoked': True})
            if self.path.endswith('/v1/runs'):
                self.server.keys.append(self.headers.get('Idempotency-Key'))
                return self.respond({'run_id': 'run-one', 'status': 'queued'})
            return self.respond({'message': {'content': 'remote reply'}})

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.calls, server.revoked, server.keys = [], [], []
    server.fail_probe, server.durable = False, False
    server.supported = False
    server.fail_capabilities = False
    server.fail_invitation = False
    server.bot_profile = 'research'
    server.redirect_path, server.redirect_target = None, None
    server.catalog = catalog_mapping(
        installation_id='install:remote', protocol_versions=(2,), link_modes=('direct',),
        persistent_process=True, text=True, attachments=False, target_profile='research',
        endpoint={'available': True, 'url': f'http://127.0.0.1:{server.server_port}',
                  'transport_security': 'loopback'},
        execution_policy=execution_policy_mapping(target_profile='research', config={}))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f'http://127.0.0.1:{server.server_port}'
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def homes(tmp_path, monkeypatch, peer_server):
    from agent import secret_scope
    previous = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    _, url = peer_server
    home = tmp_path / 'state'
    secondary = home / 'profiles' / 'secondary'
    secondary.mkdir(parents=True)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(home))
    (home / 'config.yaml').write_text('bot_peers:\n  launch-only:\n    url: http://127.0.0.1:1\n')
    (home / '.env').write_text('HERMES_PEER_LAUNCH_ONLY_KEY=launch-secret\n')
    (secondary / 'config.yaml').write_text(
        f'bot_peers:\n  lab:\n    url: {url}\n    profiles: [research]\n'
        f'  missing:\n    url: {url}\n')
    (secondary / '.env').write_text('HERMES_PEER_LAB_KEY=' + 'k' * 32 + '\n')
    try:
        yield home, secondary
    finally:
        secret_scope.set_multiplex_active(previous)


async def call(home, method, params=None, *, service=False):
    with SessionDB(home / 'state.db') as db:
        epoch = begin_runtime_epoch(db, instance_id='owner')
        authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id=str(home),
                                     instance_id='owner', db=db, epoch=epoch)
        connection = AuthorityConnection(authority, object(), {'user_id': 'reader'})
        if service:
            authority.hosted_room_service = service(authority)
        try:
            return await connection.dispatch({'id': 1, 'method': method, 'params': params or {}})
        finally:
            await connection.close()


def test_rollback_attempts_other_grants_after_a_malformed_invitation():
    from gateway.session_peers_mixed import _rollback
    revoked = []
    client = SimpleNamespace(revoke_grant=lambda **kw: revoked.append(kw['grant']))
    incomplete = _rollback(None, 'room', [(client, {'grant': 'first'}), (client, {}),
                                        (client, {'grant': 'last'})], False)
    assert incomplete and revoked == ['last', 'first']


def test_roster_is_profile_scoped_and_has_no_transport_secrets(homes, peer_server):
    _, secondary = homes
    before = dict(os.environ)
    response = asyncio.run(call(secondary, 'peers.roster'))
    assert 'result' in response, response
    bots = response['result']['bots']
    lab, missing = bots
    assert lab['profile'] == 'research' and lab['reachability'] == 'reachable'
    assert lab['installation_id'] == 'install:remote'
    assert lab['room_link']['protocol_versions'] == [2]
    assert missing['reachability'] == 'unauthenticated'
    encoded = json.dumps(response)
    for secret in ('127.0.0.1', 'http:', 'HERMES_PEER_', 'k' * 32, 'launch-secret', 'launch-only'):
        assert secret not in encoded
    assert os.environ == before
    foreign = asyncio.run(call(homes[0], 'peers.dm', {
        'route_id': lab['route_id'], 'message': 'private', 'request_id': 'r'}))
    assert foreign['error']['message'] == 'peer_route_unavailable'


@pytest.mark.parametrize('identity', ['k' * 32, 'https://private.example', 'HERMES_PEER_LAB_KEY'])
def test_roster_rejects_private_data_reflected_in_catalog(homes, peer_server, identity):
    server, _ = peer_server
    server.catalog = catalog_mapping(
        installation_id=identity, protocol_versions=(2,), link_modes=('direct',),
        persistent_process=True, text=True, attachments=False, target_profile='research',
        execution_policy=server.catalog['execution_policy'])
    response = asyncio.run(call(homes[1], 'peers.roster'))
    assert response['result']['bots'][0]['reachability'] == 'unreachable'
    assert identity not in json.dumps(response)


def test_roster_refuses_a_different_profile_from_the_selected_mirror(homes, peer_server):
    server, _ = peer_server
    bot = asyncio.run(call(homes[1], 'peers.roster'))['result']['bots'][0]
    server.bot_profile = 'default'
    response = asyncio.run(call(homes[1], 'peers.roster'))
    assert response['result']['bots'][0]['reachability'] == 'unreachable'
    response = asyncio.run(call(homes[1], 'peers.dm', dict(
        route_id=bot['route_id'], message='private', request_id='changed-mirror')))
    assert response['error']['message'] == 'peer_delivery_failed'
    assert not any(c[2] is not None for c in server.calls)


@pytest.mark.parametrize('cross_origin', [False, True])
@pytest.mark.parametrize('path', ['/p/research/v1/capabilities', '/p/research/api/sessions'])
def test_peer_discovery_and_dm_never_follow_redirects(homes, peer_server, path, cross_origin):
    server, url = peer_server
    bot = asyncio.run(call(homes[1], 'peers.roster'))['result']['bots'][0]
    server.calls.clear()
    query = '?limit=200&title=Bot+Chat&include_hidden=1' if path.endswith('/sessions') else ''
    server.redirect_path = path + query
    server.redirect_target = (url.replace('127.0.0.1', 'localhost') if cross_origin else url) + '/p/default/private'
    if path.endswith('/capabilities'):
        result = asyncio.run(call(homes[1], 'peers.roster'))
        assert result['result']['bots'][0]['reachability'] == 'unreachable'
    else:
        result = asyncio.run(call(homes[1], 'peers.dm', dict(
            route_id=bot['route_id'], request_id='redirect', message='private')))
        assert result['error']['message'] == 'peer_delivery_failed'
    assert not any('/p/default/' in c[0] for c in server.calls)


def test_roomlink_invitation_never_follows_an_advertised_redirect(peer_server):
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError
    server, url = peer_server
    server.redirect_path = '/p/research/v1/room-members/invitations'
    server.redirect_target = url + '/p/default/private'
    client = PeerRunsHTTPClient(base_url=url + '/p/research', api_key='k' * 32)
    with pytest.raises(PeerRunsHTTPError):
        client.issue_invitation(room_id='r', home_install_id='install:home',
                                authority_gateway_id='install:home', authority_epoch=1,
                                member_id='m', grant_id='g')
    assert [c[0] for c in server.calls] == [server.redirect_path]


@pytest.mark.platforms('windows')
def test_windows_route_and_replay_identity_ignore_home_case(homes, peer_server):
    from gateway.session_peers import _route_id, dm
    server, url = peer_server
    home = homes[1]
    alternate = Path(str(home).upper())
    entry = {'url': url}
    route = _route_id(home, 'lab', entry, 'research')
    assert route == _route_id(alternate, 'lab', entry, 'research')
    server.supported = True
    params = dict(route_id=route, request_id='same', message='hello')
    with _profile_runtime_scope(home):
        dm(home, params, 'reader')
    with _profile_runtime_scope(alternate):
        dm(alternate, params, 'reader')
    assert len(server.keys) == 2 and server.keys[0] == server.keys[1]


@pytest.mark.parametrize(('supported', 'durable'), [(False, False), (True, False), (True, True)])
def test_dm_uses_exact_remote_profile_hidden_bot_chat_and_replay_contract(homes, peer_server, supported, durable):
    server, _ = peer_server
    server.durable = durable
    server.supported = supported
    home = homes[1]
    bot = asyncio.run(call(home, 'peers.roster'))['result']['bots'][0]
    params = dict(route_id=bot['route_id'], message='hello target', request_id='same-request')
    first = asyncio.run(call(home, 'peers.dm', params))
    assert 'result' in first, first
    assert first['result']['session_id'] == 'hidden-chat'
    assert first['result']['idempotency_supported'] is supported
    assert first['result'].get('idempotency_durable', False) is durable
    sends = [c for c in server.calls if c[2] is not None]
    assert len(sends) == 1
    path, auth, body = sends[0]
    assert path.startswith('/p/research/') and auth == 'Bearer ' + 'k' * 32
    assert body['session_id' if supported else 'message'] == ('hidden-chat' if supported else 'hello target')
    if supported:
        asyncio.run(call(home, 'peers.dm', params))
        assert server.keys[0] == server.keys[1]
    else:
        assert first['result']['reply'] == 'remote reply'


def test_dm_replay_never_falls_back_to_chat_when_discovery_fails(homes, peer_server):
    server, _ = peer_server
    server.supported = True
    home = homes[1]
    bot = asyncio.run(call(home, 'peers.roster'))['result']['bots'][0]
    params = dict(route_id=bot['route_id'], request_id='retry', message='one turn')
    assert 'result' in asyncio.run(call(home, 'peers.dm', params))
    server.fail_capabilities = True
    response = asyncio.run(call(home, 'peers.dm', params))
    assert response['error']['message'] == 'peer_delivery_failed'
    assert len(server.keys) == 1
    assert not any(c[0].endswith('/chat') for c in server.calls)


@pytest.mark.parametrize('failure', ['none', 'probe', 'create', 'persist', 'published', 'config', 'profile', 'invite'])
def test_mixed_room_registers_scoped_route_or_compensates(homes, peer_server, monkeypatch, failure):
    from gateway import hosted_rooms as rooms
    server, _ = peer_server
    home = homes[1]
    bot = asyncio.run(call(home, 'peers.roster'))['result']['bots'][0]
    server.fail_probe = failure == 'probe'
    server.fail_invitation = failure == 'invite'
    if failure == 'profile':
        with _profile_runtime_scope(home):
            server.catalog = catalog_mapping(
                installation_id='install:remote', protocol_versions=(2,), link_modes=('direct',),
                persistent_process=True, text=True, attachments=False, target_profile='default',
                execution_policy=execution_policy_mapping(target_profile='default', config={}))
    captured = []

    def make_service(authority):
        service = CanonicalHostedRoomService(authority, asyncio.get_running_loop())
        monkeypatch.setattr(service.runtime, 'status', lambda: {'running': True, 'stopping': False})
        if failure == 'create':
            monkeypatch.setattr(service, 'create_room', lambda **kw: (_ for _ in ()).throw(ValueError('bad')))
        if failure == 'config':
            from hermes_cli.config_backend import ConfigBackendUnavailable
            monkeypatch.setattr(service, 'create_room',
                                lambda **kw: (_ for _ in ()).throw(ConfigBackendUnavailable('private detail')))
        if failure == 'persist':
            monkeypatch.setattr(service, '_save_link', lambda **kw: (_ for _ in ()).throw(OSError('disk full')))
        if failure == 'published':
            register = service.register_peer_route

            def fail_after_publish(**kwargs):
                register(**kwargs)
                raise OSError('failed after publication')

            monkeypatch.setattr(service, 'register_peer_route', fail_after_publish)
        captured.append(service)
        return service

    params = dict(room_id='mixed-test', name='Mixed', members=[
        dict(kind='local', profile='secondary', member_id='local', handle='local'),
        dict(kind='peer', route_id=bot['route_id'], member_id='remote', handle='remote')])
    response = asyncio.run(call(home, 'groups.create_mixed', params, service=make_service))
    with _profile_runtime_scope(home):
        if failure == 'none':
            assert 'result' in response, response
            route = captured[0].peer_routes[('mixed-test', 'remote')]
            assert route.target_profile == 'research' and route.grant == 'scoped-grant'
            assert captured[0].peer_clients[('mixed-test', 'remote')].api_key == ''
            assert rooms.room_state(home / 'state.db', room_id='mixed-test')['members'][1]['target']['kind'] == 'peer'
            assert not server.revoked
            invitation = next(c[2] for c in server.calls if c[0].endswith('/invitations'))
            room = response['result']['room']
            assert invitation['home_install_id'] == rooms.local_authority_gateway_id()
            assert invitation['authority_gateway_id'] == room['authority_gateway_id']
            assert invitation['authority_epoch'] == room['authority_epoch']
            assert invitation['home_install_id'] != str(home)
        else:
            reason = 'mixed_room_rollback_incomplete' if failure == 'invite' else 'mixed_room_setup_failed'
            assert response['error']['message'] == reason, response
            if failure == 'invite':
                assert not server.revoked
            else:
                assert server.revoked and set(server.revoked) == {'HermesRoom scoped-grant'}
            assert not captured[0].peer_routes
            assert rooms.list_rooms(home / 'state.db') == []
    assert 'scoped-grant' not in json.dumps(response)
