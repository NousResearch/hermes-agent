"""An exposure proxy attaches as the remote login it authenticated, never as the host.

The Webapp relays a browser's live socket to the profile's gateway. The gateway's native interactive
grant is the same-OS-user boundary: it adopts unowned historical rows on first resume and owns
store-wide verbs. ``gateway-internals.md`` forbids handing that to remote callers ("An exposure proxy
must not replace remote callers' credentials with its own native grant"), so the proxy mints an
``exposure`` grant naming the login instead: operator access to canonical sessions, nothing native.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest

from gateway.runtime_bootstrap import TicketStore, delegated_identity, grant_identity


def _login(provider, user, issuer=''):
    """The subject a dashboard login carries (``gateway.session_identity.authenticated_subject``)."""
    return 'auth:v1:' + json.dumps([provider, issuer, user], separators=(',', ':'))


ALICE = _login('basic', 'alice')


async def _authority_with_old_desktop_chat(tmp_path, monkeypatch, cwd=None):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'gui': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db
    db.create_session('old-desktop', source='desktop', model='old-model', cwd=str(cwd or tmp_path))
    db.append_message('old-desktop', role='user', content='OLD-DESKTOP')
    db.end_session('old-desktop', 'session_switch')
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    return await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='first')


def _attach(authority, tickets, purpose, subject):
    """Exactly gateway/run_api.py: redeem the minted ticket, build its identity, bind an operator."""
    from gateway.session_controls import AuthorityConnection
    ticket = tickets.mint(profile_id=authority.profile_id, subject=subject, purpose=purpose, scope='host')
    grant = tickets.redeem(ticket, profile_id=None, purpose=purpose)
    identity = (delegated_identity if purpose == 'exposure' else grant_identity)(grant)
    return AuthorityConnection(authority, object(), identity, operator=True)


@pytest.mark.asyncio
async def test_an_exposure_grant_acts_as_the_forwarded_login_never_as_the_host(tmp_path, monkeypatch):
    authority = await _authority_with_old_desktop_chat(tmp_path, monkeypatch)
    tickets = TicketStore('first', frozenset({authority.profile_id}))
    browser = _attach(authority, tickets, 'exposure', ALICE)
    # The same principal the proxy's REST forwards carry, with operator access to this host's sessions.
    assert browser.actor.subject == ALICE
    assert 'session:operator' in browser.actor.capabilities and browser.host_scoped
    assert browser.native_owner is False

    async def call(connection, method, **params):
        return await connection.dispatch({'id': 1, 'method': method, 'params': params})

    # No first-claim adoption of an unowned historical row, and no store-wide verb.
    refused = await call(browser, 'session.resume', session_id='old-desktop')
    assert refused.get('error', {}).get('message') == 'not_found', refused
    pruned = await call(browser, 'session.prune', never_active_days=30)
    assert pruned.get('error', {}).get('message') == 'permission_denied', pruned
    # The host's own native client adopts it (what `hermes sessions adopt` runs) ...
    host = _attach(authority, tickets, 'interactive', 'uid:1000')
    adopted = await call(host, 'session.resume', session_id='old-desktop')
    assert adopted.get('result', {}).get('session_id') == 'old-desktop', adopted
    # ... after which the browser login continues it through shared operator access.
    continued = await call(browser, 'session.resume', session_id='old-desktop')
    assert continued.get('result', {}).get('session_id') == 'old-desktop', continued
    assert [m['content'] for m in continued['result']['messages']] == ['OLD-DESKTOP']


def test_an_exposure_ticket_carries_only_a_named_remote_login(tmp_path):
    from gateway.control_socket import GatewayControlServer
    server = GatewayControlServer(tmp_path)
    server.ticket_store = TicketStore('inst-1', frozenset({str(tmp_path)}))

    def mint(**params):
        line = json.dumps({'protocol': 1, 'id': 7, 'verb': 'session-ticket', 'params': {
            'profile_id': str(tmp_path), 'instance_id': 'inst-1', 'purpose': 'exposure', **params}}).encode()
        return json.loads(server.handle_request_line(line, peer_subject='uid:501'))

    minted = mint(subject=ALICE, scope='host')
    assert minted['ok'] is True, minted
    grant = server.ticket_store.redeem(minted['result']['ticket'], profile_id=None, purpose='exposure')
    assert grant['scope'] == 'host'
    assert delegated_identity(grant)['user_id'] == 'alice'
    # Never the peer's own UID (the default), a bare label, a native-looking or an anonymous login.
    for params in ({}, {'subject': 'uid:501'}, {'subject': _login('local', 'alice')},
                   {'subject': _login('', 'alice')}, {'subject': _login('basic', ' ')}):
        reply = mint(**params)
        assert reply['ok'] is False, (params, reply)
    # Redemption refuses the same subjects on its own, whatever minted them.
    for subject in ('uid:501', _login('local', 'alice'), _login('', 'alice')):
        assert delegated_identity({'subject': subject, 'profile_id': 'p', 'instance_id': 'i'}) is None
    # Host scope stays limited to the purposes that attach a shared socket.
    with pytest.raises(PermissionError):
        server.ticket_store.mint(profile_id=str(tmp_path), subject='uid:1', purpose='native-http', scope='host')


async def _rpc(socket, method, **params):
    await socket.send(json.dumps({'jsonrpc': '2.0', 'id': method, 'method': method, 'params': params}))
    async with asyncio.timeout(15):
        while True:
            reply = json.loads(await socket.recv())
            if reply.get('id') == method:
                return reply


async def _dial_owner(home, descriptor, **binding):
    from websockets.asyncio.client import connect
    from tests.gateway.test_normal_runtime_boot import control
    grant = control(home, 'session-ticket', {'profile_id': str(home.resolve()),
                                             'instance_id': descriptor['instance_id'], **binding})
    url = descriptor['api_origin'].replace('http:', 'ws:') + '/api/ws'
    protocols = ['hermes-gateway-v1', 'hermes-gateway-ticket.' + grant['ticket']]
    async with connect(url, subprotocols=protocols, proxy=None) as socket:
        assert socket.subprotocol == 'hermes-gateway-v1'
        listed = await _rpc(socket, 'session.list')
        pruned = await _rpc(socket, 'session.prune', never_active_days=30)
    return listed, pruned


@pytest.mark.platforms("linux")
def test_a_real_owner_attaches_an_exposure_socket_as_its_login(tmp_path, model_peer):
    """The owner's listener redeems an exposure ticket at the same WS boundary as a native one, and
    the socket it opens reads sessions but cannot run a native-owner verb the native socket can."""
    from tests.gateway.fixtures.owner_process import running_owner
    with running_owner(tmp_path, model_peer) as (home, descriptor, _env):
        assert 'delegated-attach-v1' in descriptor['capabilities']
        listed, pruned = asyncio.run(_dial_owner(home, descriptor, purpose='exposure', scope='host', subject=ALICE))
        assert 'result' in listed, listed
        assert pruned.get('error', {}).get('message') == 'permission_denied', pruned
        _listed, native_pruned = asyncio.run(_dial_owner(home, descriptor, purpose='interactive'))
        assert native_pruned.get('result', {}).get('deleted') == 0, native_pruned
