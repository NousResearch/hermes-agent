"""A Webapp browser's live socket runs on the profile's gateway, as the browser's own login.

Before the cutover the Webapp hosted turns itself (a standalone ``serve`` backend with no session
authority), so it and the messaging gateway both wrote each profile's store. The browser now gets a
one-use Webapp ticket for its gateway (``POST /api/webapp/gateway-ticket``), the Webapp relays the
socket to the owner under a delegated exposure grant, and nothing runs in the Webapp process.
"""
import json
import sqlite3
import time

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tests.gateway.fixtures.owner_process import running_owner

BROWSER_LOGIN = 'auth:v1:["session-token","","legacy-token-owner"]'


def _webapp(monkeypatch, home):
    from hermes_cli import web_server as server
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(server.app.state, 'ui_surface', 'webapp', raising=False)
    monkeypatch.setattr(server.app.state, 'auth_required', False, raising=False)
    monkeypatch.setattr(server.app.state, 'bound_host', 'testclient', raising=False)
    # Real middleware, routers and gates; the lifespan is not this test's subject.
    return TestClient(server.app, base_url='http://testclient'), server._SESSION_TOKEN


def _rpc(ws, method, **params):
    ws.send_json({'jsonrpc': '2.0', 'id': method, 'method': method, 'params': params})
    while True:
        reply = ws.receive_json()
        if reply.get('id') == method:
            assert 'result' in reply, reply
            return reply['result']


@pytest.mark.platforms("linux")
def test_a_browser_socket_runs_its_turn_on_the_gateway_as_its_login(tmp_path, model_peer, monkeypatch):
    with running_owner(tmp_path, model_peer) as (home, descriptor, _env):
        # The suite stubs this process's code identity; the owner booted this checkout. Same code,
        # so attaching never restarts it (that ladder is test_webapp_gateway_owner's).
        from hermes_cli import version_info
        monkeypatch.setattr(version_info, 'get_code_identity', lambda refresh=False: {'sha': descriptor['code_sha']})
        client, token = _webapp(monkeypatch, home)
        minted = client.post('/api/webapp/gateway-ticket', headers={'X-Hermes-Session-Token': token, 'host': 'testclient'})
        assert minted.status_code == 200, minted.text
        assert minted.headers['cache-control'] == 'no-store'
        protocols = ['hermes-gateway-v1', 'hermes-gateway-ticket.' + minted.json()['ticket']]
        with client.websocket_connect('/api/ws?native_dial=browser', subprotocols=protocols,
                                      headers={'host': 'testclient'}) as ws:
            assert ws.accepted_subprotocol == 'hermes-gateway-v1'
            ready = ws.receive_json()
            assert ready['params']['type'] == 'gateway.ready' and ready['params']['payload']['session_authority'] is True
            sid = _rpc(ws, 'session.create', request_id='browser-launch', source='gui')['session_id']
            _rpc(ws, 'session.resume', session_id=sid)
            assert _rpc(ws, 'prompt.submit', session_id=sid, input_id='browser-input', text='RELAYED_WS_SHARED')['status'] == 'queued'
            deadline = time.monotonic() + 25
            # The fixture model acknowledges a WS_SHARED prompt by name (tests/gateway/fixtures).
            while 'LOCAL_ACK_WS_SHARED' not in json.dumps(_rpc(ws, 'session.resume', session_id=sid)['messages']):
                assert time.monotonic() < deadline, 'the owner never answered the relayed turn'
        # The owner ran it, admitted for the browser's login; the Webapp hosted nothing.
        from tui_gateway import server as sidecar
        assert not sidecar._sessions
        with sqlite3.connect(f"file:{home / 'state.db'}?mode=ro", uri=True) as db:
            principals = {row[0] for row in db.execute(
                'SELECT principal_id FROM session_admissions WHERE target_session_id=?', (sid,))}
        assert principals == {BROWSER_LOGIN}, principals
        # A tab from before the cutover (the old ?token= contract) is told to reload, never hosted.
        with pytest.raises(WebSocketDisconnect) as legacy:
            with client.websocket_connect('/api/ws?token=' + token, headers={'host': 'testclient'}):
                pass
        assert legacy.value.code == 4426
        # The Bot Desktop screen ticket is the owner's (its display.observe minted it): the screen
        # socket is relayed to the owner's own route, whose verdict comes back unchanged.
        with client.websocket_connect('/api/display/ws?display_ticket=not-minted', headers={'host': 'testclient'}) as screen:
            with pytest.raises(WebSocketDisconnect) as refused:
                screen.receive_bytes()
        assert refused.value.code == 4401


def test_the_gateway_ticket_route_needs_the_webapp_credential(monkeypatch, tmp_path):
    client, _token = _webapp(monkeypatch, tmp_path)
    assert client.post('/api/webapp/gateway-ticket', headers={'host': 'testclient'}).status_code == 401


@pytest.fixture
def owner_calls(monkeypatch):
    """Every owner lookup and start the Webapp makes; no gateway runs."""
    from hermes_cli import gateway_runtime
    from hermes_cli.gateway_runtime import GatewayDiscovery
    calls = {'discover': [], 'ensure': []}

    def discover(home, timeout=2.0):
        calls['discover'].append(str(home))
        return GatewayDiscovery('absent')

    def ensure(home, **_kw):
        calls['ensure'].append(str(home))
        return GatewayDiscovery('absent')

    monkeypatch.setattr(gateway_runtime, 'discover_gateway_endpoint', discover)
    monkeypatch.setattr(gateway_runtime, 'ensure_gateway_runtime', ensure)
    return calls


def _screen_close_code(client, query):
    with pytest.raises(WebSocketDisconnect) as closed:
        with client.websocket_connect('/api/display/ws?' + query, headers={'host': 'testclient'}) as screen:
            screen.receive_bytes()
    return closed.value.code


def test_a_screen_socket_without_a_ticket_is_refused_before_any_owner_lookup(monkeypatch, tmp_path, owner_calls):
    (tmp_path / 'config.yaml').write_text('model:\n  default: x\n', encoding='utf-8')
    client, _token = _webapp(monkeypatch, tmp_path)
    assert _screen_close_code(client, 'profile=default') == 4401
    assert owner_calls == {'discover': [], 'ensure': []}


def test_a_screen_socket_never_starts_a_gateway(monkeypatch, tmp_path, owner_calls):
    """The owner mints screen tickets (``display.observe``), so a real one arrives while it runs: the
    relay only looks the owner up, and a socket the owner has not authenticated starts nothing."""
    (tmp_path / 'config.yaml').write_text('model:\n  default: x\n', encoding='utf-8')
    client, _token = _webapp(monkeypatch, tmp_path)
    assert _screen_close_code(client, 'profile=default&display_ticket=not-minted') == 1013
    assert len(owner_calls['discover']) == 1
    assert owner_calls['ensure'] == []
