"""Private native grants authenticate HTTP, never another purpose or profile."""
import asyncio
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import httpx
import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

from tests.gateway.test_normal_runtime_boot import control


@contextmanager
def daemon(tmp_path):
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700, exist_ok=True)
    user.mkdir(exist_ok=True)
    for name in ('other', 'current'):
        sibling = home / 'profiles' / name
        sibling.mkdir(parents=True, exist_ok=True)
        (sibling / 'config.yaml').write_text(json.dumps({
            'model': {'provider': 'custom', 'default': 'sibling-' + name}}))
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False},
        'model': {'provider': 'custom', 'default': 'native-http-fixture',
                  'base_url': 'http://127.0.0.1:1/v1'},
        'auxiliary': {'title_generation': {'enabled': False}},
    }))
    root = Path(__file__).resolve().parents[2]
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ') if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
               PYTHONPATH=str(root), PYTHONUNBUFFERED='1',
               HERMES_DASHBOARD_SESSION_TOKEN='normal-http-owner')
    with (tmp_path / 'gateway.log').open('w+') as log:
        process = subprocess.Popen([sys.executable, '-m', 'gateway.run'], cwd=root, env=env,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 40
            descriptor = {}
            while process.poll() is None and time.monotonic() < deadline:
                try:
                    descriptor = control(home, 'identify')
                    if descriptor.get('state') == 'ready':
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(.1)
            log.seek(0)
            assert descriptor.get('state') == 'ready', (descriptor, log.read())
            yield home, descriptor
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def ticket(home, descriptor, purpose='native-http', **overrides):
    return control(home, 'session-ticket', {
        'profile_id': str(home.resolve()), 'instance_id': descriptor['instance_id'],
        'purpose': purpose, **overrides,
    })['ticket']


def headers(token):
    return {'X-Hermes-Gateway-Ticket': token}


async def reject_ws(origin, token):
    with pytest.raises(InvalidStatus):
        async with connect(origin.replace('http:', 'ws:') + '/api/ws', subprotocols=[
                'hermes-gateway-v1', 'hermes-gateway-ticket.' + token]):
            pass


@pytest.mark.linux_only
def test_native_http_grants_are_single_use_and_bound_to_daemon(tmp_path):
    with daemon(tmp_path) as (home, descriptor), httpx.Client(
            base_url=descriptor['api_origin'], trust_env=False) as client:
        assert client.get('/api/config').status_code == 401
        normal = {'Authorization': 'Bearer normal-http-owner'}
        assert client.get('/api/config', headers=normal).status_code == 200
        assert 'sibling-other' in client.get('/api/config?profile=other', headers=normal).text
        sibling_config = home / 'profiles' / 'other' / 'config.yaml'
        sibling_before = sibling_config.read_bytes()
        first = ticket(home, descriptor)
        response = client.get('/api/config', headers=headers(first))
        assert response.status_code == 200, response.text
        assert 'native-http-fixture' in response.text
        assert client.get('/api/config', headers=headers(first)).status_code == 401
        assert client.get('/api/config', headers={**normal, **headers('invalid')}).status_code == 401
        for origin in ('https://evil.example', '', 'null'):
            assert client.get('/api/config', headers={**headers(ticket(home, descriptor)),
                                                     'Origin': origin}).status_code == 401
        assert client.get('/api/config', headers=[
            ('X-Hermes-Gateway-Ticket', ticket(home, descriptor)),
            ('X-Hermes-Gateway-Ticket', ticket(home, descriptor)),
        ]).status_code == 401
        for purpose in ('interactive', 'exposure', 'worker-adoption'):
            assert client.get('/api/config', headers=headers(ticket(home, descriptor, purpose))).status_code == 401
        asyncio.run(reject_ws(descriptor['api_origin'], ticket(home, descriptor)))
        for overrides in ({'instance_id': 'stale'}, {'profile_id': str(tmp_path / 'other')}):
            with pytest.raises(AssertionError, match='PermissionError'):
                ticket(home, descriptor, **overrides)
        for path in ('/api/config?profile=other', '/api/config?profile=current&profile=other',
                     '/api/profiles/other/soul', '/api/sessions?profile=current'):
            assert client.get(path, headers=headers(ticket(home, descriptor))).status_code == 403
        assert client.put('/api/config', json={'profile': 'other', 'config': {}},
                           headers=headers(ticket(home, descriptor))).status_code == 403
        for content_type in ('application/json', 'application/problem+json', ''):
            response = client.put('/api/config', content=json.dumps({'profile': 'other', 'config': {}}),
                                   headers={**headers(ticket(home, descriptor)), 'Content-Type': content_type})
            assert response.status_code == 403, (content_type, response.text)
        for path in ('/api/profiles/sessions', '/api/profiles/sessions/sidebar',
                     '/api/profiles/projects/tree', '/api/cron/jobs'):
            assert client.get(path, headers=headers(ticket(home, descriptor))).status_code == 403
        assert sibling_config.read_bytes() == sibling_before
        assert client.get('/api/config?profile=default',
                          headers=headers(ticket(home, descriptor))).status_code == 200
        assert client.get('/api/profiles/sessions/sidebar?recents_profile=default',
                          headers=headers(ticket(home, descriptor))).status_code == 200
        assert client.get('/api/config?profile=current',
                          headers=headers(ticket(home, descriptor))).status_code == 200
        assert client.get('/api/profiles', headers=headers(ticket(home, descriptor))).status_code == 200
        updated = client.put('/api/config', json={'profile': 'current', 'config': {
            'display': {'skin': 'ares'}}}, headers=headers(ticket(home, descriptor)))
        assert updated.status_code == 200, updated.text
        assert client.get('/api/config', headers=headers(ticket(home, descriptor))).json()['display']['skin'] == 'ares'
        import yaml
        assert yaml.safe_load((home / 'config.yaml').read_text())['display']['skin'] == 'ares'
        query_ticket = ticket(home, descriptor)
        assert client.get('/api/config', params={'ticket': query_ticket}).status_code == 401
        assert client.get('/api/config', headers={'Cookie': 'X-Hermes-Gateway-Ticket=' + query_ticket}).status_code == 401
        expired = ticket(home, descriptor)
        time.sleep(31)  # Real production TTL, not a patched clock or shortened grant.
        assert client.get('/api/config', headers=headers(expired)).status_code == 401
        stale = ticket(home, descriptor)
        old_instance = descriptor['instance_id']
        assert 'ticket' not in json.dumps(control(home, 'identify')).lower()
    with daemon(tmp_path) as (home, descriptor), httpx.Client(
            base_url=descriptor['api_origin'], trust_env=False) as client:
        assert descriptor['instance_id'] != old_instance
        assert client.get('/api/config', headers=headers(stale)).status_code == 401
        assert client.get('/api/config', headers=headers(ticket(home, descriptor))).status_code == 200
    print(json.dumps({'native_http': 'ordinary daemon config and profiles HTTP passed',
                      'replay_expiry_restart_purpose_origin_profile': 'rejected'}))


@pytest.mark.asyncio
async def test_native_http_preserves_gated_auth_and_actual_socket_boundary(tmp_path):
    import socket
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.run_api import start_gateway_api
    from gateway.run_runtime import initialize_gateway_runtime, publish_gateway_runtime_ready
    from gateway.runtime_ownership import process_ownership
    from hermes_cli import web_server as web
    from hermes_cli.dashboard_auth import register_provider, clear_providers
    from hermes_cli.dashboard_auth.cookies import SESSION_AT_COOKIE
    from hermes_constants import get_hermes_home
    from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

    home = get_hermes_home()
    (home / 'config.yaml').write_text('dashboard:\n  public_url: https://gateway.example.test\n')
    provider = StubAuthProvider()
    register_provider(provider)
    login = provider.start_login(redirect_uri='http://127.0.0.1/callback')
    from urllib.parse import parse_qs
    pkce = parse_qs(login.cookie_payload['hermes_session_pkce'].replace(';', '&'))
    session = provider.complete_login(code='stub_code', state=pkce['state'][0],
                                      code_verifier=pkce['verifier'][0],
                                      redirect_uri='http://127.0.0.1/callback')
    process_ownership.reserve([home])
    runner = GatewayRunner(GatewayConfig())
    try:
        await initialize_gateway_runtime(runner)
        from gateway.control_socket import GatewayControlServer
        runner.session_control_server = GatewayControlServer()
        assert await runner.session_control_server.start()
        runner.session_api = await start_gateway_api(runner, host='0.0.0.0')
        port = runner.session_api.socket.getsockname()[1]
        def mint():
            return runner.session_ticket_store.mint(profile_id=str(home), subject='uid:fixture',
                                                    purpose='native-http')
        async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{port}', trust_env=False) as client:
            assert (await client.get('/api/config', headers=headers(mint()))).status_code == 503
            assert await runner.start()
            publish_gateway_runtime_ready(runner)
            assert web.app.state.auth_required is True
            bearer = {'Authorization': 'Bearer ' + session.access_token}
            cookie = {'Cookie': f'{SESSION_AT_COOKIE}={session.access_token}'}
            for normal in (bearer, cookie):
                assert (await client.get('/api/config', headers=normal)).status_code == 200
                assert (await client.get('/api/config', headers={**normal, **headers('bad')})).status_code == 401
            assert (await client.get('/api/config', headers={
                'Authorization': 'Bearer ' + web._SESSION_TOKEN})).status_code == 401
            assert (await client.get('/api/config', headers=headers(mint()))).status_code == 200
            # Trusted proxy rewrites client.host; native admission must retain actual peer.
            assert (await client.get('/api/config', headers={**headers(mint()),
                'X-Forwarded-For': '203.0.113.9'})).status_code == 200
            for origin in ('', 'https://gateway.example.test'):
                assert (await client.get('/api/config', headers={**headers(mint()),
                    'Origin': origin, **cookie})).status_code == 401
            # Real non-loopback local interface, not a forged ASGI scope or testclient IP.
            import ipaddress
            import psutil
            addresses = {info.address for entries in psutil.net_if_addrs().values()
                         for info in entries if info.family == socket.AF_INET
                         and not ipaddress.ip_address(info.address).is_loopback}
            assert addresses, 'real non-loopback interface required for this invariant'
            address = sorted(addresses)[0]
            async with httpx.AsyncClient(base_url=f'http://{address}:{port}', trust_env=False) as remote:
                assert (await remote.get('/api/config', headers={**headers(mint()),
                    'X-Forwarded-For': '127.0.0.1', 'Host': f'127.0.0.1:{port}'})).status_code == 401
                assert (await remote.get('/api/config', headers={**bearer,
                    'Host': f'127.0.0.1:{port}'})).status_code == 200
            runner.session_runtime_descriptor['state'] = 'draining'
            assert (await client.get('/api/config', headers=headers(mint()))).status_code == 503
    finally:
        await runner.stop()
        await runner.session_control_server.stop()
        process_ownership.close()
        clear_providers()
    print(json.dumps({'normal_oauth_bearer_cookie': 'passed',
                      'native_gated_auth_raw_peer_readiness': 'passed'}))


@pytest.mark.linux_only
def test_native_http_principal_satisfies_route_local_token_policy(tmp_path):
    """Routes that call ``_require_token`` honor the verified native owner (F16).

    Local Desktop runs with authMode=native and no static token; the outer seam
    admitted the ticket, so the route-local check must not 401 the same request.
    """
    with daemon(tmp_path) as (home, descriptor), httpx.Client(
            base_url=descriptor['api_origin'], trust_env=False, timeout=30) as client:
        hub = client.get('/api/dashboard/plugins/hub', headers=headers(ticket(home, descriptor)))
        assert hub.status_code == 200, hub.text
        oauth = client.delete('/api/providers/oauth/__not_a_real_provider__',
                              headers=headers(ticket(home, descriptor)))
        assert oauth.status_code == 400, oauth.text
        assert client.delete('/api/providers/oauth/__not_a_real_provider__').status_code == 401
