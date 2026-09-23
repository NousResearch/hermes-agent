"""Legacy token authentication must become an explicit authority principal."""
import asyncio
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import time

import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus


@pytest.mark.asyncio
async def test_authenticated_token_can_create_without_identityless_permissions(tmp_path):
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.run_runtime import initialize_gateway_runtime, start_gateway_runtime_api, publish_gateway_runtime_ready
    from gateway.runtime_ownership import process_ownership
    from gateway.session_controls import AuthorityConnection
    from hermes_constants import get_hermes_home
    from hermes_cli import web_server as web

    # Session creation persists a restorable model policy, even without a turn.
    (get_hermes_home() / 'config.yaml').write_text(json.dumps({
        'model': {'provider': 'custom', 'default': 'local-wire-stub',
                  'base_url': 'http://127.0.0.1:1/v1'},
        'platform_toolsets': {'cli': []},
        'auxiliary': {'title_generation': {'enabled': False}},
    }))
    process_ownership.reserve([get_hermes_home()])
    runner = GatewayRunner(GatewayConfig())
    try:
        await initialize_gateway_runtime(runner)
        from gateway.control_socket import GatewayControlServer
        runner.session_control_server = GatewayControlServer()
        assert await runner.session_control_server.start()
        await start_gateway_runtime_api(runner)
        assert await runner.start()
        publish_gateway_runtime_ready(runner)
        url = runner.session_api.api_origin.replace('http:', 'ws:') + '/api/ws'
        with pytest.raises(InvalidStatus):
            async with connect(url + '?token=wrong'):
                pass
        async with connect(url + '?token=' + web._SESSION_TOKEN) as ws:
            await ws.send(json.dumps({'jsonrpc': '2.0', 'id': 1, 'method': 'session.create',
                                      'params': {'request_id': 'token-create', 'source': 'cli'}}))
            async with asyncio.timeout(10):
                while True:
                    reply = json.loads(await ws.recv())
                    if reply.get('id') == 1:
                        break
            assert 'result' in reply, reply
        unbound = AuthorityConnection(runner.session_authority, object(), {})
        try:
            assert not unbound.actor.capabilities
        finally:
            await unbound.close()
    finally:
        await runner.stop()
        await runner.session_control_server.stop()
        process_ownership.close()


@pytest.mark.linux_only
@pytest.mark.parametrize('layout', ['normal', 'long', 'unicode', 'long-temp'])
def test_ordinary_daemon_keeps_control_auth_and_loop_witness(tmp_path, layout):
    from gateway.control_socket import identify_gateway, resolve_client_socket_path
    from gateway.shutdown_watchdog import get_loop_tick_socket_path
    from hermes_cli.gateway import probe_gateway_loop_liveness

    async def create(origin, token):
        url = origin.replace('http:', 'ws:') + '/api/ws'
        with pytest.raises(InvalidStatus):
            async with connect(url + '?token=wrong'):
                pass
        async with connect(url + '?token=' + token) as ws:
            await ws.send(json.dumps({'jsonrpc': '2.0', 'id': 1, 'method': 'session.create',
                                     'params': {'request_id': 'daemon-token', 'source': 'cli'}}))
            async with asyncio.timeout(15):
                while True:
                    reply = json.loads(await ws.recv())
                    if reply.get('id') == 1:
                        assert 'result' in reply, reply
                        return reply['result']['session_id']

    # A normal-path control, not a substitute for the deliberately long cases.
    with tempfile.TemporaryDirectory(prefix='hgw-') as short:
        if layout == 'normal':
            home = Path(short)
        elif layout == 'unicode':
            # The character count fits even though the encoded tick address does not.
            home = Path(short) / ('界' * 20)
        else:
            home = tmp_path / ('x' * 120)
        home.mkdir(mode=0o700, exist_ok=True)
        user = tmp_path / 'user'
        user.mkdir()
        (home / 'config.yaml').write_text(json.dumps({
            'gateway': {'multiplex_profiles': False},
            'model': {'provider': 'custom', 'default': 'local-wire-stub',
                      'base_url': 'http://127.0.0.1:1/v1'},
            'terminal': {'cwd': str(tmp_path)},
            'platform_toolsets': {'cli': []},
            'auxiliary': {'title_generation': {'enabled': False}},
        }))
        root = Path(__file__).resolve().parents[2]
        token = secrets.token_urlsafe(32)
        env = {key: os.environ[key] for key in ('PATH', 'LANG', 'TZ') if key in os.environ}
        env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
                   PYTHONPATH=str(root), PYTHONUNBUFFERED='1', HERMES_DASHBOARD_SESSION_TOKEN=token)
        if layout == 'long-temp':
            temp = tmp_path / ('t' * 120)
            temp.mkdir()
            env['TMPDIR'] = str(temp)
        with (tmp_path / 'daemon.log').open('w+') as log:
            process = subprocess.Popen([sys.executable, '-m', 'gateway.run'], cwd=root, env=env,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            control_path = None
            try:
                descriptor = {}
                deadline = time.monotonic() + 35
                while process.poll() is None and time.monotonic() < deadline:
                    descriptor = identify_gateway(home) or {}
                    if descriptor.get('state') == 'ready':
                        break
                    time.sleep(.05)
                log.seek(0)
                assert descriptor.get('state') == 'ready', (descriptor, log.read())
                assert descriptor['pid'] == process.pid
                control_path = resolve_client_socket_path(home)
                assert control_path is not None
                assert control_path.stat().st_mode & 0o077 == 0
                if control_path.parent != home:
                    assert control_path.parent.stat().st_mode & 0o077 == 0
                sid = asyncio.run(create(descriptor['api_origin'], token))
                heartbeat = home / 'state' / 'gateway.heartbeat'
                while not heartbeat.exists() and time.monotonic() < deadline:
                    time.sleep(.05)
                payload = json.loads(heartbeat.read_text())
                tick_path = get_loop_tick_socket_path(home, process.pid)
                receipt = {'layout': layout, 'home_bytes': len(os.fsencode(home)),
                           'tick_chars': len(str(tick_path)), 'tick_bytes': len(os.fsencode(tick_path)),
                           'control': str(control_path), 'created_session': sid,
                           'loop_tick_socket': payload['loop_tick_socket'],
                           'loop_tick_tcp_port': payload['loop_tick_tcp_port']}
                print(json.dumps(receipt))
                assert payload['loop_tick_socket'] is True, receipt
                port = payload['loop_tick_tcp_port']
                address = ('127.0.0.1', port) if port else str(get_loop_tick_socket_path(home, process.pid))
                with socket.socket(socket.AF_INET if port else socket.AF_UNIX) as peer:
                    peer.settimeout(2)
                    peer.connect(address)
                    assert peer.recv(1) == b'1'
                # A fresh heartbeat alone must not hide a missing scheduling witness.
                assert probe_gateway_loop_liveness(process.pid, home=home, stale_after=0) == 'alive'
                process.send_signal(signal.SIGINT)
                assert process.wait(timeout=20) == 0
                assert not control_path.exists()
                assert not (home / 'gateway.sock.path').exists()
                if port:
                    with socket.socket() as peer:
                        peer.settimeout(2)
                        assert peer.connect_ex(address) != 0
                else:
                    assert not get_loop_tick_socket_path(home, process.pid).exists()
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                if control_path is not None and control_path.parent != home:
                    control_path.parent.rmdir()
