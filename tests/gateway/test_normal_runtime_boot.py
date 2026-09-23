"""The ordinary no-platform process must earn its bootstrap readiness."""
import asyncio
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus


def control(home, verb, params=None):
    """Bootstrap control channel: private named pipe on Windows, AF_UNIX elsewhere."""
    request = json.dumps({'protocol': 1, 'id': 1, 'verb': verb, 'params': params or {}}).encode() + b'\n'
    if os.name == 'nt':
        from gateway.runtime_bootstrap_windows import query_runtime_control
        raw = query_runtime_control(home, request, 2)
    else:
        pointer = home / 'gateway.sock.path'
        path = pointer.read_text(encoding='utf-8').strip() if pointer.exists() else str(home / 'gateway.sock')
        with socket.socket(socket.AF_UNIX) as peer:
            peer.settimeout(2)
            peer.connect(path)
            peer.sendall(request)
            with peer.makefile('rb') as stream:
                raw = stream.readline()
    reply = json.loads(raw.split(b'\n', 1)[0])
    assert reply['ok'], reply
    return reply['result']


@pytest.fixture
def model_peer():
    from http.server import ThreadingHTTPServer
    import threading
    from tests.gateway.fixtures.shared_authority_peer import ModelPeer
    server = ThreadingHTTPServer(('127.0.0.1', 0), ModelPeer)
    server.requests, server.metadata_requests = [], []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


async def handshake(home, descriptor):
    url = descriptor['api_origin'].replace('http:', 'ws:') + '/api/ws'
    with pytest.raises(InvalidStatus):
        async with connect(url):
            pass
    binding = {'profile_id': str(home.resolve()), 'instance_id': descriptor['instance_id'],
               'purpose': 'interactive'}
    grant = control(home, 'session-ticket', binding)
    protocols = ['hermes-gateway-v1', 'hermes-gateway-ticket.' + grant['ticket']]
    async with connect(url, subprotocols=protocols) as ws:
        await ws.send(json.dumps({'jsonrpc': '2.0', 'id': 7, 'method': 'session.resume',
                                  'params': {'session_id': 'not-registered'}}))
        while True:
            reply = json.loads(await asyncio.wait_for(ws.recv(), 10))
            if reply.get('id') == 7:
                break
        assert reply['error']['message'] == 'not_found', reply
        assert ws.subprotocol == 'hermes-gateway-v1'
        async def rpc(socket, method, expected_error=None, **params):
            await socket.send(json.dumps({'jsonrpc': '2.0', 'id': method, 'method': method, 'params': params}))
            async with asyncio.timeout(15):
                while True:
                    result = json.loads(await socket.recv())
                    if result.get('id') == method:
                        if expected_error is not None:
                            assert result.get('error', {}).get('message') == expected_error, result
                            return result['error']
                        assert 'result' in result, result
                        return result['result']

        status = await rpc(ws, 'setup.status')
        assert status['provider_configured'] is True, status
        readiness = await rpc(ws, 'setup.runtime_check', provider='custom')
        assert readiness['ok'] is True and readiness['provider'] == 'custom', readiness
        for method in ('setup.status', 'setup.runtime_check'):
            await rpc(ws, method, expected_error='profile_mismatch', profile='foreign-profile')
        created = await rpc(ws, 'session.create', request_id='ordinary-launch', source='cli')
        sid = created['session_id']
        assert sid == created['stored_session_id']
        initial = await rpc(ws, 'session.resume', session_id=sid)
        admitted = await rpc(ws, 'prompt.submit', session_id=sid, input_id='ordinary-input', text='WS_SHARED')
        assert admitted['status'] == 'queued', admitted
    # A fresh control ticket and socket must see the same committed execution.
    fresh = control(home, 'session-ticket', binding)
    async with connect(url, subprotocols=['hermes-gateway-v1',
                                        'hermes-gateway-ticket.' + fresh['ticket']]) as viewer:
        async with asyncio.timeout(25):
            while True:
                snapshot = await rpc(viewer, 'session.resume', session_id=sid)
                if 'LOCAL_ACK_WS_SHARED' in json.dumps(snapshot.get('messages', [])):
                    break
                await asyncio.sleep(.05)
        assert snapshot['stored_session_id'] == sid
        # Transcript persistence precedes terminal publication; await the actual
        # replay receipt rather than assuming a persisted reply means settlement.
        async with asyncio.timeout(25):
            while True:
                replay = await rpc(viewer, 'session.events.since', session_id=sid,
                                   replay_epoch=initial['replay_epoch'],
                                   last_sequence=initial['last_sequence'])
                assert not replay['snapshot_required'], replay
                completed = [event for event in replay['events'] if event['type'] == 'message.complete']
                if completed:
                    break
                await asyncio.sleep(.05)
        assert len(completed) == 1, replay
        assert 'LOCAL_ACK_WS_SHARED' in completed[0]['payload']['text'], replay
        snapshot = await rpc(viewer, 'session.resume', session_id=sid)
        assert completed[0]['seq'] <= snapshot['last_sequence']
        edit = {'session_id': sid, 'request_id': 'ordinary-rename',
                'expected_revision': snapshot['revision'], 'operation': 'rename',
                'payload': {'title': 'Shared metadata receipt'}}
        renamed = await rpc(viewer, 'session.mutate', **edit)
        assert renamed['title'] == edit['payload']['title'], renamed
    reconnect = control(home, 'session-ticket', binding)
    async with connect(url, subprotocols=['hermes-gateway-v1',
                                        'hermes-gateway-ticket.' + reconnect['ticket']]) as viewer:
        assert await rpc(viewer, 'session.mutate', **edit) == renamed
        await rpc(viewer, 'session.mutate', expected_error='revision_conflict',
                  **{**edit, 'request_id': 'stale-rename', 'payload': {'title': 'Stale overwrite'}})
        listing = await rpc(viewer, 'session.list')
        assert next(row for row in listing['sessions'] if row['session_id'] == sid)['title'] == renamed['title']
        print(json.dumps({'ordinary_created_session': sid, 'reply_persisted': True, 'reconnect_event_replay': True,
                          'mutation_retry_after_reconnect': True, 'stale_mutation_rejected': True}))
    with pytest.raises(InvalidStatus):
        async with connect(url, subprotocols=protocols):
            pass


@pytest.mark.linux_only
def test_normal_entrypoint_earns_authenticated_authority_readiness(tmp_path, model_peer):
    home = tmp_path / 'state'
    home.mkdir(mode=0o700)
    user = tmp_path / 'user'
    user.mkdir()
    model_url = f'http://127.0.0.1:{model_peer.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False},
        'model': {'provider': 'custom', 'default': 'local-wire-stub', 'base_url': model_url},
        'auxiliary': {'title_generation': {'enabled': False}},
    }))
    root = Path(__file__).resolve().parents[2]
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
               PYTHONPATH=str(root), PYTHONUNBUFFERED='1',
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=model_url)
    with (tmp_path / 'gateway.log').open('w+') as log:
        process = subprocess.Popen([sys.executable, '-m', 'gateway.run'], cwd=root, env=env,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        descriptor = {}
        try:
            deadline = time.monotonic() + 35
            while process.poll() is None and time.monotonic() < deadline:
                try:
                    descriptor = control(home, 'identify')
                    if descriptor.get('state') == 'ready':
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(.1)
            log.flush()
            log.seek(0)
            assert descriptor.get('state') == 'ready', (descriptor, log.read())
            assert descriptor['authority_epoch'] > 0
            import sqlite3
            with sqlite3.connect(f"file:{home / 'state.db'}?mode=ro", uri=True) as db:
                assert db.execute('SELECT epoch, instance_id FROM runtime_epoch').fetchone() == (
                    descriptor['authority_epoch'], descriptor['instance_id'])
            assert descriptor['served_profiles'] == [{'profile_id': str(home), 'home': str(home)}]
            assert 'session-authority-v1' in descriptor['capabilities']
            asyncio.run(handshake(home, descriptor))
            assert len(model_peer.requests) == 1, model_peer.requests
            process.send_signal(signal.SIGINT)
            assert process.wait(timeout=20) == 0
            assert not (home / 'gateway.pid').exists()
            assert not (home / 'gateway.sock').exists()
            print(json.dumps({k: descriptor[k] for k in ('instance_id', 'authority_epoch', 'state', 'capabilities')}))
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


@pytest.mark.asyncio
async def test_api_bind_failure_unwinds_control_db_and_reservation(tmp_path, monkeypatch):
    from functools import partial
    from gateway import run_api
    from gateway.run import start_gateway
    from gateway.config import GatewayConfig
    from gateway.runtime_ownership import process_ownership
    from hermes_cli import web_server as web
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    with socket.socket() as blocker:
        blocker.bind(('127.0.0.1', 0))
        blocker.listen()
        monkeypatch.setattr(run_api, 'start_gateway_api', partial(
            run_api.start_gateway_api, port=blocker.getsockname()[1]))
        with pytest.raises(OSError):
            await start_gateway(GatewayConfig(), verbosity=None)
        assert getattr(web.app.state, 'gateway_runner', None) is None
        assert not (home / 'gateway.pid').exists()
        assert not (home / 'gateway.sock').exists()
        assert not process_ownership.owns(home)
