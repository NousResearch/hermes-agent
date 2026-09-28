"""Ordinary startup recovers native admissions without a recovery fixture call."""
import asyncio
from contextlib import closing
from http.server import ThreadingHTTPServer
import hashlib
import hmac
import json
from pathlib import Path
import socket
import sqlite3
import threading

import aiohttp

from tests.gateway.fixtures.local_recovery_probe import Model, child_env, daemon


def test_native_queue_recovers_only_after_current_adapter_publication(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests = []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{peer.server_port}/v1'
    cfg = {'model': {'provider': 'custom', 'default': 'local-wire-stub', 'base_url': base},
           'gateway': {'multiplex_profiles': False},
           'platforms': {'webhook': {'enabled': True, 'extra': {'host': '127.0.0.1', 'port': port,
               'secret': 'owned-secret', 'routes': {'fixture': {'prompt': '{text}', 'deliver': 'log'}}}}},
           'auxiliary': {'title_generation': {'enabled': False}}, 'terminal': {'cwd': str(home)}}
    config = home / 'config.yaml'
    config.write_text(json.dumps(cfg))
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=base, PYTHONUNBUFFERED='1')
    def rows():
        with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
            return dict(db.execute('SELECT request_id,status FROM session_admissions'))
    async def submit():
        body = json.dumps({'text': 'SAFE_QUEUE'}).encode()
        headers = {'X-GitHub-Delivery': 'safe', 'X-GitHub-Event': 'push',
                   'X-Hub-Signature-256': 'sha256=' + hmac.new(b'owned-secret', body, hashlib.sha256).hexdigest()}
        async with aiohttp.ClientSession() as client:
            async with client.post(f'http://127.0.0.1:{port}/webhooks/fixture', data=body, headers=headers) as response:
                assert response.status == 202, await response.text()
        assert rows() == {'safe': 'queued'}
    async def settled():
        async with asyncio.timeout(25):
            while rows()['safe'] != 'terminal':
                await asyncio.sleep(.05)
    evidence = []
    try:
        with daemon(root, home, env, barrier=True, fixture='api_recovery_daemon.py') as (proc, desc):
            asyncio.run(submit())
            evidence.append({'pid': proc.pid, 'epoch': desc['authority_epoch'], 'rows': rows()})
            proc.kill()
            proc.wait(timeout=10)
        # No current transport: keep the durable row, do not spend its claim.
        cfg['platforms']['webhook']['enabled'] = False
        config.write_text(json.dumps(cfg))
        with daemon(root, home, env, barrier=False) as (proc, desc):
            assert rows() == {'safe': 'queued'} and not peer.requests
            evidence.append({'pid': proc.pid, 'epoch': desc['authority_epoch'], 'rows': rows()})
        cfg['platforms']['webhook']['enabled'] = True
        config.write_text(json.dumps(cfg))
        # Changed connector authority and an absent configured route both fail closed.
        extra = cfg['platforms']['webhook']['extra']
        for secret, routes in [('changed-secret', {'fixture': {'prompt': '{text}', 'deliver': 'log'}}),
                               ('owned-secret', {})]:
            extra['secret'], extra['routes'] = secret, routes
            config.write_text(json.dumps(cfg))
            with daemon(root, home, env, barrier=False) as (proc, desc):
                assert rows() == {'safe': 'queued'} and not peer.requests
                evidence.append({'pid': proc.pid, 'epoch': desc['authority_epoch'], 'rows': rows()})
        extra['routes'] = {'fixture': {'prompt': '{text}', 'deliver': 'log'}}
        config.write_text(json.dumps(cfg))
        for _ in range(2):
            with daemon(root, home, env, barrier=False) as (proc, desc):
                asyncio.run(settled())
                assert peer.requests, '\n'.join(p.read_text() for p in (home / 'logs').glob('*.log')) + (home / 'restart.log').read_text()
                evidence.append({'pid': proc.pid, 'epoch': desc['authority_epoch'], 'rows': rows()})
        assert len(peer.requests) == 1, peer.requests
        assert 'SAFE_QUEUE' in json.dumps(peer.requests)
        print(json.dumps({'boots': evidence, 'inferences': len(peer.requests)}))
    finally:
        peer.release.set()
        peer.shutdown()
        peer.server_close()
