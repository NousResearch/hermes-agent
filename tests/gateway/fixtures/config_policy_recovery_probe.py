"""Unmodified daemon/config-source restart over actual loopback inference."""
import asyncio
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import sqlite3
import threading

from tests.gateway.fixtures.cli_launch_policy_probe import Model
from tests.gateway.fixtures.local_recovery_probe import daemon, websocket, rpc


def probe(tmp_path):
    root = Path(__file__).resolve().parents[3]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests, peer.lock, peer.barrier = [], threading.Lock(), threading.Barrier(2)
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    origin = f'http://127.0.0.1:{peer.server_port}'
    keys = {side: 'config-private-' + side for side in ('left', 'right')}
    cfg = {'model': {'provider': 'custom', 'default': 'fixture', 'base_url': origin + '/default/v1',
                     'api_key': 'unrelated-default-private-key'},
           'providers': {s: {'base_url': origin + '/' + s + '/v1', 'api_key': k} for s, k in keys.items()},
           'auxiliary': {'title_generation': {'enabled': False}}, 'platform_toolsets': {'cli': []}}
    config = home / 'config.yaml'
    config.write_text(json.dumps(cfg))
    before = config.read_bytes()
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'SYSTEMROOT') if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root))
    sessions, pids, snapshots = {}, [], {}
    params = {s: dict(request_id=s, cwd=str(home), model='gpt-5-' + s, provider='custom:' + s,
                     toolsets=[]) for s in keys}

    def policies():
        with sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True) as db:
            return dict(db.execute("SELECT key,value FROM state_meta WHERE key LIKE 'gateway.local_policy.v1:%'"))

    async def turn(desc, side, phase):
        async with websocket(home, desc) as ws:
            created = await rpc(ws, 'session.create', **params[side])
            assert 'result' in created, created
            sid = created['result']['session_id']
            if side in sessions:
                assert sid == sessions[side]
            sessions[side] = sid
            resumed = await rpc(ws, 'session.resume', session_id=sid)
            epoch = resumed['result']['replay_epoch']
            accepted = await rpc(ws, 'prompt.submit', session_id=sid, input_id=phase + side, text=phase + side)
            assert 'result' in accepted, accepted
            async with asyncio.timeout(30):
                while True:
                    receipt = await rpc(ws, 'prompt.receipt', session_id=sid,
                                        admission_id=accepted['result']['admission_id'])
                    if receipt['result']['status'] == 'terminal':
                        break
                    await asyncio.sleep(.03)
            events = await rpc(ws, 'session.events.since', session_id=sid, last_sequence=0, replay_epoch=epoch)
            if phase == 'denied':
                # This baseline's pre-agent display failure has generic public
                # delivery; the worker lane owns settlement/error projection.
                failure = json.dumps(events) + (home / 'restart.log').read_text()
                assert 'launch_credentials_unavailable' in failure, failure
            else:
                assert 'LAUNCH_POLICY_OK' in json.dumps(events), events
            assert not any(k in json.dumps([created, resumed, events]) for k in keys.values())

    async def pair(desc, phase):
        await asyncio.gather(*(turn(desc, s, phase) for s in keys))
    try:
        with daemon(root, home, env, barrier=False) as (proc, desc):
            pids.append(proc.pid)
            asyncio.run(pair(desc, 'first'))
            snapshots = policies()
            assert config.read_bytes() == before
        # Current routing changes must not retarget either frozen key.
        for entry in cfg['providers'].values():
            entry['base_url'] = origin + '/wrong/v1'
        config.write_text(json.dumps(cfg))
        with daemon(root, home, env, barrier=False) as (proc, desc):
            pids.append(proc.pid)
            asyncio.run(pair(desc, 'restart'))
            assert policies() == snapshots
        assert len(peer.requests) == 4, peer.requests
        for side, key in keys.items():
            requests = [r for r in peer.requests if r['body']['model'] == 'gpt-5-' + side]
            assert len(requests) == 2
            assert all(r['auth'] == 'Bearer ' + key and r['path'] == '/' + side + '/v1/chat/completions' for r in requests), requests
        cfg['providers']['left']['api_key'] = 'changed-wrong-credential'
        config.write_text(json.dumps(cfg))
        with daemon(root, home, env, barrier=False) as (proc, desc):
            pids.append(proc.pid)
            asyncio.run(pair(desc, 'denied'))
            assert policies() == snapshots
        assert len(peer.requests) == 4
        # backups/config/ holds byte-exact copies of config.yaml itself (last-known-good recovery,
        # #109463); the invariant here is that the DAEMON never persists credentials elsewhere.
        backups = home / 'backups' / 'config'
        leaks = [str(p.relative_to(home)) for p in home.rglob('*') if p.is_file() and p != config
                 and backups not in p.parents
                 and any(k.encode() in p.read_bytes() for k in keys.values())]
        assert not leaks, leaks
        return {'pids': pids, 'requests': 4, 'concurrent_inline_keys': True, 'restart_actual_inference': True,
                'frozen_endpoint_key': True, 'creation_retry_identity': True, 'changed_key_refused': True,
                'durable_policy_redacted': True, 'original_config_unchanged_by_daemon': True}
    finally:
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)
