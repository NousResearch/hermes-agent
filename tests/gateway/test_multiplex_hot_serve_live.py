"""Live hot-serve contract of a real ``python -m gateway.run`` multiplexer (Linux; loopback model).

A profile created under the running multiplexer is served by it without a restart: ``identify``
lists the new home, a ticket minted for it opens a session owned by this process, and ``profiles/``
(not the boot-time reservation) is what the next restart reserves — so ``hermes -p new`` finds an
owner instead of spawning a stray per-profile daemon that would later collide with the
multiplexer's all-or-nothing reserve. A secondary whose ``state.db`` is unusable parks only itself.
"""
import asyncio
import contextlib
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

ROOT = Path(__file__).resolve().parents[2]
# The daemon is a foreground child of the test (never detached), boots a disposable custom root
# with HOME redirected (no systemd unit, no webhook port) and is reaped in every path below.
pytestmark = [pytest.mark.linux_only, pytest.mark.spawns_gateway_lookalike]


def control(home, verb, params=None, timeout=10):
    request = json.dumps({'protocol': 1, 'id': 1, 'verb': verb, 'params': params or {}}).encode() + b'\n'
    from gateway.control_socket import resolve_client_socket_path
    path = resolve_client_socket_path(home)
    if path is None:
        raise FileNotFoundError('control socket absent')
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
        peer.settimeout(timeout)
        peer.connect(str(path))
        peer.sendall(request)
        with peer.makefile('rb') as stream:
            raw = stream.readline()
    reply = json.loads(raw.split(b'\n', 1)[0])
    assert reply.get('ok'), reply
    return reply['result']


@pytest.fixture
def peer():
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


@pytest.fixture
def mux(tmp_path, peer):
    """Disposable custom root (``<root>/profiles/<name>`` layout, as in Docker) with
    ``multiplex_profiles`` on, plus one boot-time secondary."""
    user = tmp_path / 'user'
    user.mkdir()
    root = tmp_path / 'state'
    root.mkdir(mode=0o700)
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    config = {'gateway': {'multiplex_profiles': True},
              'model': {'provider': 'custom', 'default': 'local-wire-stub', 'base_url': url},
              'auxiliary': {'title_generation': {'enabled': False}}}
    (root / 'config.yaml').write_text(json.dumps(config))
    boot = root / 'profiles' / 'boot'
    boot.mkdir(parents=True, mode=0o700)
    (boot / 'config.yaml').write_text(json.dumps(config))
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ') if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(root), PYTHONPATH=str(ROOT),
               PYTHONUNBUFFERED='1', OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url)
    return {'root': root, 'boot': boot, 'env': env, 'config': config, 'log': tmp_path / 'gateway.log'}


def start_daemon(mux, timeout=90):
    """Boot the multiplexer the way ``hermes gateway run`` does; the canonical gateway command-line
    matcher (``gateway.status.looks_like_gateway_command_line``) is what the CLI's liveness probes
    apply to the recorded PID, so the fixture must spawn a recognisable launcher."""
    log = mux['log'].open('a+', encoding='utf-8', errors='replace')
    proc = subprocess.Popen([sys.executable, '-m', 'hermes_cli.main', 'gateway', 'run', '--quiet'], cwd=ROOT,
                            env=mux['env'], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    desc = {}
    deadline = time.monotonic() + timeout
    while proc.poll() is None and time.monotonic() < deadline:
        try:
            desc = control(mux['root'], 'identify', timeout=2)
            if desc.get('state') == 'ready':
                break
        except (OSError, ValueError, AssertionError):
            pass
        time.sleep(.1)
    log.flush()
    assert desc.get('state') == 'ready', (desc, proc.poll(), tail(mux))
    return proc, desc


def tail(mux, n=6000):
    """stdout capture plus the daemon's own ``logs/gateway.log`` (``--quiet`` sends records there)."""
    texts = [p.read_text(encoding='utf-8', errors='replace') for p in (mux['log'], mux['root'] / 'logs' / 'gateway.log')
             if p.exists()]
    return '\n'.join(t[-n:] for t in texts)


def stop(proc, expect=0):
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            code = proc.wait(timeout=45)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
            raise
        if expect is not None:
            assert code == expect, code


def served_homes(desc):
    return {Path(entry['home']).resolve() for entry in desc.get('served_profiles', [])}


def recorded_served(root):
    return json.loads((root / 'gateway_state.json').read_text(encoding='utf-8')).get('served_profiles')


async def open_session_for(root, home, desc):
    """A ticket bound to *home* opens a session the multiplexer commits into *home*'s state.db."""
    grant = control(root, 'session-ticket', {'profile_id': str(home.resolve()),
                                             'instance_id': desc['instance_id'], 'purpose': 'interactive'})
    url = desc['api_origin'].replace('http:', 'ws:') + '/api/ws'
    async with connect(url, subprotocols=['hermes-gateway-v1', 'hermes-gateway-ticket.' + grant['ticket']]) as ws:
        await ws.send(json.dumps({'jsonrpc': '2.0', 'id': 1, 'method': 'session.create',
                                  'params': {'request_id': 'hot-served', 'source': 'cli'}}))
        async with asyncio.timeout(20):
            while True:
                reply = json.loads(await ws.recv())
                if reply.get('id') == 1:
                    assert 'result' in reply, reply
                    return reply['result']['session_id']


def test_profile_created_under_live_multiplexer_is_served_and_owned(mux):
    root, env = mux['root'], mux['env']
    proc, desc = start_daemon(mux)
    try:
        assert served_homes(desc) == {root.resolve(), mux['boot'].resolve()}
        # ``hermes profile create`` under the multiplexer's root: the CLI signals the control socket.
        created = subprocess.run([sys.executable, '-m', 'hermes_cli.main', 'profile', 'create', 'fresh'],
                                 cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
        assert created.returncode == 0, (created.stdout, created.stderr)
        fresh = root / 'profiles' / 'fresh'
        assert fresh.is_dir()
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and fresh.resolve() not in served_homes(desc):
            time.sleep(.2)
            desc = control(root, 'identify')
        assert fresh.resolve() in served_homes(desc), (desc, tail(mux))
        assert recorded_served(root) == ['default', 'boot', 'fresh']
        # The new profile's sessions are this multiplexer's: same instance, its own state.db.
        sid = asyncio.run(open_session_for(root, fresh, desc))
        import sqlite3
        with sqlite3.connect(f"file:{fresh / 'state.db'}?mode=ro", uri=True) as db:
            assert db.execute('SELECT instance_id FROM runtime_epoch').fetchone() == (desc['instance_id'],)
            assert db.execute('SELECT 1 FROM sessions WHERE id=?', (sid,)).fetchone()
        # ``hermes -p fresh`` resolves to the multiplexer instead of spawning a daemon of its own.
        ensured = subprocess.run([sys.executable, '-m', 'hermes_cli.main', '-p', 'fresh', 'gateway', 'ensure',
                                  '--timeout', '15'], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
        payload = json.loads(ensured.stdout.strip().splitlines()[-1])
        assert payload['state'] == 'ready' and payload['endpoint']['instance_id'] == desc['instance_id'], (payload, ensured.stderr)
        assert not (fresh / 'gateway.pid').exists()
        stop(proc)
        # The restart reserves what exists now, the hot-served profile included: no OwnershipConflict.
        proc, desc = start_daemon(mux)
        assert served_homes(desc) == {root.resolve(), mux['boot'].resolve(), fresh.resolve()}
        assert 'Cannot reserve gateway profiles' not in tail(mux)
    finally:
        stop(proc, expect=None)


def kill_recorded_gateway(root, timeout=45):
    """SIGINT the gateway whose PID ``gateway_state.json`` records (a detached spawn is not our child)."""
    pid = json.loads((root / 'gateway_state.json').read_text(encoding='utf-8')).get('pid')
    if pid:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGINT)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(.2)
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)


@pytest.mark.live_system_guard_bypass
def test_named_profile_ensure_during_multiplexer_downtime_starts_the_multiplexer(mux):
    """The multiplexer is down (restart/update) and ``hermes -p boot gateway ensure`` runs: the client
    must start the MULTIPLEXER, never a per-profile daemon (which would answer this client and then
    block the multiplexer's next all-or-nothing reserve with ``Cannot reserve gateway profiles``)."""
    root, env, boot = mux['root'], mux['env'], mux['boot']
    proc, desc = start_daemon(mux)
    stop(proc)
    assert recorded_served(root) == ['default', 'boot']
    try:
        ensured = subprocess.run([sys.executable, '-m', 'hermes_cli.main', '-p', 'boot', 'gateway', 'ensure',
                                  '--json', '--timeout', '90'], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
        payload = json.loads(ensured.stdout.strip().splitlines()[-1])
        assert payload['state'] == 'ready', (payload, ensured.stderr)
        # The answering owner is the multiplexer at the root, serving boot as a secondary.
        assert Path(payload['endpoint']['control_home']).resolve() == root.resolve(), payload
        assert served_homes(control(root, 'identify')) == {root.resolve(), boot.resolve()}
        assert not (boot / 'gateway.pid').exists() and not (boot / 'gateway.sock').exists()
    finally:
        kill_recorded_gateway(root)
    # A fresh multiplexer boot reserves every profile: nothing stray owns boot's home.
    proc, desc = start_daemon(mux)
    try:
        assert served_homes(desc) == {root.resolve(), boot.resolve()}
        assert 'Cannot reserve gateway profiles' not in tail(mux)
        stop(proc)
    finally:
        stop(proc, expect=None)


def test_broken_secondary_state_db_parks_only_that_profile(mux):
    root = mux['root']
    broken = root / 'profiles' / 'broken'
    broken.mkdir(parents=True, mode=0o700)
    (broken / 'config.yaml').write_text(json.dumps(mux['config']))
    # Not a SQLite file and not the zero-byte case the quarantine repairs: an unopenable store.
    (broken / 'state.db').mkdir()
    proc, desc = start_daemon(mux)
    try:
        assert served_homes(desc) == {root.resolve(), mux['boot'].resolve()}, desc
        assert recorded_served(root) == ['default', 'boot']
        assert "Profile 'broken' not served" in tail(mux), tail(mux)
        # The park is published (identify + gateway_state.json), so the profile's clients fail fast
        # with the real reason instead of waiting the whole deadline for a service that never comes.
        assert 'broken' in desc.get('parked_profiles', {}), desc
        recorded = json.loads((root / 'gateway_state.json').read_text(encoding='utf-8'))
        assert 'broken' in recorded.get('parked_profiles', {}), recorded
        started = time.monotonic()
        ensured = subprocess.run([sys.executable, '-m', 'hermes_cli.main', '-p', 'broken', 'gateway', 'ensure',
                                  '--json', '--timeout', '30'], cwd=ROOT, env=mux['env'], capture_output=True,
                                 text=True, timeout=60)
        payload = json.loads(ensured.stdout.strip().splitlines()[-1])
        assert (payload['state'], payload['reason_code']) == ('inaccessible', 'profile_parked'), (payload, ensured.stderr)
        assert 'unusable' in (payload.get('detail') or ''), payload
        assert ensured.returncode == 7 and time.monotonic() - started < 20, (ensured.returncode, ensured.stderr)
        stop(proc)
    finally:
        stop(proc, expect=None)
