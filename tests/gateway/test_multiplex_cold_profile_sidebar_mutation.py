"""Cold-profile sidebar mutations reach the served profile's own authority under multiplex.

Acceptance case for the Desktop local-backend-pool cutover (PR #106742): archiving a
session that belongs to a profile with no resident client (COLD) used to fail on main with
``LocalBackendSlotWaitTimeoutError`` because the Desktop's archive path reached per-profile
backend startup with background priority. With one multiplexing gateway there is no
per-profile process to start: ``PATCH /api/sessions/{id}`` carrying the owning profile in
routing metadata (``?profile=`` and ``body.profile``, exactly what
``apps/desktop/src/api/sessions.ts::mutateSessionHttp`` sends) lands on that profile's
authority, while another served profile holds a live interactive attachment.

The Desktop's endpoint for a served secondary comes from ``hermes -p <name> gateway ensure``
and its ticket is minted through the endpoint's ``control_home`` (the multiplexer root),
so this test uses that exact discovery + mint path rather than a hand-built socket call.
"""
import asyncio
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time
from types import SimpleNamespace

import httpx
import pytest
from websockets.asyncio.client import connect

from tests.gateway.test_normal_runtime_boot import control


@contextmanager
def multiplex_daemon(tmp_path, profiles=('warm', 'cold')):
    """One ``gateway.run`` serving default + *profiles*; yields (root, descriptor, env)."""
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    for name in profiles:
        secondary = home / 'profiles' / name
        secondary.mkdir(parents=True)
        (secondary / 'config.yaml').write_text(json.dumps({
            'model': {'provider': 'custom', 'default': name + '-model', 'base_url': 'http://127.0.0.1:1/v1'}}))
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': True},
        'model': {'provider': 'custom', 'default': 'launch-model', 'base_url': 'http://127.0.0.1:1/v1'},
        'auxiliary': {'title_generation': {'enabled': False}},
    }))
    root = Path(__file__).resolve().parents[2]
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               PYTHONUNBUFFERED='1')
    with (tmp_path / 'gateway.log').open('w+') as log:
        process = subprocess.Popen([sys.executable, '-m', 'gateway.run'], cwd=root, env=env,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 60
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
            yield home, descriptor, env
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def desktop_ensure(env, root, profile):
    """The exact child the Desktop spawns: ``hermes --profile <p> gateway ensure --json``."""
    result = subprocess.run([sys.executable, '-m', 'hermes_cli.main', '--profile', profile,
                             'gateway', 'ensure', '--json', '--timeout', '30'],
                            cwd=root, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)
    payload = json.loads(result.stdout)
    assert result.returncode == 0 and payload['state'] == 'ready', (result.returncode, payload, result.stderr[-500:])
    return payload['endpoint']


def native_headers(endpoint):
    from gateway.client import _session_ticket
    ticket = _session_ticket(Path(endpoint['profile_id']), SimpleNamespace(**endpoint), purpose='native-http')
    return {'X-Hermes-Gateway-Ticket': ticket}


async def create_via_ws(endpoint, request_id, hold=None):
    """Create a GUI session on *endpoint* over the canonical WS; ``hold`` keeps the attachment
    open while awaiting it (the "other profile is active" arm)."""
    from gateway.client import _session_ticket
    ticket = _session_ticket(Path(endpoint['profile_id']), SimpleNamespace(**endpoint), purpose='interactive')
    async with connect(endpoint['api_origin'].replace('http:', 'ws:') + '/api/ws',
                       subprotocols=['hermes-gateway-v1', 'hermes-gateway-ticket.' + ticket]) as ws:
        await ws.send(json.dumps({'id': 1, 'method': 'session.create',
                                  'params': {'request_id': request_id, 'source': 'gui'}}))
        while True:
            response = json.loads(await ws.recv())
            if response.get('id') == 1:
                assert 'result' in response, response
                sid = response['result']['session_id']
                break
        if hold is not None:
            await hold
        return sid


def desktop_sidebar_patch(client, endpoint, profile, sid, payload, request_id):
    """Byte-for-byte what ``mutateSessionHttp`` does: snapshot, then PATCH with identity + profile
    in both the query and the body."""
    snapshot = client.get(f'/api/sessions/{sid}/mutation-snapshot', params={'profile': profile},
                          headers=native_headers(endpoint))
    assert snapshot.status_code == 200, snapshot.text
    identity = {'request_id': request_id, 'expected_revision': snapshot.json()['runtime_revision'],
                'expected_generation': snapshot.json()['runtime_generation']}
    return client.patch(f'/api/sessions/{sid}', params={'profile': profile},
                        json={**payload, **identity, 'profile': profile}, headers=native_headers(endpoint))


@pytest.mark.linux_only
def test_cold_profile_archive_unarchive_while_another_profile_is_attached(tmp_path):
    with multiplex_daemon(tmp_path) as (home, descriptor, env), \
            httpx.Client(base_url=descriptor['api_origin'], trust_env=False, timeout=30) as client:
        root = Path(__file__).resolve().parents[2]
        served = {p['profile_id'] for p in descriptor['served_profiles']}
        cold_home, warm_home = (home / 'profiles' / 'cold').resolve(), (home / 'profiles' / 'warm').resolve()
        assert {str(home.resolve()), str(cold_home), str(warm_home)} <= served, descriptor

        # Desktop discovery for every profile resolves to the SAME instance; secondaries mint via the root.
        endpoints = {name: desktop_ensure(env, root, name) for name in ('default', 'warm', 'cold')}
        assert {e['instance_id'] for e in endpoints.values()} == {descriptor['instance_id']}
        assert endpoints['cold']['profile_id'] == str(cold_home)
        assert endpoints['cold']['control_home'] == str(home.resolve())
        assert endpoints['default']['control_home'] is None

        # The cold profile's session was created earlier through the gateway and its client left.
        cold_sid = asyncio.run(create_via_ws(endpoints['cold'], 'cold-create'))

        # Warm stays attached (live interactive WS) for the whole mutation sequence.
        outcome = {}

        async def with_warm_attached():
            loop = asyncio.get_running_loop()

            def mutate():
                archived = desktop_sidebar_patch(client, endpoints['cold'], 'cold', cold_sid, {'archived': True}, 'archive')
                assert archived.status_code == 200, archived.text
                row = client.get(f'/api/sessions/{cold_sid}', params={'profile': 'cold'},
                                 headers=native_headers(endpoints['cold'])).json()
                assert (row['archived'], row['profile']) == (1, 'cold'), row
                restored = desktop_sidebar_patch(client, endpoints['cold'], 'cold', cold_sid, {'archived': False}, 'unarchive')
                assert restored.status_code == 200, restored.text
                row = client.get(f'/api/sessions/{cold_sid}', params={'profile': 'cold'},
                                 headers=native_headers(endpoints['cold'])).json()
                assert (row['archived'], row['profile']) == (0, 'cold'), row
                outcome.update(archive=archived.json(), unarchive=restored.json())

            await create_via_ws(endpoints['warm'], 'warm-create', hold=loop.run_in_executor(None, mutate))

        asyncio.run(with_warm_attached())
        assert outcome['archive']['archived'] is True and outcome['unarchive']['archived'] is False
        assert outcome['unarchive']['revision'] == outcome['archive']['revision'] + 1

        # The write landed in the COLD profile's own store; the launch profile's store is untouched.
        with sqlite3.connect(cold_home / 'state.db') as db:
            assert db.execute('SELECT archived FROM sessions WHERE id=?', (cold_sid,)).fetchone() == (0,)
        with sqlite3.connect(home / 'state.db') as db:
            assert db.execute('SELECT COUNT(*) FROM sessions WHERE id=?', (cold_sid,)).fetchone() == (0,)

        # The default profile's ticket cannot select the cold profile's rows (profile boundary holds).
        foreign = client.get(f'/api/sessions/{cold_sid}/mutation-snapshot', params={'profile': 'cold'},
                             headers=native_headers(endpoints['default']))
        assert foreign.status_code == 403, foreign.text
