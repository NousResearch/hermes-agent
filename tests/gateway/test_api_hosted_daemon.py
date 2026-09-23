"""Real invitation/grant HTTP inference and hidden room restoration."""
import asyncio
import hashlib
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import sqlite3
import threading

import aiohttp

from tests.gateway.fixtures.local_recovery_probe import daemon, Model


def test_hosted_grant_inference_survives_owner_restart(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700); user.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests = []
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{peer.server_port}/v1'
    cfg = {'model': {'provider': 'custom', 'default': 'local-wire-stub', 'base_url': base},
           'gateway': {'platforms': {'api_server': {'enabled': True, 'port': port, 'host': '127.0.0.1'}}, 'multiplex_profiles': False},
           'auxiliary': {'title_generation': {'enabled': False}}, 'terminal': {'cwd': str(home)},
           'platform_toolsets': {'api_server': []}, 'approvals': {'mode': 'manual'}}
    (home / 'config.yaml').write_text(json.dumps(cfg))
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=base, API_SERVER_KEY='ordinary-daemon-owned-secret',
               API_SERVER_ENABLED='true', API_SERVER_PORT=str(port), PYTHONUNBUFFERED='1')
    url = f'http://127.0.0.1:{port}/v1'
    invitation, run_ids, sessions, epochs = {}, {}, [], []
    identity = {'room_id': 'room-owned', 'home_install_id': 'owned-home', 'authority_gateway_id': 'owned-authority',
                'authority_epoch': 1, 'member_id': 'member-owned'}

    async def exercise(restart):
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as client:
            if not restart:
                async with client.post(url + '/room-members/invitations', json=identity,
                    headers={'Authorization': 'Bearer ordinary-daemon-owned-secret'}) as response:
                    invitation.update(await response.json())
                    assert response.status == 201, invitation
            headers = {'Authorization': 'HermesRoom ' + invitation['grant']}
            catalog = invitation['catalog']
            async with client.get(url + '/room-members/capabilities', headers=headers) as response:
                capabilities = await response.json()
                assert response.status == 200 and capabilities['catalog'] == catalog, capabilities
            for task in ([1, 2] if restart else [1]):
                prompt = 'HOSTED_TURN_' + str(task)
                dispatch = {**identity, 'protocol_version': 2, 'target_install_id': catalog['installation_id'],
                    'target_profile': 'default', 'task_id': 'task-' + str(task), 'execution_generation': 1,
                    'source_event_seq': task, 'cancellation_scope_id': 'cancel-owned', 'prompt': prompt,
                    'prompt_digest': hashlib.sha256(prompt.encode()).hexdigest(),
                    'capability_digest': catalog['catalog_digest'],
                    'execution_policy_digest': catalog['execution_policy']['policy_digest'], 'trace_id': 'trace-owned'}
                before = len(peer.requests)
                async with client.post(url + '/runs', json={'input': prompt, 'hosted_room_dispatch': dispatch},
                    headers={**headers, 'Idempotency-Key': 'room:task-' + str(task) + ':1'}) as response:
                    accepted = await response.json()
                    assert response.status == 202, accepted
                if task in run_ids:
                    assert accepted['run_id'] == run_ids[task] and accepted['replayed'], accepted
                    assert len(peer.requests) == before
                run_ids[task] = accepted['run_id']
                async with asyncio.timeout(20):
                    while True:
                        async with client.get(url + '/runs/' + run_ids[task], headers=headers) as response:
                            result = await response.json()
                            assert response.status == 200, result
                        if result['status'] in {'completed', 'failed'}:
                            break
                        await asyncio.sleep(.05)
                assert result['status'] == 'completed' and prompt in result['output'], result
                sessions.append(result['session_id'])
            async with client.get(url + '/runs/' + run_ids[1], headers={'Authorization': 'HermesRoom invalid'}) as response:
                assert response.status == 401
    try:
        for restart in (False, True):
            with daemon(root, home, env, barrier=False) as (proc, desc):
                epochs.append(desc['authority_epoch'])
                asyncio.run(exercise(restart))
                with sqlite3.connect(home / 'state.db') as db:
                    assert db.execute('SELECT source,hidden FROM sessions WHERE id=?', (sessions[-1],)).fetchone() == ('bot_room', 1)
                proc.kill(); proc.wait(timeout=10)
        assert len(set(sessions)) == 1
        with sqlite3.connect(home / 'state.db') as db:
            row = db.execute('SELECT source,hidden,title FROM sessions WHERE id=?', (sessions[0],)).fetchone()
            assert row == ('bot_room', 1, 'Group: room-owned'), row
        assert len(peer.requests) == 2, peer.requests
        assert 'HOSTED_TURN_1' in json.dumps(peer.requests[-1]['messages'])
        print(json.dumps({'ordinary_daemon': True, 'epochs': epochs, 'session_ids': sessions,
                          'authenticated_grant': True, 'inference_count': len(peer.requests)}))
    finally:
        peer.shutdown(); peer.server_close()
