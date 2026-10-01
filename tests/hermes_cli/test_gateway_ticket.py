"""Private ticket CLI crosses a real daemon control socket and HTTP/WS ingress."""
import asyncio
import json
from pathlib import Path
import subprocess
import sys

import aiohttp
from websockets.asyncio.client import connect

from tests.gateway.fixtures.local_recovery_probe import child_env, daemon, rpc


def test_private_ticket_cli_pins_owner_and_scope_across_native_transports(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    (home / 'config.yaml').write_text(json.dumps({'model': {'provider': 'custom', 'default': 'fixture',
        'base_url': 'http://127.0.0.1:9/v1'}, 'auxiliary': {'title_generation': {'enabled': False}}}))
    secondary = home / 'profiles' / 'work'
    secondary.mkdir(parents=True)
    (secondary / 'config.yaml').write_text((home / 'config.yaml').read_text().replace('fixture', 'work-model'))
    env = child_env() | dict(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
        PYTHONPATH=str(root), OPENAI_API_KEY='loopback-only', PYTHONUNBUFFERED='1')

    def mint(desc, kind, **changes):
        request = {'profile_id': str(home), 'instance_id': desc['instance_id'], 'purpose': kind} | changes
        result = subprocess.run([sys.executable, '-m', 'hermes_cli.main', 'gateway', 'ticket'],
            cwd=root, env=env, input=json.dumps(request), text=True, capture_output=True, timeout=30)
        assert 'ticket' not in result.stderr, result.stderr
        return result

    async def exercise(desc):
        ticket = json.loads(mint(desc, 'interactive').stdout)['ticket']
        async with connect(desc['api_origin'].replace('http:', 'ws:') + '/api/ws',
            subprotocols=['hermes-gateway-v1', 'hermes-gateway-ticket.' + ticket], proxy=None) as ws:
            result = await rpc(ws, 'groups.capabilities')
            assert result['result']['driver'] is True, result
            assert 'groups.discard' in result['result']['methods']
        grant = json.loads(mint(desc, 'native-http').stdout)['ticket']
        async with aiohttp.ClientSession() as client:
            async with client.get(desc['api_origin'] + '/api/config',
                    headers={'X-Hermes-Gateway-Ticket': grant}) as response:
                assert response.status == 200, await response.text()
            async with client.get(desc['api_origin'] + '/api/config',
                    headers={'X-Hermes-Gateway-Ticket': grant}) as response:
                assert response.status == 401
        routed = json.loads(mint(desc, 'native-http', profile='work').stdout)
        assert routed['profile_id'] == str(secondary)
        async with aiohttp.ClientSession() as client:
            async with client.get(desc['api_origin'] + '/api/config?profile=work',
                    headers={'X-Hermes-Gateway-Ticket': routed['ticket']}) as response:
                assert response.status == 200, await response.text()
                assert 'work-model' in await response.text()
        for changes in ({'instance_id': 'stale'}, {'profile_id': str(user)}, {'purpose': 'exposure'}):
            rejected = mint(desc, 'interactive', **changes)
            assert rejected.returncode == 4
            assert json.loads(rejected.stdout) == {'error': 'native_ticket_unavailable'}

    with daemon(root, home, env, barrier=False) as (owner, desc):
        asyncio.run(exercise(desc))
        assert owner.poll() is None
