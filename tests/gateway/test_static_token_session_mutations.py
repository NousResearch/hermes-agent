"""The same verified static token owns the same session on HTTP and WebSocket."""
import asyncio
import json
from pathlib import Path

import aiohttp
from websockets.asyncio.client import connect

from tests.gateway.fixtures.local_recovery_probe import child_env, daemon, rpc, websocket


def test_static_token_http_mutations_reuse_ws_owner(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False},
        'model': {'provider': 'custom', 'default': 'no-inference'},
        'platform_toolsets': {'cli': []},
        'auxiliary': {'title_generation': {'enabled': False}},
    }))
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
               PYTHONPATH=str(root), HERMES_DASHBOARD_SESSION_TOKEN='owned-static-token')
    params = dict(request_id='same-create', source='cli', cwd=str(home),
                  model='no-inference', toolsets=[])
    saved = {}

    async def exercise(desc):
        origin = desc['api_origin']
        async with connect(origin.replace('http:', 'ws:') + '/api/ws?token=owned-static-token') as ws:
            created = await rpc(ws, 'session.create', **params)
            assert 'result' in created, created
            sid = created['result']['session_id']
            body = {'request_id': 'same-edit', 'expected_revision': 0, 'title': 'Static token title'}
            async with aiohttp.ClientSession() as http:
                url = origin + '/api/sessions/' + sid
                async with http.patch(url, json=body, headers={'Authorization': 'Bearer wrong-token'}) as response:
                    assert response.status in (401, 403)
                async with http.patch(url, json=body, headers={'Authorization': 'Bearer owned-static-token'}) as response:
                    result = await response.json()
                    assert response.status == 200, result
                retry = await rpc(ws, 'session.mutate', session_id=sid, request_id='same-edit',
                                  expected_revision=0, operation='sidebar', payload={'title': body['title']})
                assert 'result' in retry, retry
                assert retry['result'] == {key: value for key, value in result.items() if key != 'ok'}
                if saved:
                    assert saved == {'sid': sid, 'receipt': result}
                else:
                    saved.update(sid=sid, receipt=result)
            async with websocket(home, desc) as native:
                resumed = await rpc(native, 'session.resume', session_id=sid)
                assert resumed.get('result', {}).get('session_id') == sid, resumed

    for _ in range(2):
        with daemon(root, home, env, barrier=False) as (_, desc):
            asyncio.run(exercise(desc))
