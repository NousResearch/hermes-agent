"""#139835 (LC-13): an API session written by an earlier release stored the client's declared
``X-Hermes-Session-Key`` in ``sessions.session_key``; the caller that declares that same key is
adopted onto the canonical route, never 409'd forever. A different key still conflicts."""
import json
import os
import subprocess
import sys
from pathlib import Path


def test_runs_adopt_legacy_declared_session_key(tmp_path):
    result = subprocess.run([sys.executable, __file__], cwd=Path(__file__).resolve().parents[2],
        env=dict(os.environ, HERMES_HOME=str(tmp_path)), capture_output=True, text=True, timeout=90, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    print((tmp_path / 'legacy-receipt.json').read_text(encoding='utf-8'))


async def probe():
    import aiohttp
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority

    runner = GatewayRunner()
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='legacy-test')
    db = authority.db
    # Rows exactly as earlier releases left them: the declared key verbatim, no binding receipt.
    for sid in ('legacy1', 'legacy2', 'legacy3'):
        db.create_session(sid, source='api_server')
        db.record_gateway_session_peer(sid, source='api_server', session_key='webui:' + sid)
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={'key': 'legacy-probe-secret', 'port': 0}))
    adapter.gateway_runner = runner
    adapter.set_message_handler(runner._handle_message)
    runner.adapters[Platform.API_SERVER] = adapter
    assert await adapter.connect()
    port = adapter._site._server.sockets[0].getsockname()[1]
    out = {}

    async def post(client, name, body, key):
        headers = {'Authorization': 'Bearer legacy-probe-secret', 'Idempotency-Key': name}
        if key:
            headers['X-Hermes-Session-Key'] = key
        async with client.post(f'http://127.0.0.1:{port}/v1/runs', json=body, headers=headers) as response:
            text = await response.text()
            out[name] = [response.status, json.loads(text) if response.content_type == 'application/json' else text]
            return response.status, out[name][1]
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as client:
            # The report's request: explicit session id plus the same declared key.
            status, body = await post(client, 'explicit', {'session_id': 'legacy1', 'input': 'hi'}, 'webui:legacy1')
            assert status == 202, (status, body)
            # Adoption kept the key: selecting the conversation by it alone reaches the same row.
            status, body = await post(client, 'explicit-then-declared', {'input': 'again'}, 'webui:legacy1')
            assert status == 202, (status, body)
            assert adapter._run_statuses[body['run_id']]['session_id'] == 'legacy1', adapter._run_statuses[body['run_id']]
            # A caller declaring a DIFFERENT key never adopts a row stamped with another key.
            status, body = await post(client, 'foreign', {'session_id': 'legacy2', 'input': 'hi'}, 'webui:intruder')
            assert status == 409 and body['error']['code'] == 'admission_conflict', (status, body)
            status, body = await post(client, 'keyless', {'session_id': 'legacy2', 'input': 'hi'}, None)
            assert status == 409 and body['error']['code'] == 'admission_conflict', (status, body)
            # Declared key alone (no session id) resolves the legacy conversation, not a new one.
            status, body = await post(client, 'declared', {'input': 'hi'}, 'webui:legacy3')
            assert status == 202, (status, body)
        route = lambda sid: db.get_session(sid)['session_key']
        assert route('legacy1') == 'agent:main:api_server:dm:legacy1', route('legacy1')
        assert route('legacy2') == 'webui:legacy2', 'a refused caller must not rewrite the row'
        assert route('legacy3') == 'agent:main:api_server:dm:legacy3', route('legacy3')
        Path(os.environ['HERMES_HOME'], 'legacy-receipt.json').write_text(encoding='utf-8', data=json.dumps(out))
    finally:
        await adapter.disconnect()


if __name__ == '__main__':
    import asyncio
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    os.environ.update(OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL='http://127.0.0.1:9/v1')
    Path(os.environ['HERMES_HOME'], 'config.yaml').write_text(
        'model:\n  default: local-wire-stub\n  provider: custom\n  base_url: http://127.0.0.1:9/v1\n'
        'auxiliary:\n  title_generation:\n    enabled: false\n')
    asyncio.run(probe())
