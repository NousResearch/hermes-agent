"""A managed assignment preserves the image bytes committed at admission."""
import asyncio
import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import threading

from PIL import Image
import pytest

from tests.gateway.fixtures.local_recovery_probe import Model, child_env, daemon, rpc, websocket


@pytest.mark.parametrize('mode', ['normal', 'managed', 'safe'])
@pytest.mark.parametrize('model', ['gpt-4o', 'image-fixture'])
def test_admitted_images_reach_managed_worker(tmp_path, mode, model):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests = []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False, 'managed_workers': mode == 'managed'},
        'model': {'provider': 'custom', 'default': model, 'base_url': url},
        'agent': {'image_input_mode': 'native'}, 'platform_toolsets': {'cli': []},
        'auxiliary': {'title_generation': {'enabled': False}},
    }))
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url)

    async def exercise(desc):
        async with websocket(home, desc) as ws:
            created = await rpc(ws, 'session.create', request_id='create', source='cli', cwd=str(home),
                                model=model, provider='custom', base_url=url, toolsets=[],
                                safe_mode=mode == 'safe', ignore_user_config=mode == 'safe')
            assert 'result' in created, created
            sid = created['result']['session_id']
            image = home / 'cache/images/owned.png'
            image.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (16, 16), 'blue').save(image)
            original = image.read_bytes()
            accepted = await rpc(ws, 'prompt.submit', session_id=sid, input_id='image', text='IMAGE_SENTINEL',
                                 attachments=[{'path': str(image), 'mime': 'image/png'}])
            assert 'result' in accepted, accepted
            image.unlink()  # Execution must use retained bytes, never the staging path.
            async with asyncio.timeout(40):
                while True:
                    receipt = await rpc(ws, 'prompt.receipt', session_id=sid,
                                        admission_id=accepted['result']['admission_id'])
                    if receipt.get('result', {}).get('status') in ('terminal', 'unknown'):
                        break
                    await asyncio.sleep(.05)
            assert receipt['result']['outcome'] == 'completed', receipt
            import base64
            parts = [part for request in peer.requests for message in request['messages']
                     if message['role'] == 'user' and isinstance(message.get('content'), list)
                     for part in message['content'] if part.get('type') == 'image_url']
            assert parts, peer.requests
            assert base64.b64decode(parts[-1]['image_url']['url'].split(',', 1)[1]) == original
            if model == 'image-fixture':
                import sqlite3
                from contextlib import closing
                with closing(sqlite3.connect((home / 'state.db').as_uri() + '?mode=ro', uri=True)) as db:
                    assert db.execute('SELECT SUM(input_tokens) FROM session_model_usage WHERE session_id=? AND task=?',
                                      (sid, 'vision')).fetchone()[0] > 0

    try:
        with daemon(root, home, env, barrier=False) as (_, desc):
            asyncio.run(exercise(desc))
    finally:
        peer.release.set()
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)
