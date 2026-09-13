"""API controls and WS views resolve the same real blocking tool waiter."""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import threading

import aiohttp

from tests.gateway.fixtures.local_recovery_probe import daemon, websocket, rpc


class PromptPeer(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages = body.get('messages', [])
        if messages:
            self.server.requests.append(body)
        last = messages[-1] if messages else {}
        text = json.dumps(last)
        message = {'role': 'assistant', 'content': 'CONTROL_DONE'}
        if last.get('role') != 'tool' and ('ASK_APPROVAL' in text or 'ASK_CLARIFY' in text):
            approval = 'ASK_APPROVAL' in text
            name = 'terminal' if approval else 'clarify'
            args = {'command': 'rm -rf ' + str(self.server.target)} if approval else {
                'question': 'Choose the answer', 'choices': ['BLUE', 'GREEN']}
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{
                'id': 'call_control', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]}
        finish = 'tool_calls' if 'tool_calls' in message else 'stop'
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream' if body.get('stream') else 'application/json')
        self.end_headers()
        if body.get('stream'):
            delta = dict(message)
            if 'tool_calls' in delta:
                delta['tool_calls'][0]['index'] = 0
            data = {'id': 'control', 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}
            self.wfile.write(('data: ' + json.dumps(data) + '\n\ndata: [DONE]\n\n').encode())
        else:
            self.wfile.write(json.dumps({'id': 'control', 'choices': [{'index': 0, 'message': message, 'finish_reason': finish}]}).encode())


def test_api_and_ws_pending_controls_share_identity(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700); user.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), PromptPeer)
    peer.requests = []
    peer.target = home / 'owned-removal-target'
    peer.target.mkdir()
    (peer.target / 'owned.txt').write_text('owned')
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{peer.server_port}/v1'
    cfg = {'model': {'provider': 'custom', 'default': 'local-wire-stub', 'base_url': base},
           'gateway': {'platforms': {'api_server': {'enabled': True, 'port': port, 'host': '127.0.0.1'}}, 'multiplex_profiles': False},
           'auxiliary': {'title_generation': {'enabled': False}}, 'terminal': {'cwd': str(home)},
           'platform_toolsets': {'api_server': ['terminal', 'clarify']}, 'approvals': {'mode': 'manual'}}
    (home / 'config.yaml').write_text(json.dumps(cfg))
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=base, API_SERVER_KEY='ordinary-daemon-owned-secret',
               API_SERVER_ENABLED='true', API_SERVER_PORT=str(port), PYTHONUNBUFFERED='1')

    async def exercise(desc):
        headers = {'Authorization': 'Bearer ordinary-daemon-owned-secret'}
        async with aiohttp.ClientSession(headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as client:
            url = f'http://127.0.0.1:{port}/v1/runs'
            async def post(path, body):
                async with client.post(url + path, json=body) as response:
                    return response.status, await response.json()
            for kind, text in [('approval', 'ASK_APPROVAL'), ('clarify', 'ASK_CLARIFY')]:
                status, accepted = await post('', {'input': text, 'session_id': 'api-controls'})
                assert status == 202, accepted
                run = accepted['run_id']
                async with websocket(home, desc) as ws:
                    prompt = None
                    async with asyncio.timeout(15):
                        while not prompt:
                            resumed = await rpc(ws, 'session.resume', session_id='api-controls')
                            assert 'result' in resumed, resumed
                            prompt = next((p for p in resumed['result'].get('prompts', []) if p['kind'] == kind), None)
                            if not prompt:
                                await asyncio.sleep(.05)
                    async with client.get(url + '/' + run) as response:
                        projected = await response.json()
                    assert projected['pending_controls'] == [prompt], projected
                    body = {'request_id': prompt['prompt_id'], 'execution_generation': prompt['execution_generation']}
                    body.update({'choice': 'once'} if kind == 'approval' else {'answer': 'BLUE'})
                    path = '/' + run + ('/approval' if kind == 'approval' else '/clarify')
                    wrong = {**body, 'execution_generation': body['execution_generation'] + 1}
                    assert (await post(path, wrong))[0] == 409
                    foreign = {**body, 'request_id': 'not-this-prompt'}
                    assert (await post(path, foreign))[0] == 409
                    if kind == 'approval':
                        assert peer.target.exists()
                    status, result = await post(path, body)
                    assert status == 200 and result['status'] == 'resolved', result
                    late = await rpc(ws, 'approval.respond' if kind == 'approval' else 'clarify.respond',
                                     session_id='api-controls', execution_generation=body['execution_generation'],
                                     prompt_id=body['request_id'], **({'choice': 'once'} if kind == 'approval' else {'answer': 'GREEN'}))
                    assert 'error' in late or late['result']['status'] == 'already_resolved', late
                async with asyncio.timeout(20):
                    while True:
                        async with client.get(url + '/' + run) as response:
                            final = await response.json()
                        if final['status'] == 'completed':
                            break
                        await asyncio.sleep(.05)
                assert (await post(path, body))[0] == 409
            assert not peer.target.exists()
            tools = [m for req in peer.requests for m in req.get('messages', []) if m['role'] == 'tool']
            assert any('BLUE' in str(m['content']) for m in tools), tools
            print(json.dumps({'ordinary_daemon': True, 'shared_pending_identity': True, 'requests': len(peer.requests)}))
    try:
        with daemon(root, home, env, barrier=False) as (_, desc):
            asyncio.run(exercise(desc))
    finally:
        peer.shutdown(); peer.server_close()
