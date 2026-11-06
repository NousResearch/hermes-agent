"""Real canonical runtime; only inference HTTP is synthetic and loopback."""
import asyncio
import json
import os
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

CANARY = 'SPEC_REAL_EXECUTED_TODO_9572'


class Provider(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'data': [{'id': 'local-wire-stub', 'object': 'model'}]}).encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if 'messages' not in body:
            return self.do_GET()
        self.server.requests.append(body)
        users = [m.get('content', '') for m in body['messages'] if m['role'] == 'user']
        following = 'SPEC_NEXT' in json.dumps(users[-1:])
        count = len(self.server.requests)
        if count > 1 and not following:
            self.send_response(401)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': {'message': 'PRIVATE_PROVIDER_AUTH',
                'type': 'invalid_api_key', 'code': 'invalid_api_key'}}).encode())
            return
        message = {'role': 'assistant', 'content': 'Next answer.'} if following else {
            'role': 'assistant', 'content': 'Safe progress.', 'tool_calls': [{
                'id': 'call_spec_real', 'type': 'function', 'function': {'name': 'tool_call',
                'arguments': json.dumps({'calls': [{'name': 'todo_list', 'arguments': {'todos': [{'id': '1', 'content': CANARY, 'status': 'completed'}]}}]})}}]}
        finish = 'stop' if following else 'tool_calls'
        base = {'id': 'chatcmpl-spec', 'model': 'local-wire-stub', 'created': 1}
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream' if body.get('stream') else 'application/json')
        self.end_headers()
        if body.get('stream'):
            if 'tool_calls' in message:
                message['tool_calls'][0]['index'] = 0
            for delta, reason in ((message, None), ({}, finish)):
                frame = {**base, 'object': 'chat.completion.chunk', 'choices': [
                    {'index': 0, 'delta': delta, 'finish_reason': reason}]}
                self.wfile.write(('data: ' + json.dumps(frame) + '\n\n').encode())
                self.wfile.flush()
            self.wfile.write(b'data: [DONE]\n\n')
        else:
            self.wfile.write(json.dumps({**base, 'object': 'chat.completion', 'choices': [
                {'index': 0, 'message': message, 'finish_reason': finish}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}).encode())


def frames(wire):
    return [json.loads(line[6:]) for line in wire.decode().splitlines() if line.startswith('data: ')]


@pytest.mark.asyncio
@pytest.mark.parametrize('public', [False, True], ids=['generic', 'public'])
async def test_real_tool_failure_continuation(tmp_path, monkeypatch, public):
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Provider)
    peer.requests = []
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{peer.server_port}/v1'
    home = Path(os.environ['HERMES_HOME'])
    home.mkdir(exist_ok=True, parents=True)
    monkeypatch.setenv('OPENAI_API_KEY', 'explicit-loopback-only')
    monkeypatch.setenv('OPENAI_BASE_URL', base)
    (home / 'config.yaml').write_text(
        f'model:\n  default: local-wire-stub\n  provider: custom\n  base_url: {base}\n'
        'auxiliary:\n  title_generation:\n    enabled: false\n'
        'platform_toolsets:\n  api_server: [todo]\n'
        'memory:\n  memory_enabled: false\n  user_profile_enabled: false\n'
        'agent:\n  environment_probe: false\n'
        f'terminal:\n  cwd: {tmp_path}\n')
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_results import admission_result
    runner = GatewayRunner()
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='independent-real')
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={'key': 'fixture-api-key'}))
    adapter.gateway_runner = runner
    adapter._session_db = authority.db
    adapter.set_message_handler(runner._handle_message)
    runner.adapters[Platform.API_SERVER] = adapter
    app = web.Application()
    app.router.add_post('/v1/responses', adapter._handle_responses)
    app.router.add_get('/v1/responses/{response_id}', adapter._handle_get_response)
    headers = {'Authorization': 'Bearer fixture-api-key', 'Idempotency-Key': 'spec-real'}
    try:
        async with TestClient(TestServer(app)) as client:
            response = await client.post('/v1/responses', json={
                'input': 'SPEC_FIRST', 'stream': True, 'public_commentary': public}, headers=headers)
            assert response.status == 200
            async with asyncio.timeout(30):
                wire = await response.read()
            terminal = frames(wire)[-1]['response']
            stored = adapter._current_response_store().get(terminal['id'])
            retrieved = await (await client.get('/v1/responses/' + terminal['id'], headers=headers)).json()
            rows = authority.db._read_all('SELECT admission_id FROM session_admissions')
            result = admission_result(authority.db, rows[0]['admission_id'])
            following = await client.post('/v1/responses', json={'input': 'SPEC_NEXT', 'stream': True,
                'public_commentary': public, 'previous_response_id': terminal['id']},
                headers={**headers, 'Idempotency-Key': 'spec-real-next'})
            assert following.status == 200
            async with asyncio.timeout(30):
                next_wire = await following.read()
        assert frames(next_wire)[-1]['response']['status'] == 'completed'
        assert len(peer.requests) == 3  # tool request, failed inference, one continuation
        assert any(m.get('role') == 'tool' and CANARY in str(m.get('content'))
                   for req in peer.requests for m in req['messages']), 'real tool never executed'
        assert CANARY in repr(result), 'canonical receipt lost real executed tool'
        if public:
            assert CANARY not in wire.decode(), 'public wire exposed tool protocol'
            assert CANARY not in json.dumps(retrieved)
            assert 'PRIVATE_PROVIDER_AUTH' not in wire.decode()
            assert terminal['status'] == 'failed'
            assert retrieved == terminal
        assert CANARY in repr(stored['conversation_history']), 'public failed snapshot lost real tool context'
        assert CANARY in repr(peer.requests[-1]['messages']), 'next model request lost real tool context'
    finally:
        peer.shutdown()
        peer.server_close()
        thread.join()
        adapter._response_store.close()
        adapter._run_idempotency_store.close()
        authority.db.close()
