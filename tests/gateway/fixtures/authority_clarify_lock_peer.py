"""#140400 end to end: a `hermes chat`-shaped turn's clarify batch answered from a second viewer.

Real GatewayRunner + session API WebSocket, a loopback model asking two questions, the real clarify
tool. Socket A creates a ``cli`` session and submits with ``clarify_batch`` (what GatewayAgent sends);
socket B (the Ink TUI / Desktop window) gets the one ``clarify`` server request and answers it per
question with ``clarify.lock``, the verb those windows send.
"""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import traceback
import time


class ModelPeer(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"data": []}')

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages = body.get('messages', [])
        self.server.requests.append(messages)
        asking = bool(messages) and not any(m['role'] == 'tool' for m in messages)
        message = {'role': 'assistant', 'content': 'CLARIFY_FINISHED'}
        if asking:
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{
                'index': 0, 'id': 'owned-clarify', 'type': 'function', 'function': {
                    'name': 'clarify', 'arguments': json.dumps({'questions': [
                        {'question': 'Pick a color', 'choices': ['blue', 'green']},
                        {'question': 'Pick a size'}]})}}]}
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream' if body.get('stream') else 'application/json')
        self.end_headers()
        if body.get('stream'):
            for delta, finish in [(message, None), ({}, 'tool_calls' if asking else 'stop')]:
                frame = {'id': 'local', 'object': 'chat.completion.chunk', 'created': 1,
                         'model': 'local', 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}
                self.wfile.write(('data: ' + json.dumps(frame) + '\n\n').encode())
            self.wfile.write(b'data: [DONE]\n\n')
        else:
            self.wfile.write(json.dumps({'id': 'local', 'choices': [{'index': 0, 'message': message,
                'finish_reason': 'tool_calls' if asking else 'stop'}]}).encode())
        self.wfile.flush()


async def probe(peer):
    import websockets
    from gateway.run import GatewayRunner
    from gateway.run_api import start_gateway_api, stop_gateway_api
    from gateway.session_authority import initialize_session_authority
    from hermes_cli import web_server

    runner = GatewayRunner()
    authority = await initialize_session_authority(runner, profile_id=os.environ['HERMES_HOME'], instance_id='lock')
    api = await start_gateway_api(runner)
    url = api.api_origin.replace('http:', 'ws:') + '/api/ws?token=' + web_server._SESSION_TOKEN
    frames = {}

    async def rpc(ws, rid, method, **params):
        await ws.send(json.dumps({'jsonrpc': '2.0', 'id': rid, 'method': method, 'params': params}))
        async with asyncio.timeout(20):
            while True:
                frame = json.loads(await ws.recv())
                frames.setdefault(id(ws), []).append(frame)
                if frame.get('id') == rid and 'method' not in frame:
                    return frame

    async def server_request(ws, method):
        for frame in frames.get(id(ws), []):
            if frame.get('method') == method and isinstance(frame.get('id'), str):
                return frame
        async with asyncio.timeout(20):
            while True:
                frame = json.loads(await ws.recv())
                frames.setdefault(id(ws), []).append(frame)
                if frame.get('method') == method and isinstance(frame.get('id'), str):
                    return frame

    a = b = None
    sid = None
    try:
        a, b = await websockets.connect(url), await websockets.connect(url)
        created = await rpc(a, 1, 'session.create', source='cli', request_id='create-1', cwd=os.getcwd())
        assert 'result' in created, created
        sid = created['result']['session_id']
        assert 'result' in await rpc(b, 1, 'session.resume', session_id=sid)
        submitted = await rpc(a, 2, 'prompt.submit', session_id=sid, input_id='ask-1', text='Ask me',
                              clarify_batch=True)
        assert 'result' in submitted, submitted
        request = await server_request(b, 'clarify')
        rid, questions = request['id'], request['params']['questions']
        assert [q['question'] for q in questions] == ['Pick a color', 'Pick a size'], questions
        first = await rpc(b, 3, 'clarify.lock', request_id=rid, question_id=questions[0]['qid'], answer='green')
        assert first.get('result', {}).get('remaining') == [questions[1]['qid']], first
        last = await rpc(b, 4, 'clarify.lock', request_id=rid, question_id=questions[1]['qid'], answer='large')
        assert last.get('result') == {'status': 'ok', 'remaining': []}, last
        await asyncio.wait_for(authority.sessions[sid].task, 20)
        tools = [m for messages in peer.requests for m in messages if m['role'] == 'tool']
        dumped = json.dumps(tools)
        assert 'green' in dumped and 'large' in dumped and 'submitted' in dumped, tools
        Path(os.environ['HERMES_HOME'], 'lock-passed').write_text('both locked answers reached the real model')
    finally:
        task = authority.sessions[sid].task if sid in authority.sessions else None
        if task is not None and not task.done():
            task.cancel()
        for ws in (a, b):
            if ws is not None:
                await ws.close()
        await stop_gateway_api(api)


def main():
    peer = ThreadingHTTPServer(('127.0.0.1', 0), ModelPeer)
    peer.requests = []
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    os.environ.update(OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url)
    Path(os.environ['HERMES_HOME'], 'config.yaml').write_text(
        f'model:\n  default: local\n  provider: custom\n  base_url: {url}\n'
        'agent:\n  clarify_timeout: 25\nauxiliary:\n  title_generation:\n    enabled: false\n')
    try:
        asyncio.run(probe(peer))
    finally:
        peer.shutdown()
        peer.server_close()


if __name__ == '__main__':
    status = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        status = 1
    os._exit(status)
