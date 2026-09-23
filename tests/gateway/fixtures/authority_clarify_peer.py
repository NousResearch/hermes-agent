"""Owned loopback model, real clarify tool, native card, two authenticated sockets."""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import traceback


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
                        {'question': 'Pick a color', 'choices': ['blue', 'green']}]})}}]}
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
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.base import BasePlatformAdapter, SendResult
    from gateway.run import GatewayRunner
    from gateway.run_api import start_gateway_api, stop_gateway_api
    from gateway.session import SessionSource
    from gateway.session_authority import initialize_session_authority
    from hermes_cli import web_server
    from tools import clarify_gateway

    class Adapter(BasePlatformAdapter):
        async def connect(self, **kwargs):
            return True
        async def disconnect(self):
            pass
        async def send(self, chat_id, content, **kwargs):
            return SendResult(success=True, message_id='owned')
        async def send_typing(self, *args, **kwargs):
            pass
        async def get_chat_info(self, chat_id):
            return {'id': chat_id}
        async def send_clarify(self, **kwargs):
            return SendResult(success=True, message_id='owned-card')

    runner = GatewayRunner()
    authority = await initialize_session_authority(runner, profile_id='fixture', instance_id='clarify')
    runner.adapters[Platform.TELEGRAM] = Adapter(PlatformConfig(enabled=True, token='fixture'), Platform.TELEGRAM)
    ref = authority.register(SessionSource(platform=Platform.TELEGRAM, chat_id='clarify-chat',
                                           chat_type='dm', user_id='fixture-user'))
    api = await start_gateway_api(runner)
    url = api.api_origin.replace('http:', 'ws:') + '/api/ws?token=' + web_server._SESSION_TOKEN

    async def rpc(ws, rid, method, **params):
        await ws.send(json.dumps({'jsonrpc': '2.0', 'id': rid, 'method': method,
                                 'params': {'session_id': ref.session_id, **params}}))
        async with asyncio.timeout(10):
            while True:
                frame = json.loads(await ws.recv())
                if frame.get('id') == rid:
                    return frame

    a = b = None
    try:
        a, b = await websockets.connect(url), await websockets.connect(url)
        await rpc(a, 1, 'session.resume')
        await rpc(b, 1, 'session.resume')
        assert 'result' in await rpc(a, 2, 'prompt.submit', input_id='clarify-first', text='Ask my color')
        try:
            async with asyncio.timeout(15):
                while True:
                    snapshot = await rpc(b, 3, 'session.resume')
                    prompts = snapshot['result']['prompts']
                    if prompts:
                        break
                    await asyncio.sleep(.05)
        except TimeoutError:
            raise AssertionError('real clarify waiter missing from shared snapshot') from None
        prompt = prompts[0]
        assert prompt['kind'] == 'clarify', prompt
        assert clarify_gateway.has_pending(authority.sessions[ref.session_id].route)
        await a.close()
        a = None
        assert (await rpc(b, 4, 'session.resume'))['result']['prompts'][0]['prompt_id'] == prompt['prompt_id']
        identity = {'prompt_id': prompt['prompt_id'], 'execution_generation': prompt['execution_generation']}
        stale = await rpc(b, 5, 'clarify.respond', **{**identity, 'execution_generation': -1}, answer='green')
        assert stale['error']['data']['reason'] == 'stale_generation'
        response = await rpc(b, 6, 'clarify.respond', **identity, answer='green')
        assert response['result']['status'] == 'resolved', response
        duplicate = await rpc(b, 7, 'clarify.respond', **identity, answer='SHOULD_NOT_REPLACE')
        assert duplicate.get('result', {}).get('status') == 'already_resolved' or duplicate.get('error', {}).get('data', {}).get('reason') == 'stale_generation', duplicate
        await asyncio.wait_for(authority.sessions[ref.session_id].task, 15)
        tools = [m for messages in peer.requests for m in messages if m['role'] == 'tool']
        assert any('green' in json.dumps(m) for m in tools), tools
        assert not any('SHOULD_NOT_REPLACE' in json.dumps(m) for m in tools)
        assert (await rpc(b, 8, 'session.resume'))['result']['prompts'] == []
        Path(os.environ['HERMES_HOME'], 'clarify-passed').write_text('reply reached real model; stale reply rejected')
    finally:
        clarify_gateway.clear_session(authority.sessions[ref.session_id].route)
        task = authority.sessions[ref.session_id].task
        if task is not None:
            await asyncio.wait_for(task, 15)
        for ws in (a, b):
            if ws is not None:
                await ws.close()
        await stop_gateway_api(api)


def main():
    peer = ThreadingHTTPServer(('127.0.0.1', 0), ModelPeer)
    peer.requests = []
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    os.environ.update(OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, TELEGRAM_ALLOWED_USERS='fixture-user')
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
