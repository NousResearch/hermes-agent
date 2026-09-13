"""Disposable real approval worker, loopback model, authenticated WS clients."""
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
        request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages = request.get('messages', [])
        self.server.requests.append(request)
        tool = bool(messages) and not any(m['role'] == 'tool' for m in messages)
        message = {'role': 'assistant', 'content': 'APPROVAL_FINISHED'}
        if tool:
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{
                'id': 'owned-removal', 'type': 'function', 'function': {
                    'name': 'terminal', 'arguments': json.dumps({'command': self.server.command})}}]}
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream' if request.get('stream') else 'application/json')
        self.end_headers()
        if request.get('stream'):
            if tool:
                message['tool_calls'][0]['index'] = 0
            for delta, finish in [(message, None), ({}, 'tool_calls' if tool else 'stop')]:
                frame = {'id': 'local-approval', 'object': 'chat.completion.chunk', 'created': 1,
                         'model': 'local-control', 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}
                self.wfile.write(('data: ' + json.dumps(frame) + '\n\n').encode())
            self.wfile.write(b'data: [DONE]\n\n')
            self.wfile.flush()
            return
        self.wfile.write(json.dumps({'id': 'local-approval', 'object': 'chat.completion',
            'created': 1, 'model': 'local-control', 'choices': [{'index': 0, 'message': message,
            'finish_reason': 'tool_calls' if tool else 'stop'}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}).encode())


async def probe(peer, target):
    import websockets
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.base import BasePlatformAdapter, SendResult
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from gateway.session_authority import initialize_session_authority
    from hermes_cli import web_server
    from gateway.run_api import start_gateway_api, stop_gateway_api
    from tools.approval import list_gateway_approvals, resolve_gateway_approval
    from hermes_state_runtime import list_session_admissions
    from run_agent import AIAgent

    class NativeAdapter(BasePlatformAdapter):
        def __init__(self):
            super().__init__(PlatformConfig(enabled=True, token='fixture'), Platform.TELEGRAM)
            self.card = asyncio.Event()
        async def connect(self, *, is_reconnect=False):
            return True
        async def disconnect(self):
            pass
        async def send(self, chat_id, content, reply_to=None, metadata=None):
            return SendResult(success=True, message_id='fixture')
        async def edit_message(self, chat_id, message_id, content, *, finalize=False):
            return SendResult(success=True, message_id=message_id)
        async def send_typing(self, chat_id, metadata=None):
            pass
        async def get_chat_info(self, chat_id):
            return {'id': chat_id}
        async def send_exec_approval(self, **kwargs):
            self.card.set()
            return SendResult(success=True, message_id='card')

    runner = GatewayRunner()
    authority = await initialize_session_authority(runner, profile_id='fixture', instance_id='controls')
    adapter = NativeAdapter()
    runner.adapters[Platform.TELEGRAM] = adapter
    ref = authority.register(SessionSource(platform=Platform.TELEGRAM, chat_id='controls-chat',
                                           chat_type='dm', user_id='fixture-user'))
    api = await start_gateway_api(runner)
    port = api.socket.getsockname()[1]
    frames = []

    async def rpc(ws, rid, method, **params):
        await ws.send(json.dumps({'jsonrpc': '2.0', 'id': rid, 'method': method,
                                  'params': {'session_id': ref.session_id, **params}}))
        async with asyncio.timeout(10):
            while True:
                frame = json.loads(await ws.recv())
                frames.append(frame)
                if frame.get('id') == rid:
                    return frame

    async def connect():
        return await websockets.connect(f'ws://127.0.0.1:{port}/api/ws?token={web_server._SESSION_TOKEN}')

    a = b = None
    try:
        a, b = await connect(), await connect()
        await rpc(a, 1, 'session.resume')
        await rpc(b, 1, 'session.resume')
        submitted = await rpc(a, 2, 'prompt.submit', input_id='approval-first', text='Remove owned fixture')
        assert 'result' in submitted, submitted
        await asyncio.wait_for(adapter.card.wait(), 20)
        route = authority.sessions[ref.session_id].route
        assert list_gateway_approvals(route), 'real worker never blocked for approval'
        assert target.exists(), 'effect preceded human consent'
        # Rendering completes before the callback registers its shared projection.
        async with asyncio.timeout(3):
            while True:
                snapshot = await rpc(b, 3, 'session.resume')
                if snapshot['result'].get('prompts'):
                    break
                await asyncio.sleep(.02)
    except TimeoutError:
        raise AssertionError('approval-blocked worker absent from shared attach snapshot') from None
    else:
        prompt = snapshot['result']['prompts'][0]
        assert prompt['kind'] == 'approval'
        queued = await rpc(b, 4, 'prompt.submit', input_id='queued-after', text='Next turn')
        assert queued['result']['status'] == 'queued'
        await a.close()
        a = None
        resumed = await rpc(b, 5, 'session.resume')
        assert resumed['result']['prompts'][0]['prompt_id'] == prompt['prompt_id']
        identity = {'execution_generation': prompt['execution_generation'], 'prompt_id': prompt['prompt_id']}
        stale = await rpc(b, 6, 'approval.respond', **{**identity, 'execution_generation': -1}, choice='once')
        assert stale['error']['data']['reason'] == 'stale_generation', stale
        invalid = await rpc(b, 7, 'approval.respond', **identity, choice='SECRET_MUST_NOT_REPLAY')
        assert invalid['error']['data']['reason'] == 'invalid_params', invalid
        assert target.exists()
        answered = await rpc(b, 8, 'approval.respond', **identity, choice='once')
        assert answered['result']['status'] == 'resolved', answered
        duplicate = await rpc(b, 9, 'approval.respond', **identity, choice='deny')
        assert duplicate.get('result', {}).get('status') == 'already_resolved' or duplicate.get('error', {}).get('data', {}).get('reason') == 'stale_generation', duplicate
        await asyncio.wait_for(authority.sessions[ref.session_id].task, 25)
        assert not target.exists(), 'authorized real terminal effect missing'
        assert isinstance(authority.agent(ref), AIAgent)
        rows = list_session_admissions(authority.db, session_id=ref.session_id, pending_only=False)
        assert len(rows) == 2 and all(r['status'] == 'terminal' for r in rows), rows
        final = await rpc(b, 10, 'session.resume')
        assert final['result']['prompts'] == []
        replay = await rpc(b, 11, 'session.events.since', replay_epoch=final['result']['replay_epoch'], last_sequence=0)
        assert 'SECRET_MUST_NOT_REPLAY' not in json.dumps(replay)
        assert 'SECRET_MUST_NOT_REPLAY' not in json.dumps(authority.db.get_messages_as_conversation(ref.session_id))
        assert any(f.get('type') == 'approval.request' for f in replay['result']['events'])
        Path(os.environ['HERMES_HOME'], 'receipt.json').write_text(json.dumps({
            'real_terminal_effect': True, 'detach_kept_pending': True, 'fifo_bypassed': True,
            'duplicate_rejected': True, 'model_requests': len(peer.requests), 'admissions': len(rows)}))
    finally:
        route = authority.sessions[ref.session_id].route
        resolve_gateway_approval(route, 'deny', resolve_all=True)
        task = authority.sessions[ref.session_id].task
        if task is not None:
            await asyncio.wait_for(task, 30)
        for ws in (a, b):
            if ws is not None:
                await ws.close()
        await stop_gateway_api(api)


def main():
    import shlex
    target = Path(os.environ['HERMES_HOME'], 'owned-removal')
    target.mkdir()
    (target / 'owned.txt').write_text('disposable')
    peer = ThreadingHTTPServer(('127.0.0.1', 0), ModelPeer)
    peer.requests = []
    peer.command = 'rm -r -- ' + shlex.quote(str(target))
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    os.environ.update(OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url,
                      TELEGRAM_ALLOWED_USERS='fixture-user', TERMINAL_ENV='local')
    Path(os.environ['HERMES_HOME'], 'config.yaml').write_text(
        f'model:\n  default: local-control\n  provider: custom\n  base_url: {url}\n'
        'approvals:\n  mode: manual\n  timeout: 30\n'
        'streaming:\n  enabled: false\n'
        'auxiliary:\n  title_generation:\n    enabled: false\n')
    try:
        asyncio.run(probe(peer, target))
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
