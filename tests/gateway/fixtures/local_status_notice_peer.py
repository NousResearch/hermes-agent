"""Real owner, real AIAgent turn, authenticated local WS viewer: agent status and notices."""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
import traceback


class Peer(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"data": []}')

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.send_response(200)
        if request.get('stream'):
            self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
            for delta, fin in [({'role': 'assistant', 'content': 'DONE'}, None), ({}, 'stop')]:
                frame = {'id': 'x', 'object': 'chat.completion.chunk', 'created': 1, 'model': 'm',
                         'choices': [{'index': 0, 'delta': delta, 'finish_reason': fin}]}
                self.wfile.write(('data: ' + json.dumps(frame) + '\n\n').encode())
            self.wfile.write(b'data: [DONE]\n\n'); self.wfile.flush()
            return
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'id': 'x', 'object': 'chat.completion', 'created': 1, 'model': 'm',
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'DONE'}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}).encode())


async def probe():
    import websockets
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.run_api import start_gateway_api, stop_gateway_api
    from hermes_cli import web_server
    from hermes_cli.dashboard_auth.ws_tickets import mint_ticket
    import run_agent
    from agent.credits_tracker import AgentNotice

    original = run_agent.AIAgent.run_conversation

    def run_conversation(self, *args, **kwargs):
        self._emit_warning('PROBE_WARNING_STATUS')
        self._emit_status('PROBE_LIFECYCLE_STATUS')
        self._emit_notice(AgentNotice(text='PROBE_NOTICE', level='warn', key='probe', id='probe'))
        self._emit_notice_clear('probe')  # recovery withdraws the sticky notice (main: notification.clear)
        return original(self, *args, **kwargs)
    run_agent.AIAgent.run_conversation = run_conversation

    runner = GatewayRunner()
    await initialize_session_authority(runner, profile_id='fixture', instance_id='local')
    api = await start_gateway_api(runner)
    port = api.socket.getsockname()[1]
    web_server.app.state.auth_required = True
    ticket = mint_ticket(user_id='fixture-creator', provider='local-fixture')
    ws = await websockets.connect(f'ws://127.0.0.1:{port}/api/ws?ticket={ticket}')
    frames = []

    async def rpc(method, **params):
        await ws.send(json.dumps({'jsonrpc': '2.0', 'id': method, 'method': method, 'params': params}))
        async with asyncio.timeout(15):
            while True:
                frame = json.loads(await ws.recv())
                frames.append(frame)
                if frame.get('id') == method:
                    return frame
    try:
        sid = (await rpc('session.create', request_id='fresh', source='cli'))['result']['session_id']
        await rpc('session.resume', session_id=sid)
        await rpc('prompt.submit', session_id=sid, input_id='first', text='hi')
        async with asyncio.timeout(30):
            while not any(f.get('params', {}).get('type') == 'message.complete' for f in frames):
                frames.append(json.loads(await ws.recv()))
        events = [f['params'] for f in frames if f.get('method') == 'event']
        # The classic `hermes chat` view renders what reached it on stderr.
        import contextlib
        import io
        from hermes_cli.gateway_chat_view import GatewayChatView
        view, printed = GatewayChatView(None, {'stored_session_id': sid}), io.StringIO()
        with contextlib.redirect_stderr(printed), contextlib.redirect_stdout(io.StringIO()):
            for params in events:
                view._dispatch_event(params['type'], params.get('admission_id'), params.get('payload', {}))
        Path(os.environ['HERMES_HOME'], 'receipt.json').write_text(json.dumps({
            'events': [[p['type'], p.get('payload')] for p in events if p['type'] in (
                'status.update', 'notification.show', 'notification.clear')], 'cli_stderr': printed.getvalue()}))
    finally:
        await ws.close()
        await stop_gateway_api(api)


def main():
    suppress = len(sys.argv) > 1 and sys.argv[1] == 'suppress'
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Peer)
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    os.environ.update(OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, TERMINAL_ENV='local')
    Path(os.environ['HERMES_HOME'], 'config.yaml').write_text(
        f'model:\n  default: local-control\n  provider: custom\n  base_url: {url}\n'
        'streaming:\n  enabled: false\nauxiliary:\n  title_generation:\n    enabled: false\n'
        + ('display:\n  suppress_warning_notifications: true\n' if suppress else ''))
    try:
        asyncio.run(probe())
    finally:
        peer.shutdown()


if __name__ == '__main__':
    status = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        status = 1
    os._exit(status)
