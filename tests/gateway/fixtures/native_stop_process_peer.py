"""Real native GUI Stop with a loopback model and fixture-owned background effects."""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shlex
import sys
import threading
import time
import traceback

HOME = Path(os.environ['HERMES_HOME'])
frames = []
requests = []
PERSIST = '--persist' in sys.argv


def command(name):
    # All effects wait behind one explicit test barrier. A surviving neighbor proves the
    # barrier was released, so absence of the stopped effect is not a negative timing test.
    code = ('import threading; from pathlib import Path; '
            f'r=Path({str(HOME)!r}); (r/{(name + "-start")!r}).touch(); '
            'e=threading.Event(); '
            '\nfor _ in range(2000):\n'
            ' if (r/"release").exists(): break\n'
            ' e.wait(.02)\n'
            f'else: raise SystemExit(3)\n(r/{(name + "-effect")!r}).touch()')
    return shlex.join([sys.executable, '-c', code])


class ModelPeer(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'data': [{'id': 'local-wire-stub', 'object': 'model'}]}).encode())

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        requests.append(request)
        messages = request.get('messages', [])
        tool_messages = [m for m in messages if m['role'] == 'tool']
        if not tool_messages:
            name, args = 'terminal', {'command': command('target'),
                'background': True, 'timeout': 30, 'persist_on_release': PERSIST, 'notify_on_complete': True}
        elif len(tool_messages) == 1:
            result = json.loads(tool_messages[-1]['content'])
            name, args = 'tool_call', {'calls': [{'name': 'process_manage', 'arguments': {
                'action': 'wait', 'session_id': result['session_id'], 'timeout': 30}}]}
        else:
            name, args = None, None
        content = {'role': 'assistant', 'content': None if name else 'STOP_PROBE_DONE'}
        if name:
            content['tool_calls'] = [{'id': f'call-{len(tool_messages)}', 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(args)}}]
        base = {'id': 'chatcmpl-fixture', 'model': 'local-wire-stub', 'created': 1}
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream' if request.get('stream') else 'application/json')
        self.end_headers()
        finish = 'tool_calls' if name else 'stop'
        if request.get('stream'):
            delta = dict(content)
            if name:
                delta['tool_calls'][0]['index'] = 0
            for value, reason in [(delta, None), ({}, finish)]:
                chunk = {**base, 'object': 'chat.completion.chunk', 'choices': [
                    {'index': 0, 'delta': value, 'finish_reason': reason}]}
                self.wfile.write(('data: ' + json.dumps(chunk) + '\n\n').encode())
                self.wfile.flush()
            self.wfile.write(b'data: [DONE]\n\n')
        else:
            self.wfile.write(json.dumps({**base, 'object': 'chat.completion', 'choices': [
                {'index': 0, 'message': content, 'finish_reason': finish}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}).encode())


async def probe():
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.run_api import start_gateway_api, stop_gateway_api
    from hermes_cli import web_server
    from hermes_cli.dashboard_auth.ws_tickets import mint_ticket
    from tools.process_registry import process_registry
    from websockets.asyncio.client import connect
    from tests.waiting import async_wait_until
    from hermes_state_runtime import list_session_admissions
    import psutil

    runner = GatewayRunner()
    authority = await initialize_session_authority(runner, profile_id=str(HOME), instance_id='isolated-stop-audit')
    api = await start_gateway_api(runner)
    web_server.app.state.auth_required = True
    receipt, neighbors = {'source': 'gui', 'model_wire': 'loopback synthetic'}, []
    try:
        ticket = mint_ticket(user_id='owner', provider='fixture')
        async with connect(api.api_origin.replace('http:', 'ws:') + '/api/ws?ticket=' + ticket) as ws:
            async def until(predicate):
                async with asyncio.timeout(45):
                    while True:
                        frame = json.loads(await ws.recv())
                        frames.append(frame)
                        if predicate(frame):
                            return frame

            async def rpc(method, **params):
                await ws.send(json.dumps({'jsonrpc': '2.0', 'id': method, 'method': method, 'params': params}))
                result = await until(lambda f: f.get('id') == method)
                assert 'error' not in result, result
                return result['result']

            created = await rpc('session.create', request_id='audit-gui', source='gui',
                                toolsets=['terminal'], cwd=str(HOME))
            sid = created['session_id']
            receipt['session_id'] = sid
            receipt['submit'] = await rpc('prompt.submit', session_id=sid, input_id='audit-stop-1', text='STOP_PROBE')
            await until(lambda f: f.get('params', {}).get('type') == 'tool.start'
                        and f['params'].get('payload', {}).get('name') in {'process_manage', 'tool_call'})
            await async_wait_until(lambda: (HOME / 'target-start').exists(), timeout=10, what='real process start marker')
            snapshot = await rpc('session.resume', session_id=sid)
            receipt['generation'] = snapshot['execution_generation']
            targets = [p for p in process_registry._running.values() if p.owner_task_id == sid]
            assert len(targets) == 1, targets
            proc = targets[0]
            receipt['process'] = {'id': proc.id, 'pid': proc.pid,
                'owner_task_id': proc.owner_task_id, 'route': proc.session_key,
                'persist_on_release': proc.persist_on_release, 'host_start_time': proc.host_start_time}
            identity = psutil.Process(proc.pid)
            birth = identity.create_time()
            for name, owner, route in [('other-owner', 'foreign-owner', proc.session_key),
                                       ('other-route', sid, 'foreign-route')]:
                neighbors.append(process_registry.spawn_local(command(name), cwd=str(HOME),
                    task_id=owner, owner_task_id=owner, session_key=route, persist_on_release=True))
            receipt['stop_attempt'] = time.time()
            receipt['stop_ack'] = await rpc('session.interrupt', session_id=sid,
                                          execution_generation=snapshot['execution_generation'])
            receipt['alive_at_ack'] = not proc.exited
            await asyncio.wait_for(authority.sessions[sid].task, 20)
            receipt['admissions'] = [{k: r.get(k) for k in ('status', 'outcome', 'generation')}
                for r in list_session_admissions(authority.db, session_id=sid, pending_only=False)]
            (HOME / 'release').touch()
            for neighbor in neighbors:
                assert await asyncio.to_thread(neighbor._completion_event.wait, 20)
                assert neighbor.exit_code == 0
            assert await asyncio.to_thread(proc._completion_event.wait, 20)
            receipt['completed_native'] = process_registry.poll(proc.id)
            receipt['completion_consumed'] = proc.id in process_registry._completion_consumed
            receipt['delayed_exists'] = (HOME / 'target-effect').exists()
            receipt['neighbors_completed'] = all((HOME / (name + '-effect')).exists()
                for name in ('other-owner', 'other-route'))
            receipt['same_process_alive'] = identity.is_running() and identity.create_time() == birth
            receipt['request_count'] = len([r for r in requests if 'messages' in r])
    finally:
        # Fixture-owned registry only; never inspect or signal the live gateway.
        for proc in list(process_registry._running.values()):
            process_registry.kill_process(proc.id)
        (HOME / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        (HOME / 'frames.json').write_text(json.dumps(frames, indent=2), encoding='utf-8')
        (HOME / 'requests.json').write_text(json.dumps(requests, indent=2), encoding='utf-8')
        await stop_gateway_api(api)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    peer = ThreadingHTTPServer(('127.0.0.1', 0), ModelPeer)
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    base_url = f'http://127.0.0.1:{peer.server_port}/v1'
    os.environ.update(OPENAI_API_KEY='explicit-loopback-fixture', OPENAI_BASE_URL=base_url)
    (HOME / 'config.yaml').write_text(json.dumps({'gateway': {'multiplex_profiles': False},
        'model': {'default': 'local-wire-stub', 'provider': 'custom', 'base_url': base_url},
        'auxiliary': {'title_generation': {'enabled': False}}, 'terminal': {'cwd': str(HOME)},
        'approvals': {'mode': 'off'}, 'display': {'busy_input_mode': 'queue'}}), encoding='utf-8')
    status = 0
    try:
        asyncio.run(probe())
    except BaseException:
        traceback.print_exc()
        status = 1
    finally:
        peer.shutdown()
        peer.server_close()
        sys.stdout.flush()
        sys.stderr.flush()
    os._exit(status)
