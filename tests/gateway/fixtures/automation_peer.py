"""Ordinary daemon, real terminal child, loopback-only relay/provider peers."""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shlex
import sqlite3
import subprocess
import sys
import threading
import time

import websockets
from tests.gateway.test_normal_runtime_boot import control
from local_recovery_probe import websocket, rpc


class Model(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages = body.get('messages', [])
        if messages:
            self.server.requests.append(body)
        last = next((m.get('content', '') for m in reversed(messages) if m['role'] == 'user'), '')
        if 'HOLD_AUTOMATION' in str(last):
            self.server.blocked.set()
            self.server.release.wait(120)
        message = {'role': 'assistant', 'content': 'AUTOMATION_ACK'}
        if 'SPAWN_AUTOMATION' in str(last) and messages[-1]['role'] != 'tool':
            code = ('import pathlib,time; p=pathlib.Path(' + repr(str(self.server.gate)) + '); '
                    '\nwhile not p.exists(): time.sleep(.02)\nprint("REAL_TERMINAL_COMPLETION",flush=True)')
            command = shlex.join([sys.executable, '-c', code])
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'terminal-spawn',
                'type': 'function', 'function': {'name': 'terminal', 'arguments': json.dumps({
                    'command': command, 'background': True, 'notify_on_complete': not getattr(self.server, 'watch', False),
                    **({'watch_patterns': ['REAL_TERMINAL_COMPLETION']} if getattr(self.server, 'watch', False) else {})})}}]}
        finish = 'tool_calls' if message.get('tool_calls') else 'stop'
        payload = json.dumps({'id': 'local', 'choices': [{'index': 0, 'message': message, 'finish_reason': finish}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}).encode()
        kind = 'application/json'
        if body.get('stream'):
            if message.get('tool_calls'):
                message['tool_calls'][0]['index'] = 0
            payload = ('data: ' + json.dumps({'id': 'local', 'choices': [
                {'index': 0, 'delta': message, 'finish_reason': finish}]}) + '\n\ndata: [DONE]\n\n').encode()
            kind = 'text/event-stream'
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


async def probe(root, base, mode):
    home, user = base / 'state', base / 'user'
    home.mkdir(); user.mkdir()
    model = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    model.requests = []
    model.blocked, model.release = threading.Event(), threading.Event()
    model.gate = home / 'release-terminal'
    threading.Thread(target=model.serve_forever, daemon=True).start()
    ready = asyncio.Queue()
    outgoing = []
    async def connector(ws):
        hello = json.loads(await ws.recv())
        assert hello['platform'] == 'telegram', hello
        descriptor = {'contract_version': 1, 'platform': 'telegram', 'label': 'Fixture',
            'max_message_length': 4096, 'supports_draft_streaming': False, 'supports_edit': True,
            'supports_threads': False, 'markdown_dialect': 'plain', 'len_unit': 'chars'}
        await ws.send(json.dumps({'type': 'descriptor', 'descriptor': descriptor}) + '\n')
        await ready.put(ws)
        async for line in ws:
            frame = json.loads(line)
            outgoing.append(frame)
            if frame['type'] == 'outbound':
                await ws.send(json.dumps({'type': 'outbound_result', 'requestId': frame['requestId'],
                    'result': {'success': True, 'message_id': 'fixture-final'}}) + '\n')
            elif frame['type'] == 'going_idle':
                await ws.send(json.dumps({'type': 'going_idle_ack'}) + '\n')

    def rows():
        with sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True) as db:
            db.row_factory = sqlite3.Row
            return [dict(r) for r in db.execute('SELECT * FROM session_admissions ORDER BY seq')]

    async def wait_for(predicate, seconds=35):
        async with asyncio.timeout(seconds):
            while not predicate():
                await asyncio.sleep(.03)

    proc = None
    try:
        async with websockets.serve(connector, '127.0.0.1', 0) as relay:
            url = f'http://127.0.0.1:{model.server_port}/v1'
            cfg = {'gateway': {'multiplex_profiles': False},
                'model': {'provider': 'custom', 'default': 'local-fixture', 'base_url': url},
                'streaming': {'enabled': False}, 'terminal': {'cwd': str(home)},
                'approvals': {'mode': 'off'},
                'auxiliary': {'title_generation': {'enabled': False}}}
            (home / 'config.yaml').write_text(json.dumps(cfg))
            env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if k in os.environ}
            env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
                OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, PYTHONUNBUFFERED='1',
                GATEWAY_RELAY_URL=f'ws://127.0.0.1:{relay.sockets[0].getsockname()[1]}/relay',
                GATEWAY_RELAY_PLATFORMS='telegram', GATEWAY_RELAY_BOT_IDS='{"telegram":"fixture-bot"}',
                GATEWAY_RELAY_ID='fixture-gateway', GATEWAY_RELAY_SECRET='fixture-secret')
            with (home / 'daemon.log').open('w+') as log:
                proc = subprocess.Popen([sys.executable, '-m', 'gateway.run'], cwd=root, env=env,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                try:
                    ws = await asyncio.wait_for(ready.get(), 35)
                    desc = {}
                    deadline = time.monotonic() + 35
                    while time.monotonic() < deadline:
                        try:
                            desc = await asyncio.to_thread(control, home, 'identify')
                            if desc.get('state') == 'ready':
                                break
                        except (OSError, ValueError):
                            pass
                        await asyncio.sleep(.05)
                    assert desc.get('state') == 'ready', desc
                    async def send(identity, text):
                        await ws.send(json.dumps({'type': 'inbound', 'event': {'text': text,
                            'message_id': identity, 'source': {'platform': 'telegram', 'chat_id': 'automation-chat',
                            'chat_type': 'dm', 'user_id': 'fixture-user'}}}) + '\n')
                    await send('spawn', 'SPAWN_AUTOMATION' if mode == 'terminal' else 'WARM_AUTOMATION')
                    await wait_for(lambda: any(r['request_id'] == 'spawn' and r['status'] == 'terminal' for r in rows()))
                    if mode == 'terminal':
                        assert 'REAL_TERMINAL_COMPLETION' in json.dumps(model.requests), 'model did not request real terminal'
                    else:
                        await send('timer-config', '/heartbeat every 1m TIMER_AUTOMATION')
                        await wait_for(lambda: 'Heartbeat' in json.dumps(outgoing))
                    sid = rows()[0]['target_session_id']
                    async with websocket(home, desc) as observer:
                        assert 'result' in await rpc(observer, 'session.resume', session_id=sid)
                    await send('held', 'HOLD_AUTOMATION')
                    assert await asyncio.to_thread(model.blocked.wait, 15)
                    await send('human-follower', 'HUMAN_FIFO_FOLLOWER')
                    await wait_for(lambda: any(r['request_id'] == 'human-follower' for r in rows()))
                    model.gate.touch()
                    # Real cadence: process polls every five seconds; heartbeat interval is one minute.
                    await asyncio.sleep(8 if mode == 'terminal' else 70)
                    ledger = rows()
                    automatic = [r for r in ledger if r['principal_id'].startswith('automation:')]
                    assert automatic, {'failure': 'producer ACK bypassed canonical durable FIFO', 'ledger': ledger,
                                       'daemon_tail': (home / 'daemon.log').read_text()[-6000:]}
                    assert automatic[0]['status'] == 'queued', automatic
                    assert next(r for r in ledger if r['request_id'] == 'held')['status'] == 'started'
                    assert automatic[0]['seq'] > next(r for r in ledger if r['request_id'] == 'human-follower')['seq']
                    model.release.set()
                    await wait_for(lambda: all(r['status'] == 'terminal' for r in rows()))
                    assert all(r['outcome'] == 'completed' for r in rows()), rows()
                    texts = [next((m.get('content', '') for m in reversed(r['messages']) if m['role'] == 'user'), '')
                             for r in model.requests if r.get('messages')]
                    marker = 'REAL_TERMINAL_COMPLETION' if mode == 'terminal' else 'TIMER_AUTOMATION'
                    assert marker in str(texts[-1]), texts
                    assert 'HUMAN_FIFO_FOLLOWER' in str(texts[-2]), texts
                    print(json.dumps({'daemon_pid': proc.pid, 'ledger': rows(), 'model_inputs': texts,
                                      'observer_closed_before_completion': True}))
                except BaseException:
                    print((home / 'daemon.log').read_text()[-10000:], file=sys.stderr)
                    raise
    finally:
        model.gate.touch()
        model.release.set()
        if proc is not None:
            proc.kill(); proc.wait(timeout=10)
        model.shutdown(); model.server_close()


if __name__ == '__main__':
    asyncio.run(probe(Path(__file__).resolve().parents[3], Path(sys.argv[1]), sys.argv[2]))
