"""A background process a managed worker starts keeps its output past the worker's exit.

The worker is a per-turn interpreter. With a stdout pipe owned by its reader thread, the
child's first write after the worker exits raised SIGPIPE (``broken_pipe``) and its output
was gone; the next worker adopted only a PID. Under a worker the child writes to a
profile-scoped log and records its exit, so whoever adopts the checkpoint reads both.
"""
import asyncio
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import textwrap
import threading

import pytest

ROOT = Path(__file__).resolve().parents[2]
COMMAND = 'sleep 2; echo FIRST_$((40+2)); sleep 1; echo SECOND_$((6*7)); (exit 7)'


@pytest.mark.platforms("posix")
def test_worker_spawned_process_output_and_exit_survive_the_spawner(tmp_path):
    home = tmp_path / 'home'
    home.mkdir()
    env = {k: v for k, v in os.environ.items() if k not in ('HERMES_HOME', 'PYTHONPATH')}
    env.update(HERMES_HOME=str(home), PYTHONPATH=str(ROOT))
    # The spawner marks itself a registered worker, starts the process and dies at once.
    spawner = textwrap.dedent(f'''
        import json, os
        from agent import runtime_session_store
        runtime_session_store._worker_process = True
        from tools.process_registry import process_registry
        session = process_registry.spawn_local({COMMAND!r}, cwd={str(tmp_path)!r})
        print(json.dumps(session.id), flush=True)
        os._exit(0)
    ''')
    out = subprocess.run([sys.executable, '-c', spawner], env=env, cwd=ROOT, capture_output=True, text=True, encoding='utf-8', timeout=60, check=False)
    session_id = json.loads(out.stdout.strip().splitlines()[-1])
    adopter = textwrap.dedent(f'''
        import json
        from tools.process_registry import ProcessRegistry
        registry = ProcessRegistry()
        registry.recover_from_checkpoint()
        print(json.dumps(registry.wait({session_id!r}, timeout=30)))
    ''')
    out = subprocess.run([sys.executable, '-c', adopter], env=env, cwd=ROOT, capture_output=True, text=True, encoding='utf-8', timeout=60, check=False)
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result['status'] == 'exited', (result, out.stderr[-2000:])
    assert result['exit_code'] == 7 and result['output'] == 'FIRST_42\nSECOND_42\n', result


class Model(BaseHTTPRequestHandler):
    """Turn one starts the background command; turn two waits on it by id."""
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages = body.get('messages') or [{'role': 'system'}]
        last = messages[-1]
        if last['role'] == 'user' and last.get('content') == 'START':
            name, args = 'terminal', {'command': COMMAND, 'background': True}
        elif last['role'] == 'user' and last.get('content') == 'WAIT':
            proc = re.findall(r'proc_[0-9a-f]{12}', json.dumps(messages))[0]
            name, args = 'tool_call', {'calls': [{'name': 'process_manage',
                                                  'arguments': {'action': 'wait', 'session_id': proc, 'timeout': 30}}]}
        else:
            name, args = None, {}
        message = ({'role': 'assistant', 'content': None, 'tool_calls': [{'id': f'c{len(messages)}', 'index': 0,
                    'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]}
                   if name else {'role': 'assistant', 'content': 'OK'})
        choice = {'index': 0, 'delta': message, 'finish_reason': 'tool_calls' if name else 'stop'}
        frame = {'id': 'bg', 'model': 'm', 'choices': [choice], 'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}
        if body.get('stream'):
            payload, kind = ('data: ' + json.dumps(frame) + '\n\ndata: [DONE]\n\n').encode(), 'text/event-stream'
        else:
            choice['message'] = choice.pop('delta')
            payload, kind = json.dumps(frame).encode(), 'application/json'
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.mark.platforms("linux")
def test_next_managed_turn_reads_the_previous_workers_background_output(tmp_path):
    from tests.gateway.fixtures.local_recovery_probe import daemon, rpc, websocket
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False, 'managed_workers': True},
        'model': {'provider': 'custom', 'default': 'm', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False}}, 'approvals': {'mode': 'off'},
        'platform_toolsets': {'cli': ['terminal']}}))
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ') if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(ROOT),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url)

    def settled(request_id):
        with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
            return db.execute('SELECT status FROM session_admissions WHERE request_id=?', (request_id,)).fetchall() == [('terminal',)]

    async def exercise(desc):
        async with websocket(home, desc) as ws:
            created = await rpc(ws, 'session.create', request_id='bg', source='cli', cwd=str(home), model='m',
                                provider='custom', base_url=url, api_key='loopback-only', toolsets=['terminal'],
                                ignore_rules=True)
            sid = created['result']['session_id']
            for request_id, text in (('start', 'START'), ('wait', 'WAIT')):
                assert 'result' in await rpc(ws, 'prompt.submit', session_id=sid, input_id=request_id, text=text)
                async with asyncio.timeout(90):
                    while not settled(request_id):
                        await asyncio.sleep(.1)
            return (await rpc(ws, 'session.resume', session_id=sid))['result']['messages']
    try:
        with daemon(ROOT, home, env, barrier=False) as (_owner, desc):
            messages = asyncio.run(exercise(desc))
    finally:
        peer.shutdown()
        peer.server_close()
    waited = json.loads([m['content'] for m in messages if m['role'] == 'tool'][-1])
    assert waited['status'] == 'exited' and waited['exit_code'] == 7, waited
    assert waited['output'] == 'FIRST_42\nSECOND_42\n', waited
