"""T3 / LC-8 on a real gateway: CLI sessions attached through the gateway's own ticket run their tools in
managed workers under their client's toolchain, and the client environment never shows in a public
projection (session.info, session.list, export, search)."""
import asyncio
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sqlite3
import threading

import pytest

from tests.waiting import async_wait_until

ROOT = Path(__file__).resolve().parents[2]
PROBE = 'printf "%s|%s" "$NVM_DIR" "$(whichclient)"'


class Model(BaseHTTPRequestHandler):
    """``PROBE`` runs the terminal probe; anything else ends the turn."""
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages = body.get('messages') or [{'role': 'system'}]
        last = messages[-1]
        call = last['role'] == 'user' and last.get('content') == 'PROBE'
        message = ({'role': 'assistant', 'content': None, 'tool_calls': [{'id': f'c{len(messages)}', 'index': 0,
                    'type': 'function', 'function': {'name': 'terminal', 'arguments': json.dumps({'command': PROBE})}}]}
                   if call else {'role': 'assistant', 'content': 'OK'})
        choice = {'index': 0, 'delta': message, 'finish_reason': 'tool_calls' if call else 'stop'}
        frame = {'id': 'env', 'model': 'm', 'choices': [choice], 'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}
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


def _bin(root, name):
    path = root / name / 'bin'
    path.mkdir(parents=True)
    script = path / 'whichclient'
    script.write_text(f'#!/bin/sh\necho {name}\n')
    script.chmod(0o755)
    return path


@pytest.mark.platforms("linux")
def test_managed_workers_run_each_sessions_tools_under_its_client_toolchain(tmp_path):
    from tests.gateway.fixtures.local_recovery_probe import daemon, rpc, websocket
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    (user / '.hushlogin').touch()  # Ubuntu's bash.bashrc sudo hint on stdout
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False, 'managed_workers': True},
        'model': {'provider': 'custom', 'default': 'm', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False}}, 'approvals': {'mode': 'off'},
        'platform_toolsets': {'cli': ['terminal']}}))
    system_path = os.environ['PATH']
    bins = {name: _bin(tmp_path, name) for name in ('daemon', 'alpha', 'beta')}
    env = {k: os.environ[k] for k in ('LANG', 'TZ') if k in os.environ}
    env.update(PATH=f'{bins["daemon"]}{os.pathsep}{system_path}', HOME=str(user), USERPROFILE=str(user),
               HERMES_HOME=str(home), PYTHONPATH=str(ROOT), OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url)
    # Never printed by the probe: if it shows anywhere public, the private client map leaked.
    sentinels = {name: f'--title=client-env-sentinel-{name}' for name in ('alpha', 'beta')}

    def settled(request_id):
        with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
            rows = db.execute('SELECT status FROM session_admissions WHERE request_id=?', (request_id,)).fetchall()
            return rows == [('terminal',)]

    async def exercise(desc):
        outputs, public = {}, []
        async with websocket(home, desc) as ws:
            sessions = {}
            for name in ('alpha', 'beta', 'desktop'):
                client = {} if name == 'desktop' else {'client_env': {
                    'PATH': f'{bins[name]}{os.pathsep}{system_path}', 'NVM_DIR': f'/nvm/{name}',
                    'NODE_OPTIONS': sentinels[name]}}
                created = await rpc(ws, 'session.create', request_id=name, source='cli', cwd=str(home), model='m',
                                    provider='custom', base_url=url, api_key='loopback-only', toolsets=['terminal'],
                                    ignore_rules=True, **client)
                assert 'result' in created, created
                sessions[name] = created['result']['session_id']
                public.append(created['result'])
            for name, sid in sessions.items():
                assert 'result' in await rpc(ws, 'prompt.submit', session_id=sid, input_id=f'probe-{name}', text='PROBE')
            for name in sessions:
                await async_wait_until(lambda n=name: settled(f'probe-{n}'), timeout=90, interval=.1,
                                       what=f'{name} probe to settle')
            for name, sid in sessions.items():
                messages = (await rpc(ws, 'session.resume', session_id=sid))['result']['messages']
                outputs[name] = json.loads([m['content'] for m in messages if m['role'] == 'tool'][-1])['output']
                public.append(await rpc(ws, 'session.info', session_id=sid))
            public.append(await rpc(ws, 'session.list', limit=50))
        return outputs, sessions, public
    try:
        with daemon(ROOT, home, env, barrier=False) as (_owner, desc):
            outputs, sessions, public = asyncio.run(exercise(desc))
    finally:
        peer.shutdown()
        peer.server_close()
    assert outputs['alpha'].strip().endswith('/nvm/alpha|alpha'), outputs
    assert outputs['beta'].strip().endswith('/nvm/beta|beta'), outputs
    assert outputs['desktop'].strip().endswith('|daemon'), outputs  # no client_env: the gateway's own
    from hermes_state import SessionDB
    db = SessionDB(home / 'state.db')
    try:
        public += [db.export_session(sid, include_compacted=True) for sid in sessions.values()]
        public += [db.list_sessions_rich(search_query='sentinel', order_by_last_active=True, include_archived=True),
                   db.list_sessions_rich(limit=50), db.search_sessions(limit=50),
                   db.search_messages('sentinel')]
    finally:
        db.close()
    text = json.dumps(public, default=str)
    assert not [s for s in sentinels.values() if s in text], 'client_env leaked into a public projection'
