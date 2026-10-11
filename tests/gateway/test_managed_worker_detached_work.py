"""A managed worker exits at its turn boundary, so it must never hand out detached work.

``cronjob(action='run')`` and ``delegate_task`` schedule a daemon-pool unit only when the
session can receive the completion later; a worker process cannot (nobody drains its queue
after ``finished``, and recovery turns the orphaned ``running`` row into ``unknown``). The
work runs inline and its outcome lands in the same turn.
"""
import asyncio
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading

import pytest

from tests.gateway.fixtures.local_recovery_probe import daemon, rpc, websocket


def test_registered_worker_process_cannot_receive_async_completions(monkeypatch):
    from agent import runtime_session_store
    from gateway.session_context import async_delivery_supported
    monkeypatch.delenv('HERMES_KANBAN_TASK', raising=False)
    assert async_delivery_supported() is True
    monkeypatch.setattr(runtime_session_store, '_worker_process', True)
    assert async_delivery_supported() is False


class Model(BaseHTTPRequestHandler):
    """Parent turn: run the cron job, then answer. Cron turn: one marker reply."""
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages = body.get('messages') or [{'role': 'system'}]
        last = messages[-1]
        if last['role'] == 'user' and 'CRON_CHILD_GOAL' in str(last.get('content')):
            self.server.child_calls += 1
            message = {'role': 'assistant', 'content': 'CRON_CHILD_DONE'}
        elif last['role'] == 'user':
            call = {'name': 'cronjob_manage', 'arguments': {'action': 'run', 'job_id': self.server.job_id}}
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'run-1', 'index': 0, 'type': 'function',
                       'function': {'name': 'tool_call', 'arguments': json.dumps({'calls': [call]})}}]}
        else:
            message = {'role': 'assistant', 'content': 'PARENT_DONE'}
        choice = {'index': 0, 'delta': message, 'finish_reason': 'tool_calls' if message.get('tool_calls') else 'stop'}
        frame = {'id': 'detached', 'model': 'm', 'choices': [choice],
                 'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}
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
def test_managed_turn_runs_detachable_work_inline_and_leaves_no_orphan(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.child_calls = 0
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False, 'managed_workers': True},
        'model': {'provider': 'custom', 'default': 'm', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False}}, 'platform_toolsets': {'cli': ['cronjob']}}))
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ') if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url)
    created = subprocess.run([sys.executable, '-c', 'import json; from cron.jobs import create_job; '
                              'print(json.dumps(create_job("CRON_CHILD_GOAL", "every 1h", name="inline", deliver="local")["id"]))'],
                             env=env, cwd=root, capture_output=True, text=True, timeout=60, check=False)
    peer.job_id = json.loads(created.stdout.strip().splitlines()[-1])

    def rows(sql):
        with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
            return db.execute(sql).fetchall()

    async def exercise(desc):
        async with websocket(home, desc) as ws:
            session = await rpc(ws, 'session.create', request_id='detached', source='cli', cwd=str(home), model='m',
                                provider='custom', base_url=url, api_key='loopback-only', toolsets=['cronjob'],
                                ignore_rules=True)
            sid = session['result']['session_id']
            submitted = await rpc(ws, 'prompt.submit', session_id=sid, input_id='run', text='run the job')
            assert 'result' in submitted, submitted
            async with asyncio.timeout(90):
                while rows("SELECT status FROM session_admissions WHERE request_id='run'") != [('terminal',)]:
                    await asyncio.sleep(.1)
            return (await rpc(ws, 'session.resume', session_id=sid))['result']['messages']
    try:
        with daemon(root, home, env, barrier=False) as (_owner, desc):
            messages = asyncio.run(exercise(desc))
            # Past the worker boundary: nothing detached survives to be recovered as unknown.
            assert rows('SELECT delegation_id, state FROM async_delegations') == []
    finally:
        peer.shutdown()
        peer.server_close()
    tool = [m['content'] for m in messages if m['role'] == 'tool']
    assert len(tool) == 1 and '"execution_success": true' in tool[0], tool
    assert peer.child_calls == 1
