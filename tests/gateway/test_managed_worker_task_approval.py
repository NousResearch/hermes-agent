"""A managed local worker receives only owner-admitted request evidence."""
import asyncio
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sqlite3
import threading
import time

import pytest

from tests.gateway.fixtures.local_recovery_probe import daemon, rpc, websocket
from tests.waiting import async_wait_until


class Model(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages = body.get('messages') or []
        if any('<command>' in str(m.get('content') or '') for m in messages):
            prompt = str(messages[-1].get('content') or '')
            self.server.reviews.append(prompt)
            if 'late.txt' in prompt:
                self.server.review_started.set()
                self.server.review_release.wait(20)
            answer = 'ESCALATE' if 'Review only' in prompt else 'APPROVE'
            message = {'role': 'assistant', 'content': answer}
        elif messages and messages[-1].get('role') == 'user':
            text = str(messages[-1].get('content') or '')
            target = (self.server.review_target if 'Review only' in text else
                      self.server.late_target if 'late.txt' in text else self.server.edit_target)
            command = f'python -c "from pathlib import Path; Path({str(target)!r}).write_text(\'done\')"'
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{
                'id': 'task-tool', 'type': 'function', 'function': {'name': 'terminal',
                'arguments': json.dumps({'command': command, 'timeout': 10})}}]}
        else:
            message = {'role': 'assistant', 'content': 'TURN_DONE'}
        choice = {'index': 0, 'message': message,
                  'finish_reason': 'tool_calls' if message.get('tool_calls') else 'stop'}
        frame = {'id': 'managed-task', 'model': 'fixture-model', 'choices': [choice],
                 'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}
        if body.get('stream'):
            choice['delta'] = choice.pop('message')
            for index, call in enumerate(choice['delta'].get('tool_calls', [])):
                call['index'] = index
            payload, kind = ('data: ' + json.dumps(frame) + '\n\ndata: [DONE]\n\n').encode(), 'text/event-stream'
        else:
            payload, kind = json.dumps(frame).encode(), 'application/json'
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.mark.platforms('linux')
def test_managed_worker_task_review_and_advice_boundary(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.edit_target, peer.review_target, peer.late_target = (
        home / 'allowed.txt', home / 'review-only.txt', home / 'late.txt')
    peer.reviews, peer.review_started, peer.review_release = [], threading.Event(), threading.Event()
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False, 'managed_workers': True},
        'model': {'provider': 'custom', 'default': 'fixture-model', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False}, 'approval': {
            'provider': 'custom', 'model': 'fixture-model', 'base_url': url, 'api_key': 'loopback-only'}},
        'approvals': {'mode': 'smart'}, 'platform_toolsets': {'cli': ['terminal']}}))
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ') if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, HERMES_EXEC_ASK='1')

    def terminal(request_id):
        with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
            return db.execute('SELECT status FROM session_admissions WHERE request_id=?', (request_id,)).fetchall()

    async def exercise(desc):
        async with websocket(home, desc) as ws:
            created = await rpc(ws, 'session.create', request_id='managed-task', source='cli', cwd=str(home),
                                model='fixture-model', provider='custom', base_url=url, api_key='loopback-only',
                                toolsets=['terminal'], ignore_rules=True)
            assert 'result' in created, created
            sid = created['result']['session_id']
            submitted = await rpc(ws, 'prompt.submit', session_id=sid, input_id='edit',
                                  text='Edit only allowed.txt')
            assert 'result' in submitted, submitted
            await async_wait_until(lambda: terminal('edit') == [('terminal',)], timeout=45, interval=.05,
                                   what='managed edit settlement')
            assert peer.edit_target.read_text() == 'done'
            assert not (await rpc(ws, 'session.resume', session_id=sid))['result']['prompts']
            submitted = await rpc(ws, 'prompt.submit', session_id=sid, input_id='advice',
                                  text='Review only review-only.txt; give advice')
            assert 'result' in submitted, submitted
            deadline = time.monotonic() + 45
            while True:
                snapshot = await rpc(ws, 'session.resume', session_id=sid)
                prompt = next((p for p in snapshot['result']['prompts'] if p['kind'] == 'approval'), None)
                if prompt:
                    await rpc(ws, 'approval.respond', session_id=sid, prompt_id=prompt['prompt_id'],
                              execution_generation=prompt['execution_generation'], choice='deny')
                    break
                assert time.monotonic() < deadline, 'review-only turn never escalated'
                await asyncio.sleep(.05)
            await async_wait_until(lambda: terminal('advice') == [('terminal',)], timeout=45, interval=.05,
                                   what='managed advice settlement')
            assert not peer.review_target.exists()
            submitted = await rpc(ws, 'prompt.submit', session_id=sid, input_id='late',
                                  text='Edit only late.txt')
            assert 'result' in submitted, submitted
            assert await asyncio.to_thread(peer.review_started.wait, 20)
            receipt = await rpc(ws, 'prompt.receipt', session_id=sid,
                                admission_id=submitted['result']['admission_id'])
            assert 'result' in receipt, receipt
            stopped = await rpc(ws, 'session.interrupt', session_id=sid,
                                execution_generation=receipt['result']['execution_generation'])
            assert 'result' in stopped, stopped
            peer.review_release.set()  # a late APPROVE cannot pass the owner check
            await async_wait_until(lambda: terminal('late') != [('started',)], timeout=45, interval=.05,
                                   what='stopped managed review settlement')
            assert not peer.late_target.exists()

    try:
        with daemon(root, home, env, barrier=False) as (_owner, desc):
            asyncio.run(exercise(desc))
    finally:
        peer.review_release.set()
        peer.shutdown()
        peer.server_close()
    assert len(peer.reviews) >= 2
    assert 'Edit only allowed.txt' in peer.reviews[0]
    assert any('Review only review-only.txt' in prompt for prompt in peer.reviews)
