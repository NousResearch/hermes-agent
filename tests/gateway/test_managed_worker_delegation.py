"""delegate_task inside a managed worker (#141793): real daemon, real worker, loopback model.

Every child is reserved by the owner under the worker's execution, persists through its own
claim, returns its result to the parent's turn, and dies with the owner on a restart."""
import asyncio
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sqlite3
import threading

import pytest

from tests.gateway.fixtures.local_recovery_probe import daemon, rpc, websocket
from tests.waiting import async_wait_until

ROOT = Path(__file__).resolve().parents[2]


class Model(BaseHTTPRequestHandler):
    """The parent delegates once; the child answers (or blocks on BLOCK); the parent then finishes."""
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages = body.get('messages') or [{'role': 'system', 'content': ''}]
        users = [m.get('content') for m in messages if m['role'] == 'user']
        text = json.dumps(messages)
        name = None
        if users and users[0] in ('DELEGATE', 'DELEGATE_BLOCK') and messages[-1]['role'] == 'user':
            goal = 'CHILD_GOAL_BLOCK' if users[0] == 'DELEGATE_BLOCK' else 'CHILD_GOAL'
            name, args = 'delegate_task', {'goal': goal}
        elif 'CHILD_GOAL' in text and not any(u in ('DELEGATE', 'DELEGATE_BLOCK') for u in users):
            if 'CHILD_GOAL_BLOCK' in text:
                self.server.blocked.set()
                self.server.release.wait(60)
            message = {'role': 'assistant', 'content': 'CHILD_RESULT_42'}
        else:
            message = {'role': 'assistant', 'content': 'PARENT_DONE'}
        if name:
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'd1', 'index': 0, 'type': 'function',
                       'function': {'name': name, 'arguments': json.dumps(args)}}]}
        self.server.seen.append((users[:1], message.get('content'), [t['function']['name'] for t in body.get('tools') or []]))
        choice = {'index': 0, 'delta': message, 'finish_reason': 'tool_calls' if name else 'stop'}
        frame = {'id': 'm', 'model': 'm', 'choices': [choice], 'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5}}
        if body.get('stream'):
            payload, kind = ('data: ' + json.dumps(frame) + '\n\ndata: [DONE]\n\n').encode(), 'text/event-stream'
        else:
            choice['message'] = choice.pop('delta')
            payload, kind = json.dumps(frame).encode(), 'application/json'
        try:
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass


def _setup(tmp_path):
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.seen, peer.blocked, peer.release = [], threading.Event(), threading.Event()
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False, 'managed_workers': True},
        'model': {'provider': 'custom', 'default': 'm', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False}},
        'platform_toolsets': {'cli': ['delegation', 'todo']}}))
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ') if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(ROOT),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url)
    return home, peer, url, env


def _rows(home, sql, *args):
    with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
        return db.execute(sql, args).fetchall()


async def _submit(home, desc, url, text):
    async with websocket(home, desc) as ws:
        created = await rpc(ws, 'session.create', request_id='parent', source='cli', cwd=str(home), model='m',
                            provider='custom', base_url=url, api_key='loopback-only',
                            toolsets=['delegation', 'todo'], ignore_rules=True)
        sid = created['result']['session_id']
        assert 'result' in await rpc(ws, 'prompt.submit', session_id=sid, input_id='run', text=text)
        return sid


@pytest.mark.platforms("linux")
def test_managed_worker_delegates_through_an_owner_reserved_child(tmp_path):
    home, peer, url, env = _setup(tmp_path)

    async def exercise(desc):
        sid = await _submit(home, desc, url, 'DELEGATE')
        await async_wait_until(lambda: _rows(home, "SELECT status FROM session_admissions WHERE request_id='run'") == [
            ('terminal',)], timeout=120, interval=.1, what='run terminal')
        async with websocket(home, desc) as ws:
            return sid, (await rpc(ws, 'session.resume', session_id=sid))['result']['messages']
    try:
        with daemon(ROOT, home, env, barrier=False) as (_owner, desc):
            sid, messages = asyncio.run(exercise(desc))
    finally:
        peer.release.set()
        peer.shutdown()
        peer.server_close()
    assert any('delegate_task' in tools for _, _, tools in peer.seen), peer.seen
    tool = [m['content'] for m in messages if m['role'] == 'tool'][-1]
    assert 'CHILD_RESULT_42' in tool, tool
    assert messages[-1]['content'] == 'PARENT_DONE', messages
    children = _rows(home, "SELECT w.status, s.source, s.parent_session_id, s.end_reason, s.input_tokens FROM worker_executions w "
                     "JOIN sessions s ON s.id=w.session_id WHERE w.kind='child'")
    assert children == [('terminal', 'subagent', sid, 'agent_close', 3)], children
    child_sid = _rows(home, "SELECT session_id FROM worker_executions WHERE kind='child'")[0][0]
    assert ('assistant', 'CHILD_RESULT_42') in _rows(home, 'SELECT role, content FROM messages WHERE session_id=?', child_sid)


@pytest.mark.platforms("linux")
def test_owner_restart_mid_child_retires_the_child_and_never_replays_it(tmp_path):
    home, peer, url, env = _setup(tmp_path)

    async def first(desc):
        await _submit(home, desc, url, 'DELEGATE_BLOCK')
        assert await asyncio.to_thread(peer.blocked.wait, 90), peer.seen
        assert _rows(home, "SELECT status FROM worker_executions WHERE kind='child'") == [('running',)]
    try:
        with daemon(ROOT, home, env, barrier=False) as (owner, desc):
            asyncio.run(first(desc))
            owner.kill()
            owner.wait(timeout=10)
        peer.release.set()
        with daemon(ROOT, home, env, barrier=False):
            states = dict(_rows(home, 'SELECT kind, status FROM worker_executions'))
    finally:
        peer.release.set()
        peer.shutdown()
        peer.server_close()
    assert states == {'compute': 'unknown', 'child': 'terminal'}, states
    assert _rows(home, "SELECT s.end_reason FROM worker_executions w JOIN sessions s ON s.id=w.session_id "
                       "WHERE w.kind='child'") == [('agent_close',)]
    # The lost turn is not replayed: the parent never got a child result to finish on.
    assert not [s for s in peer.seen if s[0] == ['DELEGATE_BLOCK'] and s[1] == 'PARENT_DONE'], peer.seen


@pytest.mark.platforms("linux")
def test_relay_driven_worker_builds_and_runs_its_child_off_the_relay_loop(tmp_path):
    """NeMo Relay drives tool callbacks on its own event loop: the child's construction reads the
    store (bind_session_state) and its close finishes the execution, and the worker RPC refuses a
    synchronous call on a running loop, so every child store call must hop off it."""
    relay = pytest.importorskip('nemo_relay')
    if getattr(relay, '_native', None) is None:
        pytest.skip('NeMo Relay native binding unavailable')
    home, peer, url, env = _setup(tmp_path)
    plugins = home / 'plugins.toml'
    plugins.write_text('version = 1\n[[components]]\nkind = "observability"\nenabled = true\n'
                       '[components.config]\nversion = 4\n[components.config.atof]\nenabled = true\n'
                       '[[components.config.atof.sinks]]\ntype = "file"\noutput_directory = '
                       + json.dumps(str(home / 'relay-events')) + '\nfilename = "events.jsonl"\nmode = "overwrite"\n')
    env['HERMES_NEMO_RELAY_PLUGINS_TOML'] = str(plugins)

    async def exercise(desc):
        sid = await _submit(home, desc, url, 'DELEGATE')
        await async_wait_until(lambda: _rows(home, "SELECT status FROM session_admissions WHERE request_id='run'") == [
            ('terminal',)], timeout=120, interval=.1, what='run terminal')
        return sid
    try:
        with daemon(ROOT, home, env, barrier=False) as (_owner, desc):
            sid = asyncio.run(exercise(desc))
    finally:
        peer.shutdown()
        peer.server_close()
    assert (home / 'relay-events' / 'events.jsonl').exists(), 'Relay never drove this turn'
    tool = [c for r, c in _rows(home, 'SELECT role, content FROM messages WHERE session_id=?', sid) if r == 'tool']
    assert tool and 'CHILD_RESULT_42' in tool[-1], tool
    assert _rows(home, "SELECT w.status, s.end_reason FROM worker_executions w JOIN sessions s ON s.id=w.session_id "
                       "WHERE w.kind='child'") == [('terminal', 'agent_close')]
