"""A process a subagent hands to its parent completes into a PARENT turn, in the parent's context.

The gateway arms a completion watcher on its loop from the tool thread that started the process.
Armed from a delegate child's thread, the watcher used to run in a copy of the child's context, and so
did every turn its completion admitted: after ``process_manage(action='handoff')`` the parent's next
turn ran as a delegated child, and its own ``terminal(background=true, notify_on_complete=true)`` came
back ``notify_on_complete: false`` with the "You are a subagent" note (heartbeats unarmed, sudo
refused, the Kanban board fenced).

A watcher therefore starts in an empty context and binds the OWNING profile's scope for its whole
run: the output it prepares goes through that profile's ``transform_terminal_output`` hooks, not the
launch profile's.

Real daemon (``python -m gateway.run``) serving two profiles, each with its own output-hook plugin; a
local session on the SECONDARY profile over the websocket, scripted loopback model: the parent
delegates; the child starts a notify watcher and hands it off; its completion starts a parent turn
that starts a second notify process, whose completion must re-enter the session.
"""
import asyncio
import contextlib
import json
import re
import shlex
import signal
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from websockets.asyncio.client import connect

from tests.gateway.fixtures.local_recovery_probe import child_env, control, daemon, rpc
from tests.waiting import async_wait_until

CHILD_GOAL = 'CHILD_GOAL start a watcher and hand it to the parent'


def _call(name, **arguments):
    return {'id': f'call_{name}_{abs(hash(json.dumps(arguments))) % 10**8}', 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(arguments)}}


class ScriptedModel(BaseHTTPRequestHandler):
    """Parent: delegate, then answer the handed-off completion with a second notify process. Child: start
    a notify process, hand it off, finish."""

    def log_message(self, *args):
        pass

    def _reply(self, messages):
        state = self.server
        last = messages[-1]
        first_user = next((str(m.get('content') or '') for m in messages if m['role'] == 'user'), '')
        if CHILD_GOAL in first_user:
            called = [c['function']['name'] for m in messages if m['role'] == 'assistant'
                      for c in (m.get('tool_calls') or [])]
            if 'terminal' not in called:
                # Held open until the handoff has landed: a process that exits first cannot be handed off.
                return _call('terminal', command=f'while [ ! -e {shlex.quote(str(state.release))} ]; do sleep 0.2; done; echo FIRST_DONE',
                             background=True, notify_on_complete=True)
            if 'tool_call' not in called:  # process_manage is a deferred tool for the child
                started = next(str(m['content']) for m in messages if m['role'] == 'tool')
                state.first_proc = re.search(r'proc_[0-9a-f]+', started).group(0)
                return _call('tool_call', calls=[{'name': 'process_manage', 'arguments': {
                    'action': 'handoff', 'session_id': state.first_proc,
                    'data': 'watcher for the parent; report its result'}}])
            if last['role'] == 'tool' and not state.release.exists():
                handoff = str(last.get('content') or '')
                if re.search(r'"error":\s*"', handoff):
                    state.handoff_error = handoff
                state.release.touch()
            return 'CHILD_FINISHED'
        text = str(last.get('content') or '')
        if last['role'] == 'user' and 'START' in text and not state.delegated:
            state.delegated = True
            return _call('delegate_task', goal=CHILD_GOAL)
        if (last['role'] == 'user' and state.first_proc and f'Background process {state.first_proc}' in text
                and not state.second_started):
            state.second_started = True
            return _call('terminal', command='sleep 2; echo SECOND_DONE fence=${HERMES_DELEGATED_CHILD_CONTEXT:-none}', background=True, notify_on_complete=True)
        return 'PARENT_ACK'

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages = body.get('messages') or []
        with self.server.lock:
            if messages:
                self.server.requests.append(body)
            reply = self._reply(messages) if messages else 'PARENT_ACK'
        if isinstance(reply, dict):
            message = {'role': 'assistant', 'content': None, 'tool_calls': [reply]}
            finish = 'tool_calls'
            delta = {'role': 'assistant', 'content': None, 'tool_calls': [dict(reply, index=0)]}
        else:
            message = delta = {'role': 'assistant', 'content': reply}
            finish = 'stop'
        usage = {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}
        if body.get('stream'):
            chunks = [{'id': 'scripted', 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]},
                      {'id': 'scripted', 'choices': [], 'usage': usage}]
            payload = ''.join('data: ' + json.dumps(c) + '\n\n' for c in chunks) + 'data: [DONE]\n\n'
            payload, kind = payload.encode(), 'text/event-stream'
        else:
            payload = json.dumps({'id': 'scripted', 'choices': [{'index': 0, 'message': message, 'finish_reason': finish}],
                                  'usage': usage}).encode()
            kind = 'application/json'
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)


def _tool_results(home, sid):
    with contextlib.closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
        return [row[0] for row in db.execute(
            "SELECT content FROM messages WHERE session_id=? AND role='tool' ORDER BY id", (sid,))]


def _user_texts(home, sid):
    with contextlib.closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
        return [str(row[0] or '') for row in db.execute(
            "SELECT content FROM messages WHERE session_id=? AND role='user' ORDER BY id", (sid,))]


def _profile_home(target, marker, config):
    """A profile home whose output-hook plugin stamps ``HOOK_<MARKER>`` on every terminal output."""
    plugin = target / 'plugins' / 'output_mark'
    plugin.mkdir(parents=True)
    (plugin / 'plugin.yaml').write_text('name: output_mark\n')
    (plugin / '__init__.py').write_text(
        'def register(ctx):\n'
        f'    ctx.register_hook("transform_terminal_output", lambda **kw: "HOOK_{marker}\\n" + kw["output"])\n')
    (target / 'config.yaml').write_text(json.dumps(config))
    (target / '.env').write_text('OPENROUTER_API_KEY=loopback-only\n')


@pytest.mark.platforms("linux")  # the daemon is stopped with SIGINT, which raises on Windows
def test_handed_off_completion_turn_runs_as_the_parent(tmp_path):
    root = Path(__file__).resolve().parents[2]
    # A custom root (``<root>/profiles/<name>``, as in Docker): ``$HOME/.hermes`` is the live system.
    user, work, home = tmp_path / 'user', tmp_path / 'work', tmp_path / 'state'
    secondary = home / 'profiles' / 'second'
    secondary.mkdir(parents=True, mode=0o700)
    user.mkdir()
    work.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), ScriptedModel)
    peer.requests, peer.lock = [], threading.Lock()
    peer.first_proc, peer.delegated, peer.second_started = '', False, False
    peer.release, peer.handoff_error = tmp_path / 'release first', ''
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    config = {
        'gateway': {'multiplex_profiles': True},
        'model': {'provider': 'openrouter', 'default': 'openai/gpt-5', 'base_url': url,
                  'api_key': 'loopback-only', 'api_mode': 'chat_completions'},
        'auxiliary': {'title_generation': {'enabled': False}},
        'display': {'background_process_notifications': 'off'},
        'plugins': {'enabled': ['output_mark']},
    }
    _profile_home(home, 'LAUNCH', config)
    _profile_home(secondary, 'SECOND', config)
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               XDG_CONFIG_HOME=str(user / '.config'), HERMES_GATEWAY_LOCK_DIR=str(tmp_path / 'locks'),
               OPENROUTER_API_KEY='loopback-only', OPENAI_API_KEY='loopback-only', PYTHONUNBUFFERED='1')
    out = {}

    def second_result():
        assert not peer.handoff_error, ('the handoff was refused', peer.handoff_error)
        for content in _tool_results(secondary, out['sid']):
            with contextlib.suppress(ValueError, TypeError):
                data = json.loads(content)
                if isinstance(data, dict) and data.get('session_id') and data['session_id'] != peer.first_proc:
                    return data
        return None

    async def probe(desc):
        grant = control(home, 'session-ticket', {'profile_id': str(secondary), 'instance_id': desc['instance_id'],
                                                 'purpose': 'interactive'})
        async with connect(desc['api_origin'].replace('http:', 'ws:') + '/api/ws',
                           subprotocols=['hermes-gateway-v1', 'hermes-gateway-ticket.' + grant['ticket']]) as ws:
            created = await rpc(ws, 'session.create', request_id='handoff-parent', source='cli',
                                toolsets=['terminal', 'delegation'], cwd=str(work))
            out['sid'] = sid = created['result']['session_id']
            await rpc(ws, 'prompt.submit', session_id=sid, input_id='start', text='START')
            await async_wait_until(second_result, timeout=90, what='the parent starting a process in the handoff turn')
            out['second'] = second = second_result()
            # The second process is the parent's own: its completion re-enters the session.
            await async_wait_until(lambda: any(second['session_id'] in t for t in _user_texts(secondary, sid)),
                                   timeout=60, what="the parent's own completion re-entering the session")

    try:
        with daemon(root, home, env, barrier=False) as (proc, desc):
            try:
                asyncio.run(probe(desc))
            finally:
                peer.release.touch()
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=30)
    finally:
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)

    users = _user_texts(secondary, out['sid'])
    assert any(f'Background process {peer.first_proc} completed' in t for t in users), users
    second = out['second']
    assert second.get('notify_on_complete') is True and 'subagent_note' not in second, second
    second_notice = next(t for t in users if f"Background process {second['session_id']}" in t)
    # Commands the parent runs in that turn are its own, not stamped with the child's Kanban fence.
    assert 'SECOND_DONE fence=none' in second_notice, second_notice
    # Its output went through the secondary profile's hooks, not the launch profile's.
    assert 'HOOK_SECOND' in second_notice and 'HOOK_LAUNCH' not in second_notice, second_notice
