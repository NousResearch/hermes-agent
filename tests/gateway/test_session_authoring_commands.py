"""Authoring/workflow slash commands run the messaging gateway's own handler for a LOCAL session.

``slash.exec`` on a real daemon: /plan /learn /init /moa /blueprint answer with
the turn the messaging gateway would have rewritten (a ``send`` directive the client submits with
its own input id; nothing is admitted by the gateway), /goal sets the session's goal row and
returns its kickoff the same way, /subgoal /loop /heartbeat write the session's own row, /review
/refine answer from the session's cached agent, /curator /egress print their report.
"""
import asyncio
import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import signal
import sqlite3
import threading
import time

import pytest

from tests.gateway.fixtures.local_recovery_probe import Model, child_env, daemon, rpc, websocket


@pytest.mark.platforms("linux")
def test_local_session_authoring_commands_run_the_gateway_handler(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user, work = tmp_path / 'state', tmp_path / 'user', tmp_path / 'work'
    home.mkdir(mode=0o700)
    user.mkdir()
    work.mkdir()
    (work / 'AGENTS.md').write_text('# existing instructions\n')
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests = []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    config = json.dumps({'gateway': {'multiplex_profiles': False},
                         'model': {'provider': 'custom', 'default': 'authoring-model', 'base_url': url},
                         'auxiliary': {'title_generation': {'enabled': False}}})
    (home / 'config.yaml').write_text(config)
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, PYTHONUNBUFFERED='1')
    out, sid = {}, {}

    def admissions():
        with sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True) as db:
            return [r[0] for r in db.execute('SELECT request_id FROM session_admissions')]

    def meta(key):
        with sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True) as db:
            row = db.execute('SELECT value FROM state_meta WHERE key = ?', (key,)).fetchone()
            return json.loads(row[0]) if row else None

    async def probe(desc):
        async with websocket(home, desc) as ws:
            created = await rpc(ws, 'session.create', request_id='authoring', source='cli', toolsets=[],
                                cwd=str(work))
            sid['id'] = created['result']['session_id']
            info = (await rpc(ws, 'session.info', session_id=sid['id']))['result']
            sid['row'] = info.get('stored_session_id') or sid['id']
            # Before any turn: /review and /refine have no cached agent yet.
            for command in ('review', 'refine'):
                out['cold ' + command] = await exec_(ws, command)
            await rpc(ws, 'prompt.submit', session_id=sid['id'], input_id='warm', text='WARM_AUTHORING')
            deadline = time.monotonic() + 45
            while True:
                snap = (await rpc(ws, 'session.resume', session_id=sid['id']))['result']
                if not snap['running'] and not snap['pending'] and 'RECOVERY_ACK_WARM_AUTHORING' in json.dumps(
                        snap['messages']):
                    break
                assert time.monotonic() < deadline, ('warm turn never settled', snap)
                await asyncio.sleep(.1)
            out['admitted before'] = admissions()
            # Before /moa: its one-shot model swap evicts the cached agent (messaging behaviour).
            out['refine focus on naming'] = await exec_(ws, 'refine focus on naming')
            for command in ('plan add a cache', 'learn', 'init', 'moa', 'moa compare',
                            'blueprint', 'egress', 'curator', 'goal ship the cache', 'subgoal add tests',
                            'goal status', 'loop every 30m check the queue', 'loop', 'heartbeat every 10m ping',
                            'heartbeat', 'goal clear', 'loop clear', 'heartbeat clear'):
                out[command] = await exec_(ws, command)
            out['admitted after'] = admissions()

    async def exec_(ws, command):
        try:
            reply = await rpc(ws, 'slash.exec', session_id=sid['id'], command=command)
        except TimeoutError:
            raise AssertionError(f'slash.exec {command!r} did not answer within the rpc bound') from None
        return reply.get('result') or reply.get('error')

    try:
        with daemon(root, home, env, barrier=False) as (proc, desc):
            asyncio.run(probe(desc))
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=20)
    finally:
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)

    def send(command):
        reply = out[command]
        assert reply.get('type') == 'send' and reply.get('message'), (command, reply)
        return reply

    # Prompt builders: the messaging gateway's rewritten turn, returned for the client to submit.
    plan = send('plan add a cache')
    assert 'add a cache' in plan['message'] and plan['display'] == '/plan add a cache', plan
    assert 'add a cache' in plan.get('notice', ''), plan
    assert send('learn')['notice'], out['learn']
    init = send('init')
    assert str(work) in init['message'] and 'UPDATE the existing AGENTS.md' in init['message'], init
    assert send('moa compare')['message'] == 'compare', out['moa compare']
    assert out['moa']['type'] == 'exec' and '/moa' in out['moa']['output'], out['moa']
    assert out['blueprint']['type'] == 'exec' and out['blueprint']['output'], out['blueprint']
    assert out['egress']['type'] == 'exec' and out['egress']['output'], out['egress']
    assert out['curator']['type'] == 'exec' and 'curator' in out['curator']['output'].lower(), out['curator']
    # /goal: the session's own goal row, and its kickoff returned (never self-admitted).
    goal = send('goal ship the cache')
    assert 'ship the cache' in goal['message'] and 'ship the cache' in goal['notice'], goal
    assert 'add tests' in out['subgoal add tests']['output'], out['subgoal add tests']
    assert 'ship the cache' in out['goal status']['output'], out['goal status']
    # Nothing the commands returned was admitted by the gateway; /loop's first wake (fires now) is the
    # loop scheduler's own producer admission, not a command's.
    added = [r for r in out['admitted after'] if r not in out['admitted before'] and not r.startswith('["loop-tick"')]
    assert added == [], added
    # /loop, /heartbeat: state on the session's row.
    assert '30m' in out['loop every 30m check the queue']['output'], out['loop every 30m check the queue']
    assert 'check the queue' in out['loop']['output'], out['loop']
    assert 'ping' in out['heartbeat every 10m ping']['output'], out['heartbeat every 10m ping']
    assert 'ping' in out['heartbeat']['output'], out['heartbeat']
    # /review and /refine: the session's cached agent (none before the first turn).
    for command in ('review', 'refine'):
        assert out['cold ' + command]['type'] == 'exec', out['cold ' + command]
    assert out['refine focus on naming']['type'] == 'exec', out['refine focus on naming']
    assert 'naming' in out['refine focus on naming']['output'], out['refine focus on naming']
    for command in ('goal clear', 'loop clear', 'heartbeat clear'):
        assert out[command]['type'] == 'exec' and out[command]['output'], (command, out[command])
    assert (home / 'config.yaml').read_text() == config


class _JudgeModel(Model):
    """Echo model whose goal-judge calls answer ``continue`` once, then ``done``."""

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        text = json.dumps(body.get('messages', []))
        if 'goal' in text.lower() and 'judge' in text.lower():
            self.server.verdicts += 1
            content = json.dumps({'done': self.server.verdicts > 1, 'reason': 'probe verdict'})
        else:
            content = 'ACK'
        message = {'role': 'assistant', 'content': content}
        if body.get('stream'):
            payload = ('data: ' + json.dumps({'id': 'j', 'choices': [
                {'index': 0, 'delta': message, 'finish_reason': 'stop'}]}) + '\n\ndata: [DONE]\n\n').encode()
            kind = 'text/event-stream'
        else:
            payload = json.dumps({'id': 'j', 'choices': [{'index': 0, 'message': message, 'finish_reason': 'stop'}],
                                  'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}).encode()
            kind = 'application/json'
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.mark.platforms("linux")
def test_local_goal_kickoff_is_judged_and_continued_by_the_gateway(tmp_path):
    """The client submits /goal's kickoff; the gateway's post-turn judge then admits the continuation
    into the session's own FIFO (the local adapter has no FIFO reader) until the judge says done."""
    root = Path(__file__).resolve().parents[2]
    home, user, work = tmp_path / 'state', tmp_path / 'user', tmp_path / 'work'
    home.mkdir(mode=0o700)
    user.mkdir()
    work.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), _JudgeModel)
    peer.verdicts = 0
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False}, 'model': {'provider': 'custom', 'default': 'm', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False}}}))
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, PYTHONUNBUFFERED='1')
    out = {}

    async def probe(desc):
        async with websocket(home, desc) as ws:
            sid = (await rpc(ws, 'session.create', request_id='goal', source='tui', toolsets=[],
                             cwd=str(work)))['result']['session_id']
            kickoff = (await rpc(ws, 'slash.exec', session_id=sid, command='goal write the probe'))['result']
            await rpc(ws, 'prompt.submit', session_id=sid, input_id='kick', text=kickoff['message'])
            deadline = time.monotonic() + 60
            while True:
                status = (await rpc(ws, 'slash.exec', session_id=sid, command='goal status'))['result']['output']
                if status.startswith('✓'):
                    break
                assert time.monotonic() < deadline, status
                await asyncio.sleep(.2)
            out['status'] = status
            out['messages'] = (await rpc(ws, 'session.resume', session_id=sid))['result']['messages']
    try:
        with daemon(root, home, env, barrier=False) as (proc, desc):
            asyncio.run(probe(desc))
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=20)
    finally:
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)
    assert out['status'].startswith('✓ Goal done (2/20 turns)'), out['status']
    users = [m['content'] for m in out['messages'] if m.get('role') == 'user']
    assert users[0] == 'write the probe' and 'standing goal' in users[1], users
    with sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True) as db:
        rows = db.execute('SELECT request_id, principal_id FROM session_admissions').fetchall()
    kick = [r for r in rows if r[0] == 'kick']
    continuation = [r for r in rows if r[0] != 'kick']
    assert len(kick) == 1 and len(continuation) == 1, rows
    assert continuation[0][1].startswith('automation:') and 'goal-continuation' in continuation[0][0], rows
