"""Journey: restart while one session waits on an approval and another on a clarification.

One ordinary daemon (real authority, SQLite, in-process agent loop with the real terminal and clarify
tools, loopback model). Each session has a turn parked on its prompt and durable followers queued
behind it when the owner goes away. After a fresh daemon starts:
- the stale prompts are gone from every snapshot and answering their old ids is refused, before and
  after the FIFO moves on, and never runs the gated command;
- a crash leaves the parked turn ``unknown`` and holds its followers until /discard (no replay);
  a graceful stop settles it ``interrupted`` and the followers resume on their own;
- every input reaches the model exactly once and only as itself; the gated command never runs.
"""
import asyncio
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shlex
import signal
import sqlite3
import threading

import pytest

from tests.gateway.fixtures.local_recovery_probe import child_env, daemon, rpc, websocket

FOLLOWERS = {'approval': ('APPROVAL_FOLLOWER_1', 'APPROVAL_FOLLOWER_2'),
             'clarify': ('CLARIFY_FOLLOWER_1', 'CLARIFY_FOLLOWER_2')}
ASK = {'approval': 'ASK_APPROVAL', 'clarify': 'ASK_CLARIFY'}


class PromptModel(BaseHTTPRequestHandler):
    """ASK_* turns call the gated tool; every other user turn is answered directly."""

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages = body.get('messages', [])
        last = messages[-1] if messages else {}
        text = next((m.get('content', '') for m in reversed(messages) if m['role'] == 'user'), '')
        if messages:
            self.server.requests.append({'text': str(text), 'after_tool': last.get('role') == 'tool',
                                         'users': [m.get('content') for m in messages if m['role'] == 'user']})
        message = {'role': 'assistant', 'content': 'ACK_' + str(text)}
        if last.get('role') == 'user' and text in ASK.values():
            name, args = ('terminal', {'command': self.server.command}) if text == ASK['approval'] else (
                'clarify', {'questions': [{'question': 'Pick a color', 'choices': ['BLUE', 'GREEN']}]})
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{
                'index': 0, 'id': 'call_' + name, 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(args)}}]}
        finish = 'tool_calls' if 'tool_calls' in message else 'stop'
        frame = {'id': 'local', 'model': 'prompt-model', 'choices': [{'index': 0, 'delta': message, 'finish_reason': finish}],
                 'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}
        if body.get('stream'):
            payload, kind = ('data: ' + json.dumps(frame) + '\n\ndata: [DONE]\n\n').encode(), 'text/event-stream'
        else:
            frame['choices'][0]['message'] = frame['choices'][0].pop('delta')
            payload, kind = json.dumps(frame).encode(), 'application/json'
        try:
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass


@pytest.mark.platforms('linux')
@pytest.mark.parametrize('restart', ['crash', 'graceful'])
def test_restart_fences_open_prompts_and_never_reruns_work(tmp_path, restart):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    target = home / 'gated-removal'
    target.mkdir()
    (target / 'owned.txt').write_text('must survive')
    peer = ThreadingHTTPServer(('127.0.0.1', 0), PromptModel)
    peer.requests = []
    peer.command = 'rm -rf -- ' + shlex.quote(str(target))
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False},
        'model': {'provider': 'custom', 'default': 'prompt-model', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False}}, 'terminal': {'cwd': str(home)},
        'approvals': {'mode': 'manual', 'timeout': 120}, 'agent': {'clarify_timeout': 120},
        'platform_toolsets': {'cli': ['terminal', 'clarify']}}))
    env = child_env() | {k: os.environ[k] for k in ('TIRITH_ENABLED',) if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, PYTHONUNBUFFERED='1')

    def admissions():
        with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
            db.row_factory = sqlite3.Row
            return {r['request_id']: dict(r) for r in db.execute(
                'SELECT admission_id,request_id,target_session_id,status,outcome,seq FROM session_admissions')}

    def reached(text):
        return sum(r['text'] == text and not r['after_tool'] for r in peer.requests)

    async def call(ws, method, **params):
        reply = await rpc(ws, method, **params)
        assert 'result' in reply, reply
        return reply['result']

    async def until(predicate, timeout=40):
        async with asyncio.timeout(timeout):
            while not predicate():
                await asyncio.sleep(.03)

    async def answer(ws, sid, prompt):
        kind = prompt['kind']
        reply = await rpc(ws, kind + '.respond', session_id=sid, prompt_id=prompt['prompt_id'],
                          execution_generation=prompt['execution_generation'],
                          **({'choice': 'once'} if kind == 'approval' else {'answer': 'BLUE'}))
        return reply.get('error', {}).get('data', {}).get('reason') or reply.get('error', {}).get('message') or reply['result']

    before = {'sessions': {}, 'prompts': {}}

    async def first(desc, owner):
        async with websocket(home, desc) as ws:
            for kind in ('approval', 'clarify'):
                sid = (await call(ws, 'session.create', source='cli', request_id='journey-' + kind, cwd=str(home),
                                  model='prompt-model', provider='custom', base_url=url,
                                  toolsets=['terminal', 'clarify']))['session_id']
                before['sessions'][kind] = sid
                await call(ws, 'prompt.submit', session_id=sid, input_id=ASK[kind], text=ASK[kind])
                prompt = None
                async with asyncio.timeout(30):
                    while prompt is None:
                        snapshot = await call(ws, 'session.resume', session_id=sid)
                        prompt = next((p for p in snapshot['prompts'] if p['kind'] == kind), None)
                        await asyncio.sleep(.03)
                before['prompts'][kind] = prompt
                for name in FOLLOWERS[kind]:
                    await call(ws, 'prompt.submit', session_id=sid, input_id=name, text=name)
            rows = admissions()
            assert {k: rows[k]['status'] for k in rows} == {
                ASK['approval']: 'started', ASK['clarify']: 'started',
                **{name: 'queued' for names in FOLLOWERS.values() for name in names}}, rows
            assert target.exists()
            if restart == 'graceful':
                owner.send_signal(signal.SIGTERM)
            else:
                owner.kill()
            await asyncio.to_thread(owner.wait, 60)
        before['stopped'] = {k: (r['status'], r['outcome']) for k, r in admissions().items()}

    async def second(desc):
        async with websocket(home, desc) as ws:
            fenced = {}
            for kind, sid in before['sessions'].items():
                snapshot = await call(ws, 'session.resume', session_id=sid)
                assert snapshot['prompts'] == [], snapshot['prompts']
                fenced[kind] = await answer(ws, sid, before['prompts'][kind])
                if restart == 'crash':
                    lost = [r for r in snapshot['pending'] if r['status'] == 'unknown']
                    assert [r['input_id'] for r in lost] == [ASK[kind]], snapshot['pending']
                    queued = [r['input_id'] for r in snapshot['pending'] if r['status'] == 'queued']
                    assert queued == list(FOLLOWERS[kind]), snapshot['pending']
            assert fenced == {'approval': 'stale_generation', 'clarify': 'stale_generation'}, fenced
            if restart == 'crash':
                # Unknown work holds the FIFO: nothing ran behind it before the operator decides.
                assert not any(reached(name) for names in FOLLOWERS.values() for name in names)
                for kind, sid in before['sessions'].items():
                    snapshot = await call(ws, 'session.resume', session_id=sid)
                    lost = next(r for r in snapshot['pending'] if r['status'] == 'unknown')
                    await call(ws, 'prompt.resolve_unknown', session_id=sid, admission_id=lost['admission_id'],
                               execution_generation=lost['execution_generation'])
            await until(lambda: all(r['status'] == 'terminal' for r in admissions().values()))
            late = {kind: await answer(ws, sid, before['prompts'][kind]) for kind, sid in before['sessions'].items()}
            assert late == {'approval': 'stale_generation', 'clarify': 'stale_generation'}, late
            return {kind: await call(ws, 'session.resume', session_id=sid) for kind, sid in before['sessions'].items()}

    try:
        with daemon(root, home, env, barrier=False) as (owner, desc):
            asyncio.run(first(desc, owner))
        if restart == 'graceful':
            # A planned stop cancels the parked turns (a decision, not uncertainty); no follower ran.
            assert {before['stopped'][ASK[k]] for k in ASK} == {('terminal', 'interrupted')}, before['stopped']
        else:
            assert {before['stopped'][ASK[k]] for k in ASK} == {('started', None)}, before['stopped']
        assert not any(reached(name) for names in FOLLOWERS.values() for name in names)
        with daemon(root, home, env, barrier=False) as (_, desc):
            final = asyncio.run(second(desc))
        assert target.exists() and (target / 'owned.txt').read_text() == 'must survive'
        for kind, names in FOLLOWERS.items():
            # Exactly once, and only as itself: a lost/cancelled turn never rides a follower's user
            # turn (an unclosed tail would be merged into it and re-sent as part of that request).
            assert reached(ASK[kind]) == 1, [r['text'] for r in peer.requests]
            first_follower = next(r for r in peer.requests if r['text'] == names[0])
            assert first_follower['users'] == [ASK[kind], names[0]], first_follower['users']
            assert not any(r['after_tool'] and r['text'] == ASK[kind] for r in peer.requests), 'gated tool resumed'
            order = [r['text'] for r in peer.requests if r['text'] in names]
            assert order == list(names), order
            history = json.dumps(final[kind]['messages'])
            assert all('ACK_' + name in history for name in names), history[-2000:]
            assert final[kind]['prompts'] == []
        settled = admissions()
        assert all(settled[name]['outcome'] == 'completed' for names in FOLLOWERS.values() for name in names), settled
        print(json.dumps({'restart': restart, 'stopped': before['stopped'], 'settled': settled,
                          'requests': [r['text'] for r in peer.requests]}))
    finally:
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)
