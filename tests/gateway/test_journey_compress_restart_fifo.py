"""Journey: queued followers -> automatic compression -> gateway restart before drain.

One ordinary daemon (real authority, real SQLite, in-process agent loop, loopback model). A turn's
own preflight compression rotates the transcript while three followers wait behind it; the owner is
stopped before the followers run. After a fresh daemon starts: every admission keeps its identity,
the followers run once each in FIFO order on the compressed transcript, the turn that compressed
is never executed twice, and the compression lineage is the one committed before the restart.
"""
import asyncio
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import signal
import sqlite3
import threading

import pytest

from tests.gateway.fixtures.local_recovery_probe import child_env, daemon, rpc, websocket

FOLLOWERS = ('FOLLOWER_1', 'FOLLOWER_2', 'FOLLOWER_3')


class CompactingModel(BaseHTTPRequestHandler):
    """Summarizes on the `summary` model; the first post-summary turn request parks until released."""

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages = body.get('messages', [])
        wire = json.dumps(messages)
        text = next((m.get('content', '') for m in reversed(messages) if m['role'] == 'user'), '')
        if messages:
            self.server.requests.append({'model': body.get('model'), 'text': str(text), 'wire': wire})
        if body.get('model') == 'original' and len(wire) > 50000:
            payload = json.dumps({'error': {'message': "This model's maximum context length is 64000 tokens. "
                                                       'However, your messages resulted in 70000 tokens.',
                                            'type': 'invalid_request_error', 'code': 'context_length_exceeded'}}).encode()
            return self._send(400, payload, 'application/json')
        if body.get('model') == 'original' and 'SUMMARY_RETAINED_FACTS' in wire and not self.server.blocked.is_set():
            self.server.blocked.set()
            self.server.release.wait(60)
        # Only the pressure turns are large: the followers must not trigger a second compaction.
        reply = 'SUMMARY_RETAINED_FACTS' if body.get('model') == 'summary' else (
            'ACK_' + str(text)[:40] + (' ' + 'historical detail ' * 700 if str(text).startswith('PRESSURE_') else ''))
        message = {'role': 'assistant', 'content': reply}
        payload = json.dumps({'id': 'local', 'model': body.get('model'), 'choices': [
            {'index': 0, 'message': message, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}).encode()
        kind = 'application/json'
        if body.get('stream'):
            payload = ('data: ' + json.dumps({'id': 'local', 'choices': [
                {'index': 0, 'delta': message, 'finish_reason': 'stop'}]}) + '\n\ndata: [DONE]\n\n').encode()
            kind = 'text/event-stream'
        self._send(200, payload, kind)

    def _send(self, status, payload, kind):
        try:
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass


@pytest.mark.platforms('linux')
@pytest.mark.parametrize('restart', ['graceful', 'crash'])
def test_followers_survive_compression_and_restart_in_order_exactly_once(tmp_path, restart):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), CompactingModel)
    peer.requests = []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False},
        'model': {'provider': 'custom', 'default': 'original', 'base_url': url, 'context_length': 64000},
        'auxiliary': {'title_generation': {'enabled': False},
                      'compression': {'provider': 'custom', 'model': 'summary', 'base_url': url}},
        'compression': {'protect_first_n': 1, 'protect_last_n': 2, 'threshold_tokens': 12000,
                        'threshold': 0.5, 'in_place': False},
        'platform_toolsets': {'cli': []}}))
    (home / 'models_dev_cache.json').write_text(json.dumps({'custom': {'id': 'custom', 'models': {
        name: {'id': name, 'name': name, 'limit': {'context': 64000 if name == 'original' else 1000000,
                                                 'output': 1000}}
        for name in ('original', 'summary')}}}))
    env = child_env() | dict(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
                             PYTHONPATH=str(root), OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url,
                             PYTHONUNBUFFERED='1')

    def rows(sql, args=()):
        with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
            db.row_factory = sqlite3.Row
            return [dict(row) for row in db.execute(sql, args)]

    def admissions():
        return rows('SELECT admission_id,request_id,target_session_id,status,outcome,seq FROM session_admissions ORDER BY seq')

    def transcript(session_id):
        return [(r['id'], r['role'], r['content']) for r in rows(
            'SELECT id,role,content FROM messages WHERE session_id=? AND active=1 ORDER BY id', (session_id,))]

    def executed(marker):
        return sum(r['model'] == 'original' and r['text'].startswith(marker) for r in peer.requests)

    before = {}

    async def call(ws, method, **params):
        reply = await rpc(ws, method, **params)
        assert 'result' in reply, reply
        return reply['result']

    async def wait_rows(predicate, timeout=40):
        async with asyncio.timeout(timeout):
            while not predicate():
                await asyncio.sleep(.03)

    async def first(desc, owner):
        async with websocket(home, desc) as ws:
            sid = (await call(ws, 'session.create', source='cli', request_id='journey', model='original',
                              provider='custom', base_url=url, toolsets=[], cwd=str(home)))['session_id']
            before['sid'] = sid
            for i in range(12):
                params = dict(session_id=sid, input_id=f'pressure-{i}', text=f'PRESSURE_{i} ' + 'keep these facts ' * 100)
                accepted = await call(ws, 'prompt.submit', **params)
                await wait_rows(lambda: peer.blocked.is_set() or any(
                    r['admission_id'] == accepted['admission_id'] and r['status'] == 'terminal' for r in admissions()))
                if peer.blocked.is_set():
                    break
            else:
                raise AssertionError('automatic compression never published')
            compacting = params
            children = rows('SELECT id FROM sessions WHERE parent_session_id=?', (sid,))
            assert len(children) == 1, children
            receipts = {compacting['input_id']: accepted['admission_id']}
            for name in FOLLOWERS:
                receipts[name] = (await call(ws, 'prompt.submit', session_id=sid, input_id=name, text=name))['admission_id']
            pending = [r for r in admissions() if r['status'] != 'terminal']
            assert [(r['request_id'], r['status']) for r in pending] == [
                (compacting['input_id'], 'started')] + [(name, 'queued') for name in FOLLOWERS], pending
            before.update(compacting=compacting, receipts=receipts, child=children[0]['id'],
                          compacting_requests=executed(compacting['text'][:12]))
            assert sum(r['model'] == 'summary' for r in peer.requests) == 1
            # Restart while the compacted turn is still in flight: nothing behind it may run. A
            # graceful stop interrupts and settles it; a crash leaves it started (-> unknown).
            if restart == 'graceful':
                owner.send_signal(signal.SIGTERM)
            else:
                owner.kill()
            await asyncio.to_thread(owner.wait, 60)
            before['stopped'] = {r['request_id']: (r['status'], r['outcome']) for r in admissions()}
            before['compacted'] = transcript(before['child'])
            peer.release.set()
        assert all(executed(name) == 0 for name in FOLLOWERS), [r['text'] for r in peer.requests]

    async def second(desc):
        sid = before['sid']
        async with websocket(home, desc) as ws:
            resumed = await call(ws, 'session.resume', session_id=sid)
            unknown = [r for r in resumed['pending'] if r['status'] == 'unknown']
            head = before['receipts'][before['compacting']['input_id']]
            if restart == 'crash':
                # Started work that lost its owner is unknown and holds the FIFO; nothing ran yet.
                assert [r['admission_id'] for r in unknown] == [head], resumed['pending']
                assert [r['input_id'] for r in resumed['pending'] if r['status'] == 'queued'] == list(FOLLOWERS)
                assert all(executed(name) == 0 for name in FOLLOWERS)
            else:
                assert before['stopped'][before['compacting']['input_id']] == ('terminal', 'interrupted'), before['stopped']
                assert not unknown, resumed['pending']
            for lost in unknown:  # The documented operator step (/discard) for a turn lost in flight.
                await call(ws, 'prompt.resolve_unknown', session_id=sid, admission_id=lost['admission_id'],
                           execution_generation=lost['execution_generation'])
            await wait_rows(lambda: all(r['status'] == 'terminal' for r in admissions()))
            for input_id, admission_id in before['receipts'].items():
                text = before['compacting']['text'] if input_id == before['compacting']['input_id'] else input_id
                retried = await call(ws, 'prompt.submit', session_id=sid, input_id=input_id, text=text)
                assert retried['admission_id'] == admission_id, (input_id, retried)
            return await call(ws, 'session.resume', session_id=sid)

    try:
        with daemon(root, home, env, barrier=False) as (owner, desc):
            asyncio.run(first(desc, owner))
        with daemon(root, home, env, barrier=False) as (_, desc):
            final = asyncio.run(second(desc))
        sid, compacting = before['sid'], before['compacting']
        order = [r['text'] for r in peer.requests if r['model'] == 'original' and r['text'] in FOLLOWERS]
        assert order == list(FOLLOWERS), order
        assert executed(compacting['text'][:12]) == before['compacting_requests'], 'compacted turn replayed'
        for name in FOLLOWERS:
            wire = next(r['wire'] for r in peer.requests if r['text'] == name)
            # protect_first_n keeps PRESSURE_0 verbatim; the compacted middle is only inside the summary.
            turns = [str(m.get('content', '')) for m in json.loads(wire)]
            assert 'SUMMARY_RETAINED_FACTS' in wire and not any(t.startswith('PRESSURE_1 ') for t in turns), name
        settled = admissions()
        assert {r['target_session_id'] for r in settled} == {sid}
        assert [r['request_id'] for r in settled][-3:] == list(FOLLOWERS)
        assert [r['outcome'] for r in settled][-3:] == ['completed'] * 3, settled
        # Compaction state is the one committed before the restart: same single continuation, the root
        # still closed by compression, the summarized transcript a verbatim prefix of the live one.
        assert rows('SELECT id FROM sessions WHERE parent_session_id=?', (sid,)) == [{'id': before['child']}]
        assert rows('SELECT end_reason FROM sessions WHERE id=?', (sid,)) == [{'end_reason': 'compression'}]
        assert any('SUMMARY_RETAINED_FACTS' in str(c) for _, _, c in before['compacted'])
        assert transcript(before['child'])[:len(before['compacted'])] == before['compacted']
        stored = rows("SELECT session_id FROM messages WHERE role='user' AND content IN (?,?,?)", FOLLOWERS)
        assert [r['session_id'] for r in stored] == [before['child']] * 3, stored
        assert sum(r['model'] == 'summary' for r in peer.requests) == 1
        history = json.dumps(final['messages'])
        assert 'SUMMARY_RETAINED_FACTS' in history and all(name in history for name in FOLLOWERS)
        print(json.dumps({'admissions': settled, 'follower_order': order, 'child': before['child']}))
    finally:
        peer.release.set()
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)
