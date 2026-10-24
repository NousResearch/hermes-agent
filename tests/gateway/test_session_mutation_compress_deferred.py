"""/compress sent while a turn runs waits for that turn instead of being refused (session_busy).

Real owner daemon + loopback model: the compress is accepted as ``deferred``, runs at the turn
boundary before the next queued message (which then sees the summary), and is announced on a
``compacted`` status edge. A bare /compress is skipped when the turn already auto-compacted.
"""
import asyncio
from contextlib import closing
from http.server import ThreadingHTTPServer
import io
import json
from pathlib import Path
import sqlite3
import threading
import time

import pytest
from tests.gateway.fixtures.local_recovery_probe import child_env, daemon, websocket
from tests.gateway.test_session_mutation_compress import SummaryModel
from tests.waiting import async_wait_until


def _is_summary(body):
    return any('summarization agent' in str(m.get('content', '')) for m in body.get('messages', []))


class SlowTurnModel(SummaryModel):
    """Holds the reply to a ``SLOW_TURN`` message open until the test releases it."""

    def do_POST(self):
        raw = self.rfile.read(int(self.headers['Content-Length']))
        body = json.loads(raw)
        last_user = next((m for m in reversed(body.get('messages') or []) if m.get('role') == 'user'), {})
        if 'SLOW_TURN' in str(last_user.get('content', '')) and not _is_summary(body):
            self.server.release.wait(timeout=60)
        self.rfile = io.BytesIO(raw)
        SummaryModel.do_POST(self)


@pytest.mark.parametrize('auto_compacts', [False, True])
def test_compress_during_turn_runs_at_the_boundary(tmp_path, auto_compacts):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), SlowTurnModel)
    peer.requests, peer.release = [], threading.Event()
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    compression = {'protect_first_n': 1, 'protect_last_n': 2, 'min_tail_user_messages': 2}
    if auto_compacts:  # every turn crosses it, the slow one included
        compression['threshold_tokens'] = 12000
    cfg = {'gateway': {'multiplex_profiles': False, 'managed_workers': False},
           'model': {'provider': 'custom', 'default': 'original', 'base_url': url},
           'auxiliary': {'title_generation': {'enabled': False},
                         'compression': {'provider': 'custom', 'model': 'summary', 'base_url': url}},
           'platform_toolsets': {'cli': []}, 'compression': compression}
    (home / 'config.yaml').write_text(json.dumps(cfg), encoding='utf-8')
    (home / 'models_dev_cache.json').write_text(json.dumps({'custom': {'id': 'custom', 'models': {
        name: {'id': name, 'name': name, 'limit': {'context': 1000000, 'output': 4000}}
        for name in ('original', 'summary')}}}), encoding='utf-8')
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, PYTHONUNBUFFERED='1')

    def status(request_id):
        with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
            row = db.execute('SELECT status FROM session_admissions WHERE request_id=?', (request_id,)).fetchone()
        return row and row[0]

    async def run(desc):
        async with websocket(home, desc) as ws:
            frames = []

            async def call(method, **params):
                rid = f'{method}-{len(frames)}-{time.monotonic()}'
                await ws.send(json.dumps({'jsonrpc': '2.0', 'id': rid, 'method': method, 'params': params}))
                async with asyncio.timeout(30):
                    while (reply := json.loads(await ws.recv())).get('id') != rid:
                        frames.append(reply)
                assert 'result' in reply, reply
                return reply['result']

            sid = (await call('session.create', source='cli', request_id='owner', toolsets=[],
                              model='original', provider='custom', base_url=url))['session_id']
            for i in range(8):
                await call('prompt.submit', session_id=sid, input_id=f'H{i}', text=f'HISTORY_{i}')
                await async_wait_until(lambda i=i: status(f'H{i}') == 'terminal', timeout=30, what='history')
            await call('prompt.submit', session_id=sid, input_id='SLOW', text='SLOW_TURN')
            await async_wait_until(lambda: status('SLOW') == 'started', timeout=30, what='slow turn started')
            live = await call('session.resume', session_id=sid)
            reply = await call('session.mutate', session_id=sid, request_id='compress', operation='compress',
                               payload={}, expected_revision=live['revision'],
                               expected_generation=live['execution_generation'])
            assert reply['status'] == 'deferred', reply
            assert status('SLOW') == 'started'  # accepted without waiting for (or stopping) the turn
            await call('prompt.submit', session_id=sid, input_id='NEXT', text='NEXT_TURN')
            peer.release.set()
            await async_wait_until(lambda: status('NEXT') == 'terminal', timeout=60, what='next turn')
            await call('session.stats', session_id=sid)  # flush events published before the next turn
            return [f['params']['payload'] for f in frames if (f.get('params') or {}).get('type') == 'status.update']
    try:
        with daemon(root, home, env, barrier=False) as (_, desc):
            edges = asyncio.run(run(desc))
    finally:
        peer.release.set()
        peer.shutdown()
        peer.server_close()
    texts = [e.get('text', '') for e in edges if e.get('kind') == 'compacted']
    next_turn = next(b for b in peer.requests if 'NEXT_TURN' in json.dumps(b.get('messages')))
    if auto_compacts:
        assert any(t.startswith('Queued /compress skipped') for t in texts), edges
    else:
        # Exactly one summary (the queued compress), committed before the queued message ran.
        assert sum(map(_is_summary, peer.requests)) == 1
        assert 'SUMMARY_RETAINED_FACTS' in json.dumps(next_turn['messages'])
        assert any(t.startswith('Compressed:') for t in texts), edges
