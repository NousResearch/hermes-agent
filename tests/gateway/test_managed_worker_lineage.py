"""Real owner parity: stable admission IDs, physical history and terminal publication."""
import asyncio
from contextlib import closing
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import threading

import pytest

from tests.gateway.fixtures.local_recovery_probe import Model, child_env, daemon, rpc, websocket


@pytest.mark.parametrize(('mode', 'invariant'), [
    (mode, invariant) for mode in ('normal', 'managed', 'safe')
    for invariant in ('finish', 'reset', 'rewind')
] + [('managed', 'initialization')])
def test_worker_lineage_and_settlement(tmp_path, mode, invariant):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests = []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False, 'managed_workers': mode == 'managed'},
        'model': {'provider': 'custom', 'default': 'lineage', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False}}, 'platform_toolsets': {'cli': []}}))
    if invariant == 'initialization':
        config = json.loads((home / 'config.yaml').read_text())
        config['model']['context_length'] = 32000
        (home / 'config.yaml').write_text(json.dumps(config))
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
               PYTHONPATH=str(root), OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url)

    def query(sql, args=()):
        with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
            return db.execute(sql, args).fetchall()

    async def call(ws, method, **params):
        reply = await rpc(ws, method, **params)
        assert 'result' in reply, reply
        return reply['result']

    async def settled(ws, sid, input_id):
        async with asyncio.timeout(45):
            while query('SELECT status FROM session_admissions WHERE request_id=?', (input_id,)) != [('terminal',)]:
                await asyncio.sleep(.005)
        return await call(ws, 'session.resume', session_id=sid)

    async def mutate(ws, sid, request_id, operation, payload):
        snap = await call(ws, 'session.resume', session_id=sid)
        return await call(ws, 'session.mutate', session_id=sid, request_id=request_id,
                          operation=operation, payload=payload, expected_revision=snap['revision'],
                          expected_generation=snap['execution_generation'])

    async def exercise(desc):
        async with websocket(home, desc) as ws:
            created = await call(ws, 'session.create', request_id='create', source='cli', cwd=str(home),
                                 model='lineage', provider='custom', base_url=url, toolsets=[],
                                 safe_mode=mode == 'safe', ignore_user_config=mode == 'safe')
            sid = created['session_id']
            first = await call(ws, 'prompt.submit', session_id=sid, input_id='before', text='BEFORE_RESET')
            if invariant == 'initialization':
                async with asyncio.timeout(30):
                    while query('SELECT status FROM session_admissions') != [('unknown',)]:
                        await asyncio.sleep(.02)
                assert peer.requests == []
                errors = (home / 'logs/errors.log').read_text()
                assert 'Managed worker lost: managed_worker_failed' in errors
                assert 'invalid_worker_frame' not in errors
                return
            snap = await settled(ws, sid, 'before')
            if invariant == 'finish':
                replay = await call(ws, 'session.events.since', session_id=sid,
                                    replay_epoch=snap['replay_epoch'], last_sequence=0)
                assert any(e['type'] == 'message.complete' for e in replay['events']), replay
                if mode != 'normal':
                    pending = list((home / 'worker-outboxes').rglob('pending.json'))
                    assert pending
                    assert all(json.loads(p.read_text())['pending'] == [] for p in pending)
                    assert not query("SELECT execution_id FROM worker_executions WHERE status!='terminal'")
                duplicate = await call(ws, 'prompt.submit', session_id=sid, input_id='before', text='BEFORE_RESET')
                assert duplicate['admission_id'] == first['admission_id']
                return
            # Base's broken finish handshake needs to unwind before the independent reset probe.
            await asyncio.sleep(.5)
            reset = await mutate(ws, sid, 'reset', 'reset', {})
            target = reset['target_session_id']
            await call(ws, 'prompt.submit', session_id=sid, input_id='after', text='AFTER_RESET')
            snap = await settled(ws, sid, 'after')
            if invariant == 'reset':
                assert 'BEFORE_RESET' not in json.dumps(peer.requests[-1]['messages'])
                assert 'AFTER_RESET' in json.dumps(snap['messages'])
                assert query("SELECT session_id FROM messages WHERE role='user' AND content LIKE '%AFTER_RESET%'") == [(target,)]
                assert query('SELECT DISTINCT target_session_id FROM session_admissions') == [(sid,)]
            else:
                await asyncio.sleep(.5)
                rows = query("SELECT id FROM messages WHERE session_id=? AND role='user'", (target,))
                assert rows, snap
                rewound = await mutate(ws, sid, 'rewind', 'rewind', {'target_message_id': rows[-1][0]})
                assert rewound['rewound_count'] > 0
                retry = await call(ws, 'session.resume', session_id=sid)
                assert 'RECOVERY_ACK_AFTER_RESET' not in json.dumps(retry['messages'])
            print(json.dumps({'mode': mode, 'invariant': invariant, 'logical': sid,
                              'physical': target, 'wire': peer.requests[-1]['messages']}))

    try:
        with daemon(root, home, env, barrier=False) as (_, desc):
            asyncio.run(exercise(desc))
    finally:
        peer.release.set()
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)
