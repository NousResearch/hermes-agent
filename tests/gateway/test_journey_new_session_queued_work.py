"""Journey: client `/new` while the earlier session still holds durable queued work.

One ordinary daemon (real authority, SQLite, in-process agent loop, loopback model). The classic CLI
view's own `/new` handler (``hermes_cli.gateway_chat_commands``) runs over a real ``GatewayClient``
while session A has a turn in flight and two followers queued. Documented contract (PR behaviour
change 3): `/new` rebinds only the invoking view; A keeps its FIFO. Asserted:
- B is a fresh canonical session: its own FIFO (it runs while A is blocked), and no A input ever
  reaches B's transcript or B's model requests;
- A's followers stay queued on A, visible in A's snapshot (not B's), and drain on A in order, once;
- another viewer still attached to A stays on A and sees A's completions;
- ``restart``: the owner crashes after `/new`; A's in-flight turn is unknown and holds only A's
  followers (B keeps working); discarding it releases them on A, exactly once.
"""
import asyncio
from contextlib import closing
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import threading

import pytest

from tests.gateway.fixtures.local_recovery_probe import Model, child_env, daemon, rpc, websocket

A_FOLLOWERS = ('A_FOLLOWER_1', 'A_FOLLOWER_2')


@pytest.mark.platforms('linux')
@pytest.mark.parametrize('restart', [False, True], ids=['live', 'restart'])
def test_new_session_never_absorbs_or_drops_earlier_queued_work(tmp_path, restart):
    from hermes_cli.gateway_chat_commands import run_command
    from hermes_cli.gateway_chat_view import GatewayChatView
    from hermes_cli.gateway_client import GatewayClient

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
        'gateway': {'multiplex_profiles': False},
        'model': {'provider': 'custom', 'default': 'view-model', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False}}, 'platform_toolsets': {'cli': []}}))
    env = child_env() | dict(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
                             OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, PYTHONUNBUFFERED='1')

    def admissions():
        with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
            db.row_factory = sqlite3.Row
            return {r['request_id']: dict(r) for r in db.execute(
                'SELECT request_id,target_session_id,status,outcome,seq FROM session_admissions')}

    def user_turns(session_id):
        with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
            return [r[0] for r in db.execute("SELECT content FROM messages WHERE session_id=? AND role='user' "
                                              'ORDER BY id', (session_id,))]

    def requests():
        return [[m.get('content') for m in r['messages'] if m['role'] == 'user'] for r in peer.requests]

    async def until(predicate, timeout=30):
        async with asyncio.timeout(timeout):
            while not predicate():
                await asyncio.sleep(.03)

    ids = {}

    async def first(desc, owner):
        async with websocket(home, desc) as ws, websocket(home, desc) as other, GatewayClient(ws) as client:
            snapshot = await client.rpc('session.create', request_id='journey-a', source='cli', cwd=str(home),
                                        model='view-model', provider='custom', base_url=url, toolsets=[])
            view = GatewayChatView(client, snapshot, quiet=True)
            a = ids['a'] = view.session_id
            assert 'result' in await rpc(other, 'session.resume', session_id=a)
            await client.rpc('prompt.submit', session_id=a, input_id='A_HEAD', text='BLOCK_STARTED')
            assert await asyncio.to_thread(peer.blocked.wait, 20)
            for name in A_FOLLOWERS:
                await client.rpc('prompt.submit', session_id=a, input_id=name, text=name)
            assert await run_command(view, '/new', '') is True
            b = ids['b'] = view.session_id
            assert b != a
            # B is its own FIFO: it answers while A's head is still blocked in the model.
            await view.submit('B_TURN')
            await until(lambda: any(r['target_session_id'] == b and r['status'] == 'terminal'
                                    for r in admissions().values()))
            rows = admissions()
            assert {k: (r['target_session_id'], r['status']) for k, r in rows.items() if k.startswith('A_')} == {
                'A_HEAD': (a, 'started'), **{n: (a, 'queued') for n in A_FOLLOWERS}}, rows
            b_snapshot = await client.rpc('session.resume', session_id=b)
            a_snapshot = (await rpc(other, 'session.resume', session_id=a))['result']
            assert b_snapshot['pending'] == [] and b_snapshot['messages'][0]['content'] == 'B_TURN', b_snapshot
            assert [r['input_id'] for r in a_snapshot['pending']] == ['A_HEAD', *A_FOLLOWERS], a_snapshot['pending']
            assert requests()[-1] == ['B_TURN'], requests()
            if restart:
                owner.kill()
                await asyncio.to_thread(owner.wait, 10)
                peer.release.set()
                return
            peer.release.set()
            await until(lambda: all(r['status'] == 'terminal' for r in admissions().values()))
            # The viewer that stayed on A saw A's work finish; the /new view was rebound to B only.
            events = (await rpc(other, 'session.events.since', session_id=a, replay_epoch=a_snapshot['replay_epoch'],
                                last_sequence=a_snapshot['last_sequence']))['result']['events']
            assert {e.get('payload', {}).get('text') for e in events if e['type'] == 'message.complete'} >= {
                'RECOVERY_ACK_' + n for n in A_FOLLOWERS}, events
            assert view.session_id == b

    async def second(desc):
        a, b = ids['a'], ids['b']
        async with websocket(home, desc) as ws, GatewayClient(ws) as client:
            a_snapshot = await client.rpc('session.resume', session_id=a)
            lost = [r for r in a_snapshot['pending'] if r['status'] == 'unknown']
            assert [r['input_id'] for r in lost] == ['A_HEAD'], a_snapshot['pending']
            assert [r['input_id'] for r in a_snapshot['pending'] if r['status'] == 'queued'] == list(A_FOLLOWERS)
            # A's unknown head pauses only A: B takes and finishes new work meanwhile.
            await client.rpc('session.resume', session_id=b)
            await client.rpc('prompt.submit', session_id=b, input_id='B_AFTER_RESTART', text='B_AFTER_RESTART')
            await until(lambda: admissions()['B_AFTER_RESTART']['status'] == 'terminal')
            assert all(admissions()[n]['status'] == 'queued' for n in A_FOLLOWERS)
            assert not any(n in sent for sent in requests() for n in A_FOLLOWERS)
            await client.rpc('prompt.resolve_unknown', session_id=a, admission_id=lost[0]['admission_id'],
                             execution_generation=lost[0]['execution_generation'])
            await until(lambda: all(r['status'] == 'terminal' for r in admissions().values()))

    try:
        with daemon(root, home, env, barrier=False) as (owner, desc):
            asyncio.run(first(desc, owner))
        if restart:
            with daemon(root, home, env, barrier=False) as (_, desc):
                asyncio.run(second(desc))
        a, b = ids['a'], ids['b']
        rows = admissions()
        assert all(rows[n]['target_session_id'] == a and rows[n]['outcome'] == 'completed' for n in A_FOLLOWERS), rows
        assert rows['A_HEAD']['outcome'] == ('interrupted' if restart else 'completed'), rows['A_HEAD']
        assert {r['target_session_id'] for k, r in rows.items() if k.startswith('B_') or len(k) == 32} == {b}, rows
        sent = requests()
        assert [s[-1] for s in sent if s[-1] in A_FOLLOWERS] == list(A_FOLLOWERS), sent
        assert sum(s[-1] == 'BLOCK_STARTED' for s in sent) == 1, sent
        b_users = user_turns(b)
        assert b_users == ['B_TURN'] + (['B_AFTER_RESTART'] if restart else []), b_users
        assert not any(t in ('BLOCK_STARTED', *A_FOLLOWERS) for s in sent if 'B_TURN' in s for t in s), sent
        assert [t for t in user_turns(a) if t in A_FOLLOWERS] == list(A_FOLLOWERS)
        print(json.dumps({'restart': restart, 'a': a, 'b': b, 'admissions': rows, 'requests': sent}))
    finally:
        peer.release.set()
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)
