"""``/moa`` then ``/refine`` on a local session (T7).

``/moa`` arms a one-shot model swap and drops the session's cached agent; when its turn settles the
restore drops the MoA-built agent too. ``/refine`` used to read the conversation off the cached
agent, so straight after a ``/moa`` turn it answered "Nothing to refine yet". The conversation
belongs to the session (its stored transcript), so ``/refine`` still finds it.

Real daemon (``python -m gateway.run``), ``slash.exec`` (``session_commands.execute_command``),
loopback model.
"""
import asyncio
import json
import signal
import sqlite3
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from tests.gateway.fixtures.local_recovery_probe import Model, child_env, daemon, rpc, websocket
from tests.waiting import async_wait_until


@pytest.mark.platforms("linux")  # the daemon is stopped with SIGINT, which raises on Windows
def test_local_refine_finds_the_conversation_after_a_moa_turn(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user, work = tmp_path / 'state', tmp_path / 'user', tmp_path / 'work'
    home.mkdir(mode=0o700)
    user.mkdir()
    work.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests = []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False},
        'model': {'provider': 'custom', 'default': 'refine-model', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False}},
    }))
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, PYTHONUNBUFFERED='1')
    out = {}

    async def probe(desc):
        async with websocket(home, desc) as ws:
            created = await rpc(ws, 'session.create', request_id='moa-refine', source='cli', toolsets=[],
                                cwd=str(work))
            sid = created['result']['session_id']

            def rows():
                with sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True) as db:
                    return dict(db.execute('SELECT request_id,status FROM session_admissions'))

            async def settle(input_id, marker):
                await async_wait_until(lambda: rows().get(input_id) == 'terminal', timeout=45,
                                       what=f'{input_id} turn terminal')
                if marker is not None:
                    snap = (await rpc(ws, 'session.resume', session_id=sid))['result']
                    assert marker in json.dumps(snap['messages']), snap

            await rpc(ws, 'prompt.submit', session_id=sid, input_id='warm', text='WARM_REFINE')
            await settle('warm', 'RECOVERY_ACK_WARM_REFINE')
            moa = (await rpc(ws, 'slash.exec', session_id=sid, command='moa compare the options'))['result']
            out['moa'] = moa
            # The client submits /moa's rewritten turn with its own input id; it settles either way.
            await rpc(ws, 'prompt.submit', session_id=sid, input_id='moa', text=moa['message'])
            await settle('moa', None)
            reply = await rpc(ws, 'slash.exec', session_id=sid, command='refine keep the naming')
            out['refine'] = reply.get('result') or reply.get('error')
            # The review fork reaches the model with the session's conversation and the focus.
            await async_wait_until(lambda: any('keep the naming' in json.dumps(r) for r in peer.requests),
                                   timeout=30, what='the review fork reaching the model')
            review = next(r for r in peer.requests if 'keep the naming' in json.dumps(r))
            out['review_saw_conversation'] = 'WARM_REFINE' in json.dumps(review['messages'])

    try:
        with daemon(root, home, env, barrier=False) as (proc, desc):
            asyncio.run(probe(desc))
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=20)
    finally:
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)

    assert out['moa']['type'] == 'send', out
    refine = out['refine']
    assert refine.get('type') == 'exec', out
    assert 'Nothing to refine' not in refine['output'], ('/refine lost the conversation after /moa', out)
    assert 'Reviewing this conversation' in refine['output'] and 'keep the naming' in refine['output'], out
    assert out['review_saw_conversation'], out
