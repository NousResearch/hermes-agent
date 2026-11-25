"""Prompt-cache guard: an announced client changes nothing the model is sent.

The same two-turn conversation runs twice on one real gateway process, recorded by a fake provider:
once from a connection that never said hello, once from one that announced a ``desktop`` label and
sent ``client.activity`` reports between the turns (so its turns carry the owner's client stamp and
its presence is tracked). Every provider request body (system prompt, tools, each message and any
cache marker) must be byte-identical across the two runs, request by request.
"""
from __future__ import annotations

import asyncio
from contextlib import closing
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import threading

from tests.gateway.fixtures.local_recovery_probe import Model, child_env, daemon, rpc, websocket
from tests.waiting import wait_until

DESKTOP = {'name': 'hermes-desktop', 'version': '2.1.0', 'kind': 'desktop'}
TURNS = ('FIRST_TURN', 'SECOND_TURN')


def _terminal(home, request_id):
    with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
        row = db.execute('SELECT status FROM session_admissions WHERE request_id=?', (request_id,)).fetchone()
    return row is not None and row[0] == 'terminal'


def _stamps(home):
    with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
        return {rid: json.loads(payload).get('surface_v1')
                for rid, payload in db.execute('SELECT request_id, payload_json FROM session_admissions')}


def test_provider_requests_are_byte_identical_with_and_without_an_announced_client(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    model = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    model.requests = []
    model.blocked, model.release = threading.Event(), threading.Event()
    thread = threading.Thread(target=model.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{model.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False},
        'model': {'provider': 'custom', 'default': 'local-wire-stub', 'base_url': base},
        'auxiliary': {'title_generation': {'enabled': False}},
        'platform_toolsets': {'gui': []}, 'terminal': {'cwd': str(home)},
    }))
    env = child_env() | dict(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
        PYTHONPATH=str(root), OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=base, PYTHONUNBUFFERED='1')
    bodies = {}

    async def conversation(desc, run, announce):
        async with websocket(home, desc) as ws:
            if announce:
                assert (await rpc(ws, 'client.capabilities', client=DESKTOP))['result']['client_activity'] is True
            created = await rpc(ws, 'session.create', request_id='conversation-' + run, source='gui',
                                cwd=str(home), toolsets=[])
            sid = created['result']['session_id']
            start = len(model.requests)
            for n, text in enumerate(TURNS):
                if announce and n:
                    for idle in (30, 0):
                        assert (await rpc(ws, 'client.activity', sessions=[sid], idle_seconds=idle,
                                          basis='system'))['result'] == {}
                reply = await rpc(ws, 'prompt.submit', session_id=sid, submission_id=f'{run}-{n}', text=text)
                assert reply.get('result', {}).get('status') == 'queued', reply
                await asyncio.to_thread(wait_until, lambda: _terminal(home, f'{run}-{n}'), timeout=60)
            bodies[run] = [json.dumps(body, sort_keys=True) for body in model.requests[start:]]

    async def drive(desc):
        await conversation(desc, 'plain', announce=False)
        await conversation(desc, 'announced', announce=True)

    try:
        with daemon(root, home, env, barrier=False) as (_proc, desc):
            asyncio.run(drive(desc))
    finally:
        model.release.set()
        model.shutdown()
        model.server_close()
        thread.join(timeout=5)

    stamps = _stamps(home)
    assert stamps['plain-0'] is None and stamps['announced-1'] == {'client': DESKTOP}, 'the second run is stamped'
    assert len(bodies['plain']) == len(TURNS) and len(bodies['announced']) == len(TURNS), bodies
    for plain, announced in zip(bodies['plain'], bodies['announced']):
        assert announced == plain
