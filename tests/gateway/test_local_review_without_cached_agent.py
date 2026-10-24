"""``/review`` on a local session whose cached agent is gone.

The gateway drops a session's cached agent after an idle hour, under memory pressure, and around a
``/moa`` turn. ``/review`` read only the cached agent, so a session left alone for an hour answered
"Nothing to review yet — send a message first" until the next message rebuilt one. The conversation
is the session's stored transcript; the reviewer reviews it, its completion re-enters the session, and
its request carries what a review from the resident agent carries (the session's provider routing,
not the daemon's startup copy, and its prefill examples).

Real daemon (``python -m gateway.run``), ``slash.exec`` (``session_commands.execute_command``),
loopback model. ``/moa`` drops the cached agent here because the idle sweep's first pass is a minute in.
"""
import asyncio
import contextlib
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
def test_local_review_reviews_the_stored_conversation_once_the_cached_agent_is_gone(tmp_path):
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
    config = {
        'gateway': {'multiplex_profiles': False},
        'model': {'provider': 'openrouter', 'default': 'openai/gpt-5', 'base_url': url,
                  'api_key': 'loopback-only', 'api_mode': 'chat_completions'},
        'auxiliary': {'title_generation': {'enabled': False}},
        'provider_routing': {'only': ['daemon-start'], 'data_collection': 'allow'},
        'prefill_messages_file': 'prefill.json',
    }
    (home / 'prefill.json').write_text(json.dumps([{'role': 'user', 'content': 'PREFILL_EXAMPLE'},
                                                  {'role': 'assistant', 'content': 'Noted.'}]))
    (home / 'config.yaml').write_text(json.dumps(config))
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENROUTER_API_KEY='loopback-only', OPENAI_API_KEY='loopback-only', PYTHONUNBUFFERED='1')
    out = {}

    def settled():
        with contextlib.closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
            return all(row[0] == 'terminal' for row in db.execute('SELECT status FROM session_admissions'))

    async def review(ws, sid, focus):
        before = len(peer.requests)
        reply = await rpc(ws, 'slash.exec', session_id=sid, command='review ' + focus)
        result = reply.get('result') or reply.get('error')
        assert 'Review started' in json.dumps(result), (focus, result)

        def reviewer(request):
            return any(m.get('content') == 'Review: ' + focus for m in request.get('messages', []))
        await async_wait_until(lambda: any(reviewer(r) for r in peer.requests[before:]), timeout=45,
                               what=f'the {focus} reviewer reaching the model')
        request = next(r for r in peer.requests[before:] if reviewer(r))
        text = json.dumps(request['messages'])
        # The reviewer's completion comes back to this session as a delegation-completion turn.
        await async_wait_until(lambda: any('ASYNC DELEGATION' in json.dumps(r) for r in peer.requests[before:]),
                               timeout=60, what=f'the {focus} completion re-entering the session')
        await async_wait_until(settled, timeout=60, what=f'the {focus} completion turn settling')
        return {'provider': request.get('provider'), 'prefill': 'PREFILL_EXAMPLE' in text,
                'saw_conversation': 'OPENED_PR_42' in text}

    async def probe(desc):
        async with websocket(home, desc) as ws:
            # A local session takes the profile's config as of its creation, not the daemon's startup copy.
            config['provider_routing'] = {'only': ['session'], 'data_collection': 'deny'}
            (home / 'config.yaml').write_text(json.dumps(config))
            created = await rpc(ws, 'session.create', request_id='cold-review', source='cli', toolsets=[],
                                cwd=str(work))
            sid = created['result']['session_id']
            await rpc(ws, 'prompt.submit', session_id=sid, input_id='warm', text='OPENED_PR_42')
            await async_wait_until(settled, timeout=45, what='the warm turn settling')
            out['warm'] = await review(ws, sid, 'resident pass')
            moa = (await rpc(ws, 'slash.exec', session_id=sid, command='moa compare the options'))['result']
            await rpc(ws, 'prompt.submit', session_id=sid, input_id='moa', text=moa['message'])
            await async_wait_until(settled, timeout=45, what='the /moa turn settling')
            out['cold'] = await review(ws, sid, 'cold pass')

    try:
        with daemon(root, home, env, barrier=False) as (proc, desc):
            asyncio.run(probe(desc))
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=20)
    finally:
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)

    assert out['cold']['saw_conversation'], out
    assert out['warm']['provider'] == {'only': ['session'], 'data_collection': 'deny'}, out
    assert out['cold'] == out['warm'], ('a cold /review must send what a resident one sends', out)
