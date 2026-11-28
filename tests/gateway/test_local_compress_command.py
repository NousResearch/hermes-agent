"""``/compact`` on a LOCAL session commits the gateway's canonical compress mutation (#148272).

Bot Mode rewrites ``/new`` and ``/reset`` into ``/compact``, so that chat's only reset path is this
command. The gateway already implements compression as ``session.mutate compress`` (Desktop's native
action, ACP's route); ``slash.exec`` answered ``unsupported_command`` because ``command_verdict`` has
no ``compress`` entry, which left the bot with no way to shrink its context.
"""
import asyncio
import json
import signal
import sqlite3
import threading
import time
from contextlib import closing
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from tests.gateway.fixtures.local_recovery_probe import child_env, daemon, rpc, websocket
from tests.gateway.test_session_mutation_compress import SummaryModel


def test_compress_is_a_local_session_control():
    """The verdict is what refuses ``/compact`` before any handler runs (defect 1 of the report)."""
    from gateway.session_commands import command_verdict

    assert command_verdict('compress', '') == 'control'
    assert command_verdict('compress', '--preview') == 'control'
    assert command_verdict('compress', 'here 3') == 'control'


@pytest.mark.platforms("linux")  # the daemon is stopped with SIGINT, which raises on Windows
def test_compact_on_a_local_session_commits_the_compress_mutation(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user, work = tmp_path / 'state', tmp_path / 'user', tmp_path / 'work'
    home.mkdir(mode=0o700)
    user.mkdir()
    work.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), SummaryModel)
    peer.requests = []
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False},
        'model': {'provider': 'custom', 'default': 'original', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False},
                      'compression': {'provider': 'custom', 'model': 'summary', 'base_url': url}},
        'platform_toolsets': {'cli': []},
        'compression': {'protect_first_n': 1, 'protect_last_n': 1, 'min_tail_user_messages': 2,
                        'in_place': True},
    }))
    (home / 'models_dev_cache.json').write_text(json.dumps({'custom': {'id': 'custom', 'models': {
        name: {'id': name, 'name': name, 'limit': {'context': 1000000, 'output': 4000}}
        for name in ('original', 'summary')}}}))
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, PYTHONUNBUFFERED='1')
    out = {}

    async def probe(desc):
        async with websocket(home, desc) as ws:
            async def call(method, **params):
                response = await rpc(ws, method, **params)
                assert 'result' in response, response
                return response['result']

            sid = (await call('session.create', source='cli', request_id='compact', toolsets=[],
                              model='original', provider='custom', base_url=url,
                              cwd=str(work)))['session_id']
            out['sid'] = sid

            async def settled(text, timeout=45):
                snap = await call('session.resume', session_id=sid)
                deadline = time.monotonic() + timeout
                while snap['running'] or snap['pending'] or text not in json.dumps(snap['messages']):
                    assert time.monotonic() < deadline, ('turn never settled', snap)
                    await asyncio.sleep(.1)
                    snap = await call('session.resume', session_id=sid)
                return snap

            for i in range(6):
                await call('prompt.submit', session_id=sid, input_id=f'turn-{i}', text=f'HISTORY_{i}')
                await settled(f'HISTORY_{i}')
            before = await call('session.resume', session_id=sid)
            # A viewer's own connection submits ``/compact`` exactly as the Bot composer's fallback does.
            reply = await rpc(ws, 'slash.exec', session_id=sid, command='/compact')
            out['reply'] = reply

            def visible(snap):
                return [f"{m['role']}:{(m.get('content') or '')[:200]}" for m in snap['messages']
                        if not m.get('archived')]

            def contents(snap):
                return json.dumps([m.get('content') or '' for m in snap['messages'] if not m.get('archived')])

            out['before'] = visible(before)
            out['text_before'] = contents(before)
            after = await call('session.resume', session_id=sid)
            out['after'] = visible(after)
            out['text_after'] = contents(after)
            out['stats'] = await call('session.stats', session_id=sid)
            await call('prompt.submit', session_id=sid, input_id='after-compact', text='AFTER_COMPACT')
            out['final'] = await settled('AFTER_COMPACT')
            with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
                out['rows'] = db.execute(
                    'SELECT session_id, active FROM messages ORDER BY rowid').fetchall()

    try:
        with daemon(root, home, env, barrier=False) as (proc, desc):
            asyncio.run(probe(desc))
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=20)
    finally:
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)

    result = out['reply'].get('result')
    assert result is not None, out['reply']
    # The committed rewrite is what the transcript now reads: the oldest history is replaced by the
    # summary, and the protected tail (min_tail_user_messages) survives verbatim.
    assert 'HISTORY_1' in out['text_before'], out['text_before'][:800]
    # The rewrite committed: the compacted turns are replaced by the summary row, the protected tail
    # (min_tail_user_messages) survives verbatim, and the visible transcript shrinks.
    assert 'SUMMARY_RETAINED_FACTS' in out['text_after'], out['after'][:800]
    assert any(row.startswith('assistant:[CONTEXT COMPACTION') for row in out['after']), out['after']
    assert 'user:HISTORY_1' not in out['after'], out['after']
    assert 'user:HISTORY_5' in out['after'], out['after']
    assert len(out['after']) < len(out['before']), (out['before'], out['after'])
    assert 'Compressed' in result['output'], result
    # Main's agent counts the compaction (status bar, /usage): the replacement carries it before and
    # after its next turn, so /compact is a session boundary like Desktop's native action.
    assert out['stats']['compression_count'] == 1, out['stats']
    assert 'AFTER_COMPACT' in json.dumps(out['final']['messages'])
    # In-place compression archives the removed rows: the transcript stays one session.
    assert out['final']['session_id'] == out['sid']
    assert {row[0] for row in out['rows']} <= {out['sid']}, out['rows']
