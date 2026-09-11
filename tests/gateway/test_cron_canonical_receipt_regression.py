"""Cron consumes the real Bot Chat receipt over an owned daemon's WebSocket."""
import asyncio
from contextlib import closing
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest

from tests.gateway.fixtures.local_recovery_probe import Model, child_env, daemon, rpc, websocket


def admissions(home):
    with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute('SELECT * FROM session_admissions ORDER BY seq')]


def wait_for(predicate):
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.05)
    pytest.fail('Timed out waiting for the test-owned admission')


@pytest.fixture
def canonical_bot(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HOME', str(user))
    monkeypatch.setenv('USERPROFILE', str(user))
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
        PYTHONPATH=str(root), OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=base,
        PYTHONUNBUFFERED='1')

    async def prepare(desc):
        async with websocket(home, desc) as ws:
            created = await rpc(ws, 'session.create', request_id='cron-target', source='gui',
                                cwd=str(home), toolsets=[])
            assert 'result' in created, created
            sid = created['result']['session_id']
            renamed = await rpc(ws, 'session.mutate', session_id=sid, request_id='bot-title',
                expected_revision=created['result']['revision'], operation='rename',
                payload={'title': 'Bot Chat'})
            assert 'result' in renamed, renamed
            warm = await rpc(ws, 'prompt.submit', session_id=sid, input_id='warm', text='WARM_HISTORY')
            assert 'result' in warm, warm
            await asyncio.to_thread(wait_for, lambda: admissions(home)[0]['status'] == 'terminal')
            held = await rpc(ws, 'prompt.submit', session_id=sid, input_id='held', text='BLOCK_STARTED')
            assert 'result' in held, held
            assert await asyncio.to_thread(model.blocked.wait, 20)
            return sid

    try:
        with daemon(root, home, env, barrier=False) as (proc, desc):
            sid = asyncio.run(prepare(desc))
            try:
                yield SimpleNamespace(home=home, model=model, sid=sid, pid=proc.pid)
            finally:
                model.release.set()
        assert proc.poll() is not None
    finally:
        model.release.set()
        model.shutdown()
        model.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize('phase', ['queued', 'settled'])
def test_cron_tracks_real_canonical_receipt(canonical_bot, phase):
    from cron.scheduler_delivery import _deliver_to_bot_chat

    bot = canonical_bot
    job = {'id': 'owned-cron', 'name': 'Receipt check', 'execution_id': 'one-owned-execution'}
    content = 'CRON_REAL_RECEIPT_PROBE'
    initial = _deliver_to_bot_chat(job, content, '')

    def owned_rows():
        return [row for row in admissions(bot.home) if content in row['payload_json']]

    before = owned_rows()
    assert len(before) == 1 and before[0]['status'] == 'queued', (initial, before)
    admission_id = before[0]['admission_id']
    if phase == 'settled':
        bot.model.release.set()
        wait_for(lambda: owned_rows()[0]['status'] == 'terminal')
        observed = _deliver_to_bot_chat(job, content, '')
    else:
        observed = initial
    after = owned_rows()
    print(json.dumps({'phase': phase, 'daemon_pid': bot.pid, 'cron_result': observed,
                      'admission_id': admission_id, 'admission_status': after[0]['status'],
                      'admission_count': len(after),
                      'bookkeeping': job.get('_bot_chat_delivery_receipts')}))
    assert len(after) == 1 and after[0]['admission_id'] == admission_id
    if phase == 'queued':
        assert observed and 'queued' in observed, observed
        assert job['_bot_chat_delivery_receipts']['bot-chat:(own)']['status'] == 'queued'
        assert _deliver_to_bot_chat(job, content, '') == observed
    else:
        assert observed is None, observed
        assert job['_bot_chat_delivery_receipts']['bot-chat:(own)']['status'] == 'settled'
        assert _deliver_to_bot_chat(job, content, '') is None
    assert len(owned_rows()) == 1
