"""Cron Bot Chat delivery creates the profile's canonical Bot Chat when it does not exist yet.

Main's lane was ``hermes chat -c "Bot Chat" --create-if-missing``; the canonical authority
door must keep that semantics, or a fresh profile's very first ``deliver: bot-chat`` job is
recorded as failed with nowhere to land.
"""
from contextlib import closing
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import threading

import pytest

from tests.gateway.fixtures.local_recovery_probe import Model, child_env, daemon


@pytest.mark.linux_only
def test_cron_bot_chat_delivery_creates_the_missing_bot_chat_and_admits(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    model = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    model.requests = []
    model.blocked, model.release = threading.Event(), threading.Event()
    threading.Thread(target=model.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{model.server_port}/v1'
    try:
        (home / 'config.yaml').write_text(json.dumps({
            'gateway': {'multiplex_profiles': False},
            'model': {'provider': 'custom', 'default': 'local-wire-stub', 'base_url': url},
            'auxiliary': {'title_generation': {'enabled': False}}, 'terminal': {'cwd': str(home)}}),
            encoding='utf-8')
        env = child_env()
        env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
                   OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, PYTHONUNBUFFERED='1')
        monkeypatch.setenv('HERMES_HOME', str(home))
        monkeypatch.setattr(Path, 'home', lambda: user)
        with daemon(root, home, env, barrier=False):
            from cron.scheduler_delivery import _deliver_to_bot_chat
            job = {'id': 'digest', 'name': 'Digest', 'execution_id': 'run-1'}
            first = _deliver_to_bot_chat(job, 'FIRST_PAYLOAD', '')
            receipt = job['_bot_chat_delivery_receipts']['bot-chat:(own)']
            assert receipt['status'] in ('queued', 'claimed', 'settled'), (first, receipt)
            with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
                titled = db.execute("SELECT id FROM sessions WHERE title='Bot Chat'").fetchall()
                assert len(titled) == 1, titled
                admitted = db.execute('SELECT target_session_id, payload_json FROM session_admissions').fetchall()
            assert len(admitted) == 1 and 'FIRST_PAYLOAD' in admitted[0][1], admitted
            # The next run lands in the SAME Bot Chat: created once, then resolved by name.
            second = _deliver_to_bot_chat({**job, 'execution_id': 'run-2'}, 'SECOND_PAYLOAD', '')
            with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
                assert db.execute("SELECT count(*) FROM sessions WHERE title='Bot Chat'").fetchone()[0] == 1
                targets = {row[0] for row in db.execute('SELECT target_session_id FROM session_admissions')}
            assert len(targets) == 1, (first, second, targets)
    finally:
        model.shutdown()
        model.server_close()
