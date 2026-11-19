"""/new on a chat paused on a lost turn says what it leaves behind, and the way back works (#145177).

A gateway restart mid-turn fences the turn ``unknown`` and pauses the chat's queue; messages sent
while paused stay saved behind it. ``/new`` moves the chat to a fresh conversation and keeps those
messages saved on the old one (test_native_pause_notice pins that), so the reset reply has to say
how many were left and how to run them: ``/resume`` the old conversation, then ``/discard``.
"""
from contextlib import closing
from http.server import ThreadingHTTPServer
import importlib.machinery
import json
from pathlib import Path
import queue
import sqlite3
import threading
import time

import pytest

from tests.gateway.fixtures.local_recovery_probe import Model, child_env, daemon
from tests.gateway.test_native_telegram_startup_recovery import BotAPI, wait_for

LEFT_MARKER = 'still saved on the previous conversation'


@pytest.mark.skipif(importlib.machinery.PathFinder.find_spec("telegram") is None,
                    reason="python-telegram-bot not installed (on-demand extra)")
@pytest.mark.platforms("linux")
def test_new_on_a_paused_chat_names_the_saved_messages_and_resume_discard_runs_them(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    model = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    model.requests = []
    model.blocked, model.release = threading.Event(), threading.Event()
    bot = ThreadingHTTPServer(('127.0.0.1', 0), BotAPI)
    bot.calls, bot.sent, bot.updates = [], [], queue.Queue()
    for peer in (model, bot):
        threading.Thread(target=peer.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{model.server_port}/v1'
    cfg = {'model': {'provider': 'custom', 'default': 'local-wire-stub', 'base_url': base},
           'gateway': {'multiplex_profiles': False},
           'platforms': {'telegram': {'enabled': True, 'extra': {
               'base_url': f'http://127.0.0.1:{bot.server_port}/bot', 'dm_policy': 'allowlist'}}},
           'auxiliary': {'title_generation': {'enabled': False}},
           'platform_toolsets': {'telegram': []}, 'terminal': {'cwd': str(home)}}
    (home / 'config.yaml').write_text(json.dumps(cfg), encoding='utf-8')
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=base, PYTHONUNBUFFERED='1',
               TELEGRAM_BOT_TOKEN='987654321:owned-loopback-fixture', TELEGRAM_ALLOWED_USERS='202')

    def rows():
        with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
            return {json.loads(payload)['text']: {'session': sid, 'status': status}
                    for sid, status, payload in db.execute(
                        'SELECT target_session_id,status,payload_json FROM session_admissions')}

    def sent_texts():
        return [m.get('text', '').replace('\\', '') for m in bot.sent]

    def diagnostic():
        return repr(rows()) + '\n' + repr(sent_texts()) + '\n' + '\n'.join(
            p.read_text(encoding='utf-8')[-6000:] for p in (home / 'logs').glob('*.log'))

    update_id = 0

    def push(text):
        nonlocal update_id
        update_id += 1
        bot.updates.put({'update_id': update_id, 'message': {
            'message_id': update_id, 'date': int(time.time()),
            'chat': {'id': 202, 'type': 'private', 'first_name': 'Owned'},
            'from': {'id': 202, 'is_bot': False, 'first_name': 'Owned'}, 'text': text}})

    def send(text, status):
        push(text)
        wait_for(lambda: rows().get(text, {}).get('status') == status, diagnostic)

    try:
        with daemon(root, home, env, barrier=False) as (proc, _desc):
            send('WARM_HISTORY', 'terminal')
            send('BLOCK_STARTED', 'started')
            assert model.blocked.wait(20), diagnostic()
            proc.kill()
            assert proc.wait(timeout=10) == -9
        model.release.set()
        with daemon(root, home, env, barrier=False) as (proc, _desc):
            wait_for(lambda: rows()['BLOCK_STARTED']['status'] == 'unknown', diagnostic)
            paused = rows()['BLOCK_STARTED']['session']
            send('AFTER_CRASH_ONE', 'queued')
            send('AFTER_CRASH_TWO', 'queued')
            before = len(bot.sent)
            push('/new')
            wait_for(lambda: any('Confirm /new' in t for t in sent_texts()[before:]), diagnostic)
            push('/approve')
            wait_for(lambda: any('Session reset' in t or 'New session' in t for t in sent_texts()[before:]),
                     diagnostic)
            reset_reply = next(t for t in sent_texts()[before:] if 'Session reset' in t or 'New session' in t)
            # The reset reply names what it left behind and the way back.
            assert LEFT_MARKER in reset_reply and '2 message' in reset_reply, reset_reply
            assert f'/resume {paused}' in reset_reply and '/discard' in reset_reply, reset_reply
            assert rows()['AFTER_CRASH_ONE']['status'] == 'queued'
            # Control: the fresh conversation answers, and a /new with nothing paused says nothing extra.
            send('AFTER_RESET', 'terminal')
            # The way back the reply names works from the chat: resume, discard, and the saved run.
            push(f'/resume {paused}')
            wait_for(lambda: any('Resumed' in t or 'resumed' in t for t in sent_texts()), diagnostic)
            push('/discard')
            wait_for(lambda: rows()['AFTER_CRASH_TWO']['status'] == 'terminal', diagnostic)
            wait_for(lambda: any('RECOVERY_ACK_AFTER_CRASH_TWO' in t for t in sent_texts()), diagnostic)
            assert rows()['BLOCK_STARTED']['status'] != 'unknown'
            before = len(bot.sent)
            push('/new')
            wait_for(lambda: any('Confirm /new' in t or 'Session reset' in t or 'New session' in t
                                 for t in sent_texts()[before:]), diagnostic)
            if any('Confirm /new' in t for t in sent_texts()[before:]):
                push('/approve')
            wait_for(lambda: any('Session reset' in t or 'New session' in t for t in sent_texts()[before:]),
                     diagnostic)
            assert not any(LEFT_MARKER in t for t in sent_texts()[before:]), sent_texts()[before:]
    finally:
        model.release.set()
        for peer in (model, bot):
            peer.shutdown()
            peer.server_close()
