"""A messaging user whose session FIFO is paused is told once, and can /reset out of it.

An owner crash mid-turn leaves the started admission ``unknown`` and the session paused
(fail-closed, never weakened here). Before this fix the next Telegram message was admitted,
``admit_message`` awaited it forever and the platform user saw nothing at all.
"""
import asyncio
from contextlib import closing
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import queue
import sqlite3
import threading
import time

import pytest

from tests.gateway.fixtures.local_recovery_probe import Model, child_env, daemon
from tests.gateway.test_native_telegram_startup_recovery import BotAPI, wait_for

PAUSE_MARKER = 'This conversation is paused'


@pytest.mark.asyncio
async def test_paused_fifo_releases_the_messaging_waiter_with_one_notice_per_episode(tmp_path, monkeypatch):
    """No daemon: the authority alone. A refused head pauses the FIFO; every messaging waiter
    parked on that session returns (never hangs), the first gets the notice, the rest None,
    and once the FIFO moves again the next pause is a fresh episode with its own notice."""
    from gateway.platforms.event import MessageEvent
    from gateway.session_authority import SessionAuthority
    from gateway.session_contract import Principal, Submission
    from gateway.session_controls import AuthorityConnection
    from gateway.session_ingress import admit_message
    from tests.gateway.test_authority_unknown_resolution import Peer, _restarted_owner_with_unknown_head

    executed = []
    store, authority, ref, unknown, follower = await _restarted_owner_with_unknown_head(tmp_path, monkeypatch, executed)
    try:
        actor = Principal('human', 'p', frozenset({'session:submit'}), 'fixture')

        async def admit_native(event):
            return await authority.submit(actor, Submission(event.text, ref, {'text': event.text}, 'queue'))
        monkeypatch.setattr(authority, 'admit_native', admit_native)
        authority._schedule = lambda ref: SessionAuthority._schedule(authority, ref)
        source = authority.sessions['s'].source
        first = asyncio.create_task(admit_message(authority, MessageEvent(text='one', source=source)))
        second = asyncio.create_task(admit_message(authority, MessageEvent(text='two', source=source)))
        replies = await asyncio.wait_for(asyncio.gather(first, second), 5)
        assert sum(r is not None and PAUSE_MARKER in r for r in replies) == 1, replies
        assert executed == []
        # A restart drain is transient: the saved row runs on the next owner, so the notice must
        # not tell the user to /reset (that would discard the very message that is about to run).
        from gateway.session_ingress import pause_notice
        authority.sessions['s'].pause_notified = False
        draining = pause_notice(authority, ref, 'runtime_draining')
        assert 'restarting' in draining and '/reset' not in draining, draining
        authority.sessions['s'].pause_notified = False
        # The operator resumes the FIFO; the saved followers run, no new notice was owed.
        viewer = AuthorityConnection(authority, Peer(), {'user_id': 'human'})
        await viewer.dispatch({'id': 1, 'method': 'prompt.resolve_unknown', 'params': {
            'session_id': 's', 'admission_id': unknown['admission_id'], 'execution_generation': unknown['generation']}})
        await asyncio.wait_for(authority.sessions['s'].task, 5)
        assert executed == ['FOLLOWER', 'one', 'two']
        assert not authority.sessions['s'].pause_notified
        await viewer.close()
    finally:
        store.close_all_db_handles()


@pytest.mark.linux_only
def test_crash_mid_turn_then_platform_message_gets_one_pause_notice_and_reset_unpauses(tmp_path):
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
        # Telegram MarkdownV2 escapes underscores; compare on the unescaped text.
        return [m.get('text', '').replace('\\', '') for m in bot.sent]

    def notices():
        return [t for t in sent_texts() if PAUSE_MARKER in t]

    def diagnostic():
        return repr(rows()) + '\n' + repr(sent_texts()) + '\n' + '\n'.join(
            p.read_text(encoding='utf-8') for p in (home / 'logs').glob('*.log'))

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
        with daemon(root, home, env, barrier=False) as (proc, desc):
            send('WARM_HISTORY', 'terminal')
            send('BLOCK_STARTED', 'started')
            assert model.blocked.wait(20), diagnostic()
            proc.kill()
            assert proc.wait(timeout=10) == -9
        model.release.set()
        with daemon(root, home, env, barrier=False) as (proc, desc):
            wait_for(lambda: rows()['BLOCK_STARTED']['status'] == 'unknown', diagnostic)
            paused_session = rows()['BLOCK_STARTED']['session']
            # First message onto the paused FIFO: admitted (durable), the user is told once,
            # and the adapter loop keeps going (the second message is admitted too).
            send('AFTER_CRASH_ONE', 'queued')
            wait_for(lambda: len(notices()) == 1, diagnostic)
            assert '/reset' in notices()[0], notices()
            send('AFTER_CRASH_TWO', 'queued')
            assert rows()['AFTER_CRASH_TWO']['session'] == paused_session
            # The named escape works from the platform: /reset (confirmed) moves the chat to
            # a fresh session and the next message runs and is answered.
            push('/reset')
            wait_for(lambda: any('Confirm /new' in t for t in sent_texts()), diagnostic)
            push('/approve')
            wait_for(lambda: any('Session reset' in t or 'New session' in t for t in sent_texts()), diagnostic)
            send('AFTER_RESET', 'terminal')
            wait_for(lambda: any('RECOVERY_ACK_AFTER_RESET' in t for t in sent_texts()), diagnostic)
            after = rows()
            assert after['AFTER_RESET']['session'] != paused_session
            # Exactly one notice for the whole pause episode, and the pause itself is intact:
            # the lost turn stays unknown and its followers never ran on their own.
            assert len(notices()) == 1, notices()
            assert after['BLOCK_STARTED']['status'] == 'unknown'
            assert after['AFTER_CRASH_ONE']['status'] == after['AFTER_CRASH_TWO']['status'] == 'queued'
        texts = [next(m['content'] for m in reversed(r['messages']) if m['role'] == 'user') for r in model.requests]
        assert not any('AFTER_CRASH' in t for t in texts), texts
        print(json.dumps({'notice': notices()[0], 'final_rows': rows()}))
    finally:
        model.release.set()
        for peer in (model, bot):
            peer.shutdown()
            peer.server_close()
