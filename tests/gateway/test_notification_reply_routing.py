"""Notification correlation through the real CLI sender and authenticated ingress."""
import argparse
import asyncio
import json
import sqlite3
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource
from hermes_state import SessionDB


CHAT = '15551234567@s.whatsapp.net'
ORIGIN = '20260101_120000_a1b2c3'


def send_notification(home, monkeypatch, *, wire_result=None, message='Publication needs approval',
                      expected_exit=0, patch_route=False, wire_exception=None):
    from tools import send_message_tool as send
    from hermes_cli.send_cmd import cmd_send
    from gateway.session_context import set_session_vars
    db = SessionDB(db_path=home / 'state.db')
    db.create_session(ORIGIN, source='desktop')
    db.close()
    set_session_vars(session_id=ORIGIN, source='desktop', profile='default')
    monkeypatch.setattr('gateway.config.load_gateway_config', lambda: GatewayConfig(
        platforms={Platform.WHATSAPP: PlatformConfig(enabled=True)}))
    wire = AsyncMock(
        return_value=wire_result if wire_result is not None else {'success': True, 'message_id': 'notice-1'},
        side_effect=wire_exception,
    )
    if patch_route:
        monkeypatch.setattr(send, '_send_to_platform', wire)
    else:
        monkeypatch.setitem(send._TEXT_SENDERS, 'whatsapp', wire)
    with pytest.raises(SystemExit) as exit_info:
        cmd_send(argparse.Namespace(to='whatsapp:' + CHAT, message=message,
                                   json=True, file=None, subject=None, quiet=False))
    assert exit_info.value.code == expected_exit
    if patch_route or wire_result is not None or expected_exit == 0:
        wire.assert_awaited_once()
    else:
        wire.assert_not_awaited()
    return home / 'notification-replies.db'


def test_cli_notification_retains_origin_across_connections(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    path = send_notification(tmp_path, monkeypatch)
    assert path.exists(), 'CLI send lost the originating desktop session and outbound message ID'
    with sqlite3.connect(path) as db:
        row = db.execute('SELECT session_id, chat_id FROM notification_routes WHERE message_id=?',
                         ('notice-1',)).fetchone()
    assert row == (ORIGIN, '15551234567')


def test_terminal_subprocess_send_captures_origin_without_contextvars(tmp_path, monkeypatch):
    import subprocess
    import sys
    from tools.environments.local import _make_run_env
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    send_notification(tmp_path, monkeypatch)
    env = _make_run_env({})
    assert env['HERMES_SESSION_ID'] == ORIGIN
    env['HOME'] = str(tmp_path)
    program = '''
import argparse
from gateway.config import GatewayConfig, Platform, PlatformConfig
import gateway.config
from tools import send_message_tool as send
from hermes_cli.send_cmd import cmd_send
gateway.config.load_gateway_config = lambda: GatewayConfig(platforms={Platform.WHATSAPP: PlatformConfig(enabled=True)})
async def wire(*args):
    return {'success': True, 'message_id': 'child-notice'}
send._TEXT_SENDERS['whatsapp'] = wire
cmd_send(argparse.Namespace(to='whatsapp:15551234567@s.whatsapp.net', message='Child approval notification', json=True))
'''
    result = subprocess.run([sys.executable, '-c', program], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload['reply_routing']['session_id'] == ORIGIN
    with sqlite3.connect(tmp_path / 'notification-replies.db') as db:
        assert db.execute('SELECT session_id FROM notification_routes WHERE message_id=?',
                          ('child-notice',)).fetchone()[0] == ORIGIN


@pytest.mark.parametrize('wire_result, expected_exit', [
    ({'error': 'Bridge timeout'}, 1), ({'success': True, 'message_id': None}, 0),
])
def test_uncertain_send_never_creates_an_executable_route(tmp_path, monkeypatch, wire_result, expected_exit):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    path = send_notification(tmp_path, monkeypatch, wire_result=wire_result, expected_exit=expected_exit)
    assert path.exists(), 'Send intent must be durable before touching the transport'
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT status FROM notification_routes').fetchone()[0] == 'uncertain'
    runner = inbound_runner(monkeypatch)
    result = asyncio.run(runner._handle_message(event(quote=None)))
    assert 'quote' in result.lower()
    runner._hm_pending_reply_intercepts.assert_not_awaited()
    result = asyncio.run(runner._handle_message(event(quote='unreturned-message-id')))
    assert 'correlate' in result.lower()
    runner._hm_pending_reply_intercepts.assert_not_awaited()


def test_same_native_reply_id_cannot_authorize_two_notifications(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    path = send_notification(tmp_path, monkeypatch)
    send_notification(tmp_path, monkeypatch, wire_result={'success': True, 'message_id': 'notice-2'})
    runner = inbound_runner(monkeypatch)
    asyncio.run(runner._handle_message(event()))
    result = asyncio.run(runner._handle_message(event(quote='notice-2')))
    assert 'queued' not in result.lower()
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT count(*) FROM notification_routes WHERE status='queued'").fetchone()[0] == 1


def test_distinct_followups_to_one_notification_are_each_queued(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    path = send_notification(tmp_path, monkeypatch)
    runner = inbound_runner(monkeypatch)

    first = asyncio.run(runner._handle_message(event(text='First answer')))
    followup = event(text='Additional detail')
    followup.message_id = 'reply-2'
    second = asyncio.run(runner._handle_message(followup))

    assert 'queued' in first.lower()
    assert 'queued' in second.lower()
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT reply_id, reply_text FROM notification_reply_queue ORDER BY created').fetchall() == [
            ('reply-1', 'First answer'), ('reply-2', 'Additional detail')]


@pytest.mark.parametrize('rotate_before_reply', [True, False])
def test_compression_rotated_owner_receives_reply_at_live_tip(tmp_path, monkeypatch, rotate_before_reply):
    import threading
    from tui_gateway.session_notification_replies import poll_replies
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    path = send_notification(tmp_path, monkeypatch)
    runner = inbound_runner(monkeypatch)
    db = SessionDB(db_path=tmp_path / 'state.db')
    if not rotate_before_reply:
        asyncio.run(runner._handle_message(event()))
    db.end_session(ORIGIN, 'compression')
    db.create_session('compressed-tip', source='desktop', parent_session_id=ORIGIN)
    db.close()
    if rotate_before_reply:
        asyncio.run(runner._handle_message(event()))
    session = {'session_key': 'compressed-tip', 'source': 'desktop', 'agent': object(),
               'history_lock': threading.Lock(), 'running': False}
    calls = []

    poll_replies('tip-tab', session, tmp_path,
                 lambda rid, sid, owner, text, **kwargs: calls.append((sid, owner, text)) or True)

    assert calls == [('tip-tab', session, 'Go ahead')]
    with sqlite3.connect(path) as route_db:
        assert route_db.execute('SELECT status FROM notification_reply_queue').fetchone()[0] == 'dispatched'


def test_explicitly_closed_owner_keeps_reply_queued_for_manual_resume(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    path = send_notification(tmp_path, monkeypatch)
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.end_session(ORIGIN, 'user_exit')
    db.close()

    result = asyncio.run(inbound_runner(monkeypatch)._handle_message(event()))

    assert 'open or resume' in result.lower()
    with sqlite3.connect(path) as route_db:
        assert route_db.execute('SELECT session_id, status FROM notification_reply_queue').fetchone() == (ORIGIN, 'queued')


def test_corrupt_routing_store_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    path = send_notification(tmp_path, monkeypatch)
    path.write_bytes(b'broken database')
    runner = inbound_runner(monkeypatch)
    result = asyncio.run(runner._handle_message(event()))
    assert 'unavailable' in result.lower()
    runner._hm_pending_reply_intercepts.assert_not_awaited()

    ordinary = event(text='How is the weather?', quote=None)
    result = asyncio.run(runner._handle_message(ordinary))
    assert result == 'ordinary chat'
    runner._hm_pending_reply_intercepts.assert_awaited_once()


@pytest.mark.parametrize('message', ['X' * 8000, 'MEDIA:/tmp/test-notification.png'])
def test_notification_payload_is_not_rejected_by_reply_routing(tmp_path, monkeypatch, message):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    media = tmp_path / 'test-notification.png'
    media.write_bytes(b'image')
    message = message.replace('/tmp/test-notification.png', str(media))
    path = send_notification(tmp_path, monkeypatch, message=message, patch_route=True,
                             wire_result={'success': True, 'message_id': 'notice-last',
                                          'message_ids': ['notice-first', 'notice-last']})
    with sqlite3.connect(path) as db:
        rows = db.execute("SELECT message_id, status FROM notification_routes "
                          "WHERE message_id NOT LIKE 'sending:%' ORDER BY message_id").fetchall()
    assert rows == [('notice-first', 'pending'), ('notice-last', 'pending')]


def test_partial_multipart_failure_correlates_delivered_ids_without_resend(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    pdf = tmp_path / 'report.pdf'
    pdf.write_bytes(b'%PDF-1.4')
    path = send_notification(
        tmp_path, monkeypatch, message=f'Report attached\nMEDIA:{pdf}', expected_exit=1, patch_route=True,
        wire_result={'error': 'second attachment failed', 'message_ids': ['text-id', 'pdf-id']})
    with sqlite3.connect(path) as db:
        rows = db.execute("SELECT message_id, status FROM notification_routes "
                          "WHERE message_id NOT LIKE 'sending:%' ORDER BY message_id").fetchall()
    assert rows == [('pdf-id', 'pending'), ('text-id', 'pending')]


def test_dispatch_exception_leaves_durable_uncertain_intent(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    path = send_notification(tmp_path, monkeypatch, expected_exit=1, patch_route=True,
                             wire_exception=RuntimeError('connection dropped after write'))
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT status FROM notification_routes').fetchone()[0] == 'uncertain'


def test_chunk_sender_retains_ids_before_partial_failure():
    from tools.send_message_tool import _send_chunks
    results = iter([
        {'success': True, 'message_id': 'chunk-1'},
        {'error': 'chunk 2 failed', 'message_ids': ['chunk-2-partial']},
    ])

    result = asyncio.run(_send_chunks(
        ['one', 'two'], lambda *_: _async_next(results), collect_message_ids=True))

    assert result['error'] == 'chunk 2 failed'
    assert result['message_ids'] == ['chunk-1', 'chunk-2-partial']


async def _async_next(iterator):
    return next(iterator)


def inbound_runner(monkeypatch):
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {}
    runner.session_store = None
    runner._scale_to_zero_note_real_inbound = lambda: None
    runner._hm_pending_reply_intercepts = AsyncMock(return_value='ordinary chat')
    monkeypatch.setenv('WHATSAPP_ALLOWED_USERS', CHAT)
    return runner


def event(text='Go ahead', quote='notice-1', sender=CHAT, chat=CHAT):
    return MessageEvent(text=text, message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.WHATSAPP, user_id=sender, chat_id=chat, chat_type='dm'),
        message_id='reply-1', reply_to_message_id=quote, reply_to_is_own_message=bool(quote))


def install_lid_alias(home, lid='999991234567890', phone='15551234567'):
    mapping_dir = home / 'whatsapp' / 'session'
    mapping_dir.mkdir(parents=True, exist_ok=True)
    (mapping_dir / f'lid-mapping-{lid}.json').write_text(json.dumps(phone), encoding='utf-8')
    (mapping_dir / f'lid-mapping-{phone}_reverse.json').write_text(json.dumps(lid), encoding='utf-8')


def test_phone_target_accepts_authenticated_lid_alias(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    install_lid_alias(tmp_path)
    path = send_notification(tmp_path, monkeypatch)
    lid = '999991234567890@lid'
    incoming = event(sender=lid, chat=lid)
    runner = inbound_runner(monkeypatch)
    monkeypatch.setenv('WHATSAPP_ALLOWED_USERS', '15551234567')

    result = asyncio.run(runner._handle_message(incoming))

    assert 'queued' in result.lower()
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT reply_id FROM notification_routes WHERE message_id=?',
                          ('notice-1',)).fetchone()[0] == 'reply-1'


def test_expired_routes_do_not_intercept_unquoted_approval(tmp_path, monkeypatch):
    from gateway.notification_replies import is_reply_candidate
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    path = send_notification(tmp_path, monkeypatch)
    with sqlite3.connect(path) as db:
        db.execute('UPDATE notification_routes SET created=?', (time.time() - 8 * 24 * 60 * 60,))
    incoming = event(quote=None)
    runner = inbound_runner(monkeypatch)

    assert is_reply_candidate(incoming) is False
    result = asyncio.run(runner._handle_message(incoming))

    assert result == 'ordinary chat'
    runner._hm_pending_reply_intercepts.assert_awaited_once()


def test_unknown_historical_approval_quote_is_not_current_chat_authority(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    send_notification(tmp_path, monkeypatch)
    runner = inbound_runner(monkeypatch)

    result = asyncio.run(runner._handle_message(event(quote='historical-message')))

    assert 'cannot correlate' in result.lower()
    assert 'approval' in result.lower()
    runner._hm_pending_reply_intercepts.assert_not_awaited()


@pytest.mark.parametrize('case, expected', [
    ('normal', 'queued'), ('unquoted', 'quote'), ('ordinary', 'ordinary chat'),
    ('replay', 'already'), ('deleted', 'unavailable'), ('stale', 'expired'),
    ('unsupported', 'originating session'), ('wrong-chat', 'cannot be verified'),
    ('unauthorized', None), ('attachment', 'text-only'), ('empty', 'text-only'),
])
def test_authenticated_reply_never_reaches_current_chat_controls(tmp_path, monkeypatch, case, expected):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    path = send_notification(tmp_path, monkeypatch)
    runner = inbound_runner(monkeypatch)
    incoming = event()
    if case == 'unquoted':
        incoming.reply_to_message_id = None
    if case == 'ordinary':
        incoming = event(text='What is the weather?', quote=None)
    if case == 'wrong-chat':
        incoming.source.chat_id = '15550000000@s.whatsapp.net'
    if case == 'unauthorized':
        incoming.source.user_id = '15550000000@s.whatsapp.net'
    if case == 'attachment':
        incoming.media_urls = ['/tmp/not-read.png']
    if case == 'empty':
        incoming.text = ''
    if case == 'deleted':
        with sqlite3.connect(tmp_path / 'state.db') as db:
            db.execute('DELETE FROM sessions WHERE id=?', (ORIGIN,))
    if case in ('stale', 'unsupported'):
        with sqlite3.connect(path) as db:
            if case == 'stale':
                db.execute('UPDATE notification_routes SET created=0')
            else:
                db.execute("UPDATE notification_routes SET source='telegram'")
    if case == 'replay':
        asyncio.run(runner._handle_message(incoming))
    result = asyncio.run(runner._handle_message(incoming))
    if expected is None:
        assert result is None
    else:
        assert expected in result.lower(), result
    if case == 'ordinary':
        runner._hm_pending_reply_intercepts.assert_awaited_once()
    else:
        runner._hm_pending_reply_intercepts.assert_not_awaited()
    with sqlite3.connect(path) as db:
        row = db.execute('SELECT status, reply_text FROM notification_routes WHERE message_id=?',
                         ('notice-1',)).fetchone()
    if case in ('normal', 'replay'):
        assert row == ('queued', 'Go ahead')
    else:
        assert row[1] is None


@pytest.mark.parametrize('quoted', [True, False])
def test_notification_reply_bypasses_busy_adapter_without_touching_its_task(tmp_path, monkeypatch, quoted):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    send_notification(tmp_path, monkeypatch)
    runner = inbound_runner(monkeypatch)
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={'dm_policy': 'allowlist'}))
    adapter._message_handler = runner._handle_message
    incoming = asyncio.run(adapter._build_message_event({
        'chatId': CHAT, 'senderId': CHAT, 'messageId': 'reply-1', 'body': 'Go ahead',
        'isGroup': False, 'hasQuotedMessage': quoted,
        'quotedMessageId': 'notice-1' if quoted else None, 'quotedText': 'Publication needs approval',
    }))
    assert incoming is not None
    assert incoming.reply_to_message_id == ('notice-1' if quoted else None)
    key = adapter._event_session_key(incoming)
    guard = object()
    adapter._active_sessions[key] = guard
    adapter._heal_stale_session_lock = lambda *a: None
    adapter._handle_message_while_active = AsyncMock()
    adapter._send_with_retry = AsyncMock()
    asyncio.run(adapter.handle_message(incoming))
    adapter._send_with_retry.assert_awaited_once()
    adapter._handle_message_while_active.assert_not_awaited()
    assert adapter._active_sessions[key] is guard


@pytest.mark.parametrize('case', ['busy', 'deleted', 'expired', 'other-session', 'other-profile', 'wrong-surface', 'refused', 'uncertain', 'replay'])
def test_owner_mailbox_fences_stale_foreign_and_replayed_work(tmp_path, monkeypatch, case):
    import threading
    from tui_gateway.session_notification_replies import poll_replies
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    path = send_notification(tmp_path, monkeypatch)
    asyncio.run(inbound_runner(monkeypatch)._handle_message(event()))
    session = {'session_key': ORIGIN, 'source': 'desktop', 'agent': object(),
               'history_lock': threading.Lock(), 'running': case == 'busy'}
    home = tmp_path
    if case == 'other-session':
        session['session_key'] = 'unrelated'
    if case == 'other-profile':
        home = tmp_path / 'other-profile'; home.mkdir()
    if case == 'wrong-surface':
        session['source'] = 'whatsapp'
    if case == 'deleted':
        with sqlite3.connect(tmp_path / 'state.db') as db:
            db.execute('DELETE FROM sessions WHERE id=?', (ORIGIN,))
    if case == 'expired':
        with sqlite3.connect(path) as db:
            db.execute('UPDATE notification_routes SET created=0')
    calls = []
    def submit(*args, **kwargs):
        calls.append(args)
        if case == 'uncertain':
            raise RuntimeError('dispatch outcome unknown')
        return case != 'refused'
    if case == 'uncertain':
        with pytest.raises(RuntimeError):
            poll_replies('tab', session, home, submit)
    else:
        poll_replies('tab', session, home, submit)
    if case in ('replay', 'uncertain'):
        session['running'] = False
        poll_replies('tab', session, home, submit)
    assert len(calls) == (1 if case in ('replay', 'refused', 'uncertain') else 0)
    with sqlite3.connect(path) as db:
        status = db.execute('SELECT status FROM notification_routes').fetchone()[0]
    if case == 'uncertain':
        assert status == 'dispatching'
    if case == 'refused':
        assert status == 'queued' and session['running'] is False
    if case in ('deleted', 'expired'):
        assert status == 'expired'


def test_desktop_owner_poller_continues_existing_surface_once(tmp_path, monkeypatch):
    import threading
    from tui_gateway import server
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(server, '_hermes_home', tmp_path)
    send_notification(tmp_path, monkeypatch)
    runner = inbound_runner(monkeypatch)
    asyncio.run(runner._handle_message(event()))
    original = {'session_key': ORIGIN, 'source': 'desktop', 'profile_home': str(tmp_path),
                'history_lock': threading.Lock(), 'running': False, 'agent': object(),
                'history': [{'role': 'assistant', 'content': 'Original work'}]}
    delivered = threading.Event()
    calls = []
    def submit(rid, sid, session, text, **kwargs):
        calls.append((sid, session, text))
        delivered.set()
        return True
    monkeypatch.setattr(server, '_run_prompt_submit', submit)
    monkeypatch.setattr(server, '_notif_poll_kanban', lambda *a: None)
    stop = threading.Event()
    thread = threading.Thread(target=server._notification_poller_loop,
                              args=(stop, 'original-tab', original), daemon=True)
    thread.start()
    try:
        assert delivered.wait(8), 'Owning desktop poller never received the queued WhatsApp reply'
    finally:
        stop.set()
        thread.join(5)
    assert calls == [('original-tab', original, 'Go ahead')]
    assert original['source'] == 'desktop'
    with sqlite3.connect(tmp_path / 'notification-replies.db') as db:
        assert db.execute('SELECT status FROM notification_routes').fetchone()[0] == 'dispatched'


def test_owner_mailbox_uses_real_prompt_admission_and_persists_visible_turn(tmp_path, monkeypatch):
    """Only the model call is substituted; owner admission, turn runner, events,
    session history, and SQLite persistence use their production implementations.
    """
    import threading
    from types import SimpleNamespace
    from tui_gateway import server
    from tui_gateway.session_notification_replies import poll_replies

    class InlineThread:
        def __init__(self, target=None, daemon=None, args=(), kwargs=None):
            self.target, self.args, self.kwargs = target, args, kwargs or {}

        def start(self):
            if self.target:
                self.target(*self.args, **self.kwargs)

        def is_alive(self):
            return False

        def join(self, timeout=None):
            return None

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.delenv('HERMES_PROFILE', raising=False)
    monkeypatch.setattr(server.threading, 'Thread', InlineThread)
    monkeypatch.setattr(server, '_wire_callbacks', lambda sid: None)
    monkeypatch.setattr(server, '_sync_agent_model_with_config', lambda sid, session: None)
    monkeypatch.setattr(server, '_sync_agent_compression_with_config', lambda sid, session: None)
    monkeypatch.setattr(server, '_sync_bot_capabilities', lambda sid, session: None)
    monkeypatch.setattr(server, '_session_cwd', lambda session: str(tmp_path))
    monkeypatch.setattr(server, '_register_session_cwd', lambda session: None)
    monkeypatch.setattr(server, '_tts_stream_begin', lambda: None)
    monkeypatch.setattr(server, '_sync_session_key_after_compress', lambda *args, **kwargs: None)
    monkeypatch.setattr(server, '_get_usage', lambda agent: {})
    emitted = []
    monkeypatch.setattr(server, '_emit', lambda kind, sid, payload=None: emitted.append((kind, sid, payload)))

    send_notification(tmp_path, monkeypatch)
    asyncio.run(inbound_runner(monkeypatch)._handle_message(event()))
    state = SessionDB(db_path=tmp_path / 'state.db')

    def deterministic_model_boundary(message, **kwargs):
        state.append_message(ORIGIN, 'user', kwargs['persist_user_message'])
        state.append_message(ORIGIN, 'assistant', 'Deterministic owner acknowledgement')
        return {'final_response': 'Deterministic owner acknowledgement', 'messages': [
            *kwargs['conversation_history'],
            {'role': 'user', 'content': kwargs['persist_user_message']},
            {'role': 'assistant', 'content': 'Deterministic owner acknowledgement'},
        ]}

    agent = SimpleNamespace(
        session_id=ORIGIN, model='test/model', provider='test',
        run_conversation=deterministic_model_boundary, clear_interrupt=lambda: None,
    )
    owner = {
        'session_key': ORIGIN, 'source': 'desktop',
        'history_lock': threading.Lock(), 'running': False, 'agent': agent,
        'history': [{'role': 'assistant', 'content': 'Original work'}],
        'history_version': 0, 'attached_images': [], 'image_counter': 0,
        'cols': 80, 'slash_worker': None, 'show_reasoning': False,
        'tool_progress_mode': 'all', 'inflight_turn': None,
    }
    server._sessions['real-owner-tab'] = owner
    try:
        poll_replies('real-owner-tab', owner, tmp_path, server._run_prompt_submit)
    finally:
        server._sessions.pop('real-owner-tab', None)
        server._release_active_session_slot(owner)
        state.close()

    assert [message['content'] for message in owner['history'][-2:]] == [
        'Go ahead', 'Deterministic owner acknowledgement']
    persisted = SessionDB(db_path=tmp_path / 'state.db')
    try:
        assert [(message['role'], message['content']) for message in persisted.get_messages(ORIGIN)[-2:]] == [
            ('user', 'Go ahead'), ('assistant', 'Deterministic owner acknowledgement')]
    finally:
        persisted.close()
    assert any(kind == 'message.start' and sid == 'real-owner-tab' for kind, sid, _ in emitted)
    assert any(kind == 'message.complete' and payload['text'] == 'Deterministic owner acknowledgement'
               for kind, _, payload in emitted)
    with sqlite3.connect(tmp_path / 'notification-replies.db') as db:
        assert db.execute('SELECT status FROM notification_reply_queue').fetchone()[0] == 'dispatched'
