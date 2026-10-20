"""A serial receive loop's busy burst still merges, and the loop never waits out the hold window.

Signal (``_sse_listener``) and IRC (``_receive_loop``) await ``handle_message`` before reading their
next input. A busy input held for its burst window must therefore return at once; otherwise the
burst's next input arrives only after the burst committed alone (main returned once the input was
merged into the pending slot). The reader still never runs ahead of what it returned on: a command
waits for those inputs to commit, and a chat holds at most one uncommitted burst. Real
GatewayRunner, adapter, SessionAuthority and drain over SQLite; the running turn is a stand-in.
"""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session_authority import initialize_session_authority
from gateway.session_ingress_burst import holds_busy_burst
from gateway.session_results import execution_result
from hermes_state_runtime import list_session_admissions
from tests.waiting import async_wait_until


def _rows(authority):
    return [row for sid in list(authority.sessions)
            for row in list_session_admissions(authority.db, session_id=sid, pending_only=False)]


async def _busy_owner(monkeypatch, adapter, source, **wiring):
    """Wire ``adapter`` to a real runner and authority, then start turn 'A' and hold it running."""
    runner = GatewayRunner(GatewayConfig())

    async def send(chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id='sent')
    monkeypatch.setattr(adapter, 'send', send)
    monkeypatch.setattr(adapter, 'send_typing', lambda *a, **k: asyncio.sleep(0))
    adapter.gateway_runner = runner
    runner.adapters = {adapter.platform: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter, **wiring)
    running, finish, executed = asyncio.Event(), asyncio.Event(), []

    async def turn(event, source, quick_key, run_generation):
        executed.append((event.text, len(event.media_urls)))
        if event.text == 'A':
            running.set()
            await finish.wait()
        execution_result.get()['result'] = {'final_response': 'ok', 'messages': []}
        return 'ok'
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
    await adapter.handle_message(MessageEvent(text='A', source=source, message_id='A'))
    await asyncio.wait_for(running.wait(), 10)
    key = adapter._event_session_key(MessageEvent(text='', source=source))

    async def settle(turns):
        finish.set()
        await async_wait_until(lambda: not holds_busy_burst(adapter, key) and len(executed) >= turns and all(
            row['status'] == 'terminal' for row in _rows(authority)), timeout=15, what='the follow-ups to settle')
        await asyncio.to_thread(authority.db.close)
        return executed[1:]
    return SimpleNamespace(runner=runner, authority=authority, key=key, settle=settle)


def _signal(monkeypatch):
    from plugins.platforms.signal.adapter import SignalAdapter
    monkeypatch.setenv('SIGNAL_ALLOWED_USERS', '+15550001')
    adapter = SignalAdapter(PlatformConfig(enabled=True, extra={'account': '+15559999'}))
    return adapter, adapter.build_source(chat_id='+15550001', chat_type='dm', user_id='+15550001')


def _photo(source, tmp_path, n):
    path = tmp_path / f'p{n}.png'
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + bytes(24) + bytes([n]))
    return MessageEvent(text='', message_type=MessageType.PHOTO, source=source, message_id=f'P{n}',
                        media_urls=[str(path)], media_types=['image/png'])


def _hold_capture(monkeypatch):
    """The real media capture, held on an Event: a slow disk under the burst's commit."""
    import gateway.session_envelope as envelope
    entered, release, real = threading.Event(), threading.Event(), envelope.capture_native_media

    def held(paths):
        entered.set()
        release.wait(10)
        return real(paths)
    monkeypatch.setattr(envelope, 'capture_native_media', held)
    return entered, release


@pytest.mark.asyncio
async def test_signal_busy_photos_merge_without_holding_the_reader(monkeypatch, tmp_path):
    adapter, source = _signal(monkeypatch)
    owner = await _busy_owner(monkeypatch, adapter, source)
    for n in (1, 2):  # as _sse_listener does: the next envelope only after handle_message returned
        await adapter.handle_message(_photo(source, tmp_path, n))
        assert owner.key in adapter._shared_bursts, 'the reader waited out the hold window'
    assert await owner.settle(2) == [('', 2)]


@pytest.mark.asyncio
async def test_irc_split_text_merges_through_the_receive_loop(monkeypatch):
    from gateway import platform_registry
    from gateway.platform_registry import PlatformEntry, PlatformRegistry
    from tests.gateway._plugin_adapter_loader import load_plugin_adapter
    monkeypatch.setenv('IRC_ALLOWED_USERS', 'sender')
    module = load_plugin_adapter('irc')
    registry = PlatformRegistry()
    monkeypatch.setattr(platform_registry, 'platform_registry', registry)
    module.register(SimpleNamespace(register_platform=lambda **kw: registry.register(PlatformEntry(**kw))))
    adapter = module.IRCAdapter(PlatformConfig(enabled=True, typing_indicator=False, extra={
        'server': 'irc.example.invalid', 'nickname': 'testbot', 'allowed_users': ['sender']}))
    source = adapter.build_source(chat_id='sender', chat_name='sender', chat_type='dm',
                                  user_id='sender', user_name='sender')
    owner = await _busy_owner(monkeypatch, adapter, source, busy_text_mode='queue')
    adapter._reader = asyncio.StreamReader()
    adapter._reader.feed_data(b':sender!u@h PRIVMSG testbot :first half\r\n'
                              b':sender!u@h PRIVMSG testbot :second half\r\n')
    adapter._reader.feed_eof()
    await asyncio.wait_for(adapter._receive_loop(), 10)
    assert await owner.settle(2) == [('first half\nsecond half', 0)]


@pytest.mark.asyncio
async def test_restart_after_a_held_text_waits_for_its_commit(monkeypatch):
    """A text, then an authorized /restart inside the burst window: the text is queued before
    admission closes, so the next owner runs it."""
    adapter, source = _signal(monkeypatch)
    owner = await _busy_owner(monkeypatch, adapter, source, busy_text_mode='queue')
    monkeypatch.setattr(owner.runner, 'request_restart', lambda **kw: setattr(owner.runner, '_draining', True))
    await adapter.handle_message(MessageEvent(text='B', source=source, message_id='B'))
    await adapter.handle_message(MessageEvent(text='/restart', source=source, message_id='R'))
    await async_wait_until(lambda: not holds_busy_burst(adapter, owner.key), timeout=5, what='the burst to land')
    assert [row['request_id'] for row in _rows(owner.authority)] == ['A', 'B']
    owner.runner._draining = False  # the stand-in for the next owner
    assert await owner.settle(2) == [('B', 0)]


@pytest.mark.asyncio
async def test_new_waits_for_a_burst_still_committing(monkeypatch, tmp_path):
    """A photo still capturing when /new arrives commits first, so /new cancels it with the old
    conversation instead of it running in the new one."""
    adapter, source = _signal(monkeypatch)
    entered, release = _hold_capture(monkeypatch)
    owner = await _busy_owner(monkeypatch, adapter, source)
    try:
        await adapter.handle_message(_photo(source, tmp_path, 1))
        assert await asyncio.to_thread(entered.wait, 5)
        new = asyncio.create_task(adapter.handle_message(MessageEvent(text='/new', source=source, message_id='N')))
        await asyncio.wait({new}, timeout=2.0)  # a negative window (>= 2 s): main's /new would land here
    finally:
        release.set()
    await asyncio.wait_for(new, 10)
    assert await owner.settle(1) == []


@pytest.mark.asyncio
async def test_next_burst_waits_while_the_chats_last_one_commits(monkeypatch, tmp_path):
    """The chat's first burst is still committing: the reader waits before holding another one,
    as main's reader waited at the first commit, so uncommitted input stays bounded."""
    adapter, source = _signal(monkeypatch)
    entered, release = _hold_capture(monkeypatch)
    owner = await _busy_owner(monkeypatch, adapter, source)
    try:
        await adapter.handle_message(_photo(source, tmp_path, 1))
        assert await asyncio.to_thread(entered.wait, 5)
        second = asyncio.create_task(adapter.handle_message(_photo(source, tmp_path, 2)))
        done, _ = await asyncio.wait({second}, timeout=2.0)  # a negative window: >= 2 s per tests/AGENTS.md
        assert not done, 'the reader went on while the chat still had a burst committing'
    finally:
        release.set()
    await asyncio.wait_for(second, 10)
    assert await owner.settle(3) == [('', 1), ('', 1)]


@pytest.mark.asyncio
async def test_restart_waits_for_a_held_text_queued_behind_a_slow_capture(monkeypatch, tmp_path):
    """Idle video V (slow capture), then queue-mode text B (held as a burst, then queued behind V's
    route reservation, which lets the serial reader go on), then an authorized /restart. The command
    barrier must wait until B is durably committed, not only until its FIFO position is fixed: admission
    closes after it, and an input with no row then is lost for good."""
    adapter, source = _signal(monkeypatch)
    entered, release = _hold_capture(monkeypatch)
    runner = GatewayRunner(GatewayConfig())

    async def send(chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id='sent')
    monkeypatch.setattr(adapter, 'send', send)
    monkeypatch.setattr(adapter, 'send_typing', lambda *a, **k: asyncio.sleep(0))
    adapter.gateway_runner = runner
    runner.adapters = {adapter.platform: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter, busy_text_mode='queue')
    monkeypatch.setattr(runner, 'request_restart', lambda **kw: setattr(runner, '_draining', True))

    async def turn(event, source, quick_key, run_generation):
        execution_result.get()['result'] = {'final_response': 'ok', 'messages': []}
        return 'ok'
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
    clip = tmp_path / 'clip.mp4'
    clip.write_bytes(b'video-bytes' * 1000)
    video = MessageEvent(text='', message_type=MessageType.VIDEO, source=source, message_id='V',
                         media_urls=[str(clip)], media_types=['video/mp4'])
    restart = None
    try:
        await adapter.handle_message(video)
        assert await asyncio.to_thread(entered.wait, 5)
        await adapter.handle_message(MessageEvent(text='B', source=source, message_id='B'))
        restart = asyncio.create_task(adapter.handle_message(
            MessageEvent(text='/restart', source=source, message_id='R')))
        await async_wait_until(lambda: any(entry[1] >= 2 for entry in authority._route_turns.values()),
                               timeout=5, what='B queued behind the capturing V')
        await asyncio.wait({restart}, timeout=2.0)  # a negative window: the restart must still wait here
    finally:
        release.set()
    await asyncio.wait_for(restart, 15)
    assert sorted(row['request_id'] for row in _rows(authority)) == ['B', 'V'], _rows(authority)
    runner._draining = False
    await async_wait_until(lambda: all(row['status'] == 'terminal' for row in _rows(authority)), timeout=15,
                           what='both turns to run')
    await asyncio.to_thread(authority.db.close)


@pytest.mark.asyncio
async def test_restart_waits_for_a_direct_busy_input_queued_behind_a_slow_capture(monkeypatch, tmp_path):
    """The same with the default busy mode: the text is no burst, so it goes straight to the shared
    runtime, queues behind the capture and releases the reader before it is durable. With no burst commit
    to wait on, the /restart it reads next must still wait for that row (Signal live: both lost)."""
    from gateway import session_ingress_burst
    from gateway.session_authority import SessionAuthority
    from gateway.session_runtime_workers import mutation_tasks
    adapter, source = _signal(monkeypatch)
    entered, release = _hold_capture(monkeypatch)
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)  # no drain: rows stay queued
    runner = GatewayRunner(GatewayConfig())

    async def send(chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id='sent')
    monkeypatch.setattr(adapter, 'send', send)
    adapter.gateway_runner = runner
    runner.adapters = {adapter.platform: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter)  # no text burst: B goes straight to dispatch_shared_busy
    monkeypatch.setattr(runner, 'request_restart', lambda **kw: setattr(runner, '_draining', True))
    passed, real_gate = [], session_ingress_burst.burst_gate

    async def gate(adapter, event, session_key):  # the rows when the command passes the reader's barrier
        behind = await real_gate(adapter, event, session_key)
        if event.get_command() == 'restart':
            passed.append(sorted(row['request_id'] for row in _rows(authority)))
        return behind
    monkeypatch.setattr(session_ingress_burst, 'burst_gate', gate)
    clip = tmp_path / 'clip.mp4'
    clip.write_bytes(b'video bytes' * 100)
    key = runner._session_key_for_source(source)
    try:
        await adapter.handle_message(MessageEvent(text='A', source=source, message_id='A', message_type=MessageType.VIDEO,
                                                  media_urls=[str(clip)], media_types=['video/mp4']))
        assert await asyncio.to_thread(entered.wait, 5)
        await adapter.handle_message(MessageEvent(text='B', source=source, message_id='B'))
        restart = asyncio.create_task(adapter.handle_message(MessageEvent(text='/restart', source=source,
                                                                           message_id='R')))
        await async_wait_until(lambda: passed or authority._route_turns.get(key, (None, 0))[1] >= 2,
                               what='B to queue behind the capture')
    finally:
        release.set()
    await asyncio.wait_for(restart, 10)
    try:
        assert passed == [['A', 'B']], passed
    finally:
        runner._draining = False
        await adapter.cancel_background_tasks()
        await async_wait_until(lambda: not mutation_tasks(authority), timeout=10)
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_buzz_split_text_merges_through_the_reader(monkeypatch):
    """Buzz was left out of the #140973 serial-reader sweep: both the WS listener and the poll
    sweep await ``_handle_events`` before reading the next batch (#141378)."""
    adapter, state, source = _buzz(monkeypatch)
    owner = await _busy_owner(monkeypatch, adapter, source, busy_text_mode='queue')
    # as both readers do: the next batch only after the previous _handle_events returned
    await adapter._handle_events('chan', state, [_buzz_event(1, 'first half')])
    await adapter._handle_events('chan', state, [_buzz_event(2, 'second half')])
    assert await owner.settle(2) == [('first half\nsecond half', 0)]


def _buzz(monkeypatch):
    from collections import OrderedDict

    from plugins.platforms.buzz.adapter import BuzzAdapter

    monkeypatch.setenv('BUZZ_ALLOWED_USERS', 'aa11bb22')
    adapter = BuzzAdapter(PlatformConfig(enabled=True))
    async def _name(pubkey):
        return pubkey[:8]
    adapter._resolve_user_name = _name  # the CLI-backed resolver has no place in a unit test
    state = {"chat_type": "dm", "last_ts": 0, "seen": OrderedDict(), "event_meta": OrderedDict()}
    adapter._channel_state['chan'] = state
    return adapter, state, adapter.build_source(chat_id='chan', chat_type='dm', user_id='aa11bb22')


def _buzz_event(i, text):
    return {'id': f'e{i}', 'kind': 9, 'pubkey': 'aa11bb22', 'content': text, 'created_at': i}


@pytest.mark.asyncio
async def test_buzz_cursor_never_passes_an_uncommitted_input(monkeypatch):
    """Buzz saves its receive cursor after ``_handle_events`` returns, deferred while a busy burst is
    held. The reader marks the next event seen before dispatching it, so a write that lands in that gap
    must still name only committed inputs: a restart restores the saved cursor and skips the rest."""
    adapter, state, source = _buzz(monkeypatch)
    owner = await _busy_owner(monkeypatch, adapter, source, busy_text_mode='queue')
    saved = []
    monkeypatch.setattr(type(adapter), '_write_cursors', staticmethod(lambda path, payload: saved.append(
        (payload['channels']['chan']['last_ts'], {row['request_id'] for row in _rows(owner.authority)}))))
    await adapter._handle_events('chan', state, [_buzz_event(1, 'held')])
    assert owner.key in adapter._shared_bursts and saved == []  # returned with the input only held
    gate, real_name = asyncio.Event(), adapter._resolve_user_name

    async def slow_name(pubkey):  # e2 is marked seen, not yet dispatched, while the burst commits
        await gate.wait()
        return await real_name(pubkey)
    adapter._resolve_user_name = slow_name
    second = asyncio.create_task(adapter._handle_events('chan', state, [_buzz_event(2, 'next')]))
    await async_wait_until(lambda: saved, timeout=15, what='the deferred cursor write')
    gate.set()
    await asyncio.wait_for(second, 10)
    assert await owner.settle(3) == [('held', 0), ('next', 0)]
    assert saved[0][0] == 1, saved
    # Every saved cursor names only events that had an admission row when it was written.
    assert all(f'e{n}' in rows for last_ts, rows in saved for n in range(1, last_ts + 1)), saved


@pytest.mark.asyncio
async def test_buzz_cursor_write_waiting_on_disk_skips_a_newer_held_burst(monkeypatch):
    """e1's deferred write is due but waits behind a slow earlier write; meanwhile e2 arrives and is
    held. The waiting write must still land the cursor as it was when e1 committed, not e2's."""
    adapter, state, source = _buzz(monkeypatch)
    owner = await _busy_owner(monkeypatch, adapter, source, busy_text_mode='queue')
    saved = []
    monkeypatch.setattr(type(adapter), '_write_cursors', staticmethod(lambda path, payload: saved.append(
        (payload['channels']['chan']['last_ts'], {row['request_id'] for row in _rows(owner.authority)}))))
    await adapter._cursor_write_lock.acquire()  # the earlier write, still in its fsync
    await adapter._handle_events('chan', state, [_buzz_event(1, 'held')])
    await async_wait_until(lambda: adapter._cursor_write_lock._waiters, timeout=15,
                           what="e1's committed write to queue on the lock")
    await adapter._handle_events('chan', state, [_buzz_event(2, 'next')])
    assert owner.key in adapter._shared_bursts  # e2 is held, not committed
    adapter._cursor_write_lock.release()
    await async_wait_until(lambda: saved, timeout=15, what='the waiting cursor write')
    assert saved[0][0] == 1, saved
    assert await owner.settle(3) == [('held', 0), ('next', 0)]
    assert all(f'e{n}' in rows for last_ts, rows in saved for n in range(1, last_ts + 1)), saved
