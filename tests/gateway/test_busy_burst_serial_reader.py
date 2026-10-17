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
        await asyncio.wait({new}, timeout=0.5)
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
        done, _ = await asyncio.wait({second}, timeout=0.5)
        assert not done, 'the reader went on while the chat still had a burst committing'
    finally:
        release.set()
    await asyncio.wait_for(second, 10)
    assert await owner.settle(3) == [('', 1), ('', 1)]
