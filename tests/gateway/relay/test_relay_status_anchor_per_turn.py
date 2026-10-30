"""A relay Slack turn's "is working" status stays on that turn's own message.

Slack's status line is thread-scoped. A top-level DM turn has no thread in its metadata, so the
relay adapter anchors the status on the triggering message. That anchor must be fixed when the turn
starts: resolved later from per-chat state (the chat's latest inbound message), a second top-level
message or a ``/stop`` sent while the turn runs moves the running turn's heartbeats and its final
clear onto the newer message, and a turn that ends without a reply leaves its own status set.

The composed tests use a real ``GatewayRunner`` + session authority + ``RelayAdapter`` +
``WebSocketRelayTransport`` against a fixture connector; only the model call
(``_handle_message_with_agent``) is stubbed. ``_left_working`` replays the wire frames the way
Slack renders them.
"""
import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session_results import execution_result
from tests.waiting import async_wait_until

websockets = pytest.importorskip('websockets')

FLAT_DM_SESSION = {'slack': {'reply_in_thread': True, 'dm_top_level_threads_as_sessions': False}}
DM_THREAD_SESSIONS = {'slack': {'reply_in_thread': True, 'dm_top_level_threads_as_sessions': True}}
FLAT_REPLIES = {'slack': {'reply_in_thread': False}}


@asynccontextmanager
async def _gateway(tmp_path, instance, extra):
    """Yields ``(runner, send, outgoing, adapter)``; ``send(message_id, chat_id, text, ...)`` delivers
    one inbound Slack message through the fixture connector. A chat message runs as a producer
    admission (``session_producer_lifecycle``), a command through ``_process_message_background``."""
    from gateway.relay.adapter import RelayAdapter
    from gateway.relay.descriptor import CapabilityDescriptor
    from gateway.relay.ws_transport import WebSocketRelayTransport
    from gateway.session_authority import initialize_session_authority

    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    authority = await initialize_session_authority(runner, profile_id='default', instance_id=instance)
    outgoing, ready = [], asyncio.Queue()
    descriptor = CapabilityDescriptor(1, 'slack', 'Fixture', 4096, False, True, True, 'plain', 'chars',
                                      supported_ops=('send', 'edit', 'typing', 'react'))

    async def connector(ws):
        await ws.recv()  # hello
        await ws.send(json.dumps({'type': 'descriptor', 'descriptor': json.loads(descriptor.to_json())}) + '\n')
        await ready.put(ws)
        async for line in ws:
            frame = json.loads(line)
            outgoing.append(frame)
            if frame['type'] == 'outbound':
                await ws.send(json.dumps({'type': 'outbound_result', 'requestId': frame['requestId'],
                                          'result': {'success': True, 'message_id': 'sent'}}) + '\n')

    try:
        async with websockets.serve(connector, '127.0.0.1', 0) as server:
            transport = WebSocketRelayTransport(
                f'ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/relay', 'slack', 'fixture-bot',
                gateway_id='fixture-gateway', upgrade_secret='fixture-secret', reconnect=False)
            adapter = RelayAdapter(PlatformConfig(enabled=True, extra=extra), descriptor, transport)
            adapter.gateway_runner = runner
            runner.adapters[Platform.RELAY] = adapter
            runner._wire_adapter_handlers(adapter)
            try:
                assert await adapter.connect()
                ws = await asyncio.wait_for(ready.get(), 5)

                async def send(message_id, chat_id, text, *, thread_id=None, chat_type='dm'):
                    source = {'platform': 'slack', 'chat_type': chat_type, 'chat_id': chat_id,
                              'user_id': 'U-' + chat_id, 'scope_id': 'T1'}
                    if thread_id is not None:
                        source['thread_id'] = thread_id
                    event = {'text': text, 'message_id': message_id, 'source': source}
                    await ws.send(json.dumps({'type': 'inbound', 'event': event, 'bufferId': 'buf-' + message_id})
                                  + '\n')

                yield runner, send, outgoing, adapter
            finally:
                await adapter.disconnect()
    finally:
        authority.db.close()


def _actions(outgoing):
    return [f['action'] for f in outgoing if f.get('type') == 'outbound']


def _thread(action):
    return (action.get('metadata') or {}).get('thread_id')


def _heartbeats(outgoing):
    return [a for a in _actions(outgoing) if a.get('op') == 'typing' and a.get('content') != '']


def _left_working(actions):
    """Threads Slack still shows "is working" on after these outbound actions: a heartbeat sets the
    thread's status, an empty typing frame clears it, and the app posting in the thread clears it too."""
    working = {}
    for action in actions:
        thread = _thread(action)
        if thread is None:
            continue
        if action.get('op') == 'typing':
            working[thread] = action.get('content') != ''
        elif action.get('op') in ('send', 'send_media'):
            working[thread] = False
    return {thread for thread, on in working.items() if on}


def _typing_refreshes():
    return [task for task in asyncio.all_tasks() if not task.done()
            and getattr(task.get_coro(), '__qualname__', '').endswith('._keep_typing')]


def _record(result):
    """The turn's execution result, for the authority's settlement."""
    captured = execution_result.get()
    if captured is not None:
        captured['result'] = result


def _stub_model(runner, monkeypatch, behaviours):
    executed = []

    async def turn(event, *_args, **_kwargs):
        executed.append(event.text)
        behaviour = behaviours.get(event.text)
        if behaviour is not None:
            return await behaviour(event)
        _record({'final_response': 'reply ' + event.text, 'messages': []})
        return 'reply ' + event.text

    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
    return executed


async def _settle(outgoing):
    await async_wait_until(lambda: not _typing_refreshes(), timeout=10, what='every typing refresh stopped')
    seen = [-1]

    def quiet():  # the last clear frames, sent after the refresh ended, have arrived
        before, seen[0] = seen[0], len(outgoing)
        return before == seen[0]

    await async_wait_until(quiet, timeout=10, interval=0.3, what='the connector stopped receiving frames')


async def _overlapping_dm_turns_then_stop(tmp_path, monkeypatch, instance, extra):
    """The report's steps: a long top-level DM turn, a second message while it runs, then ``/stop``.
    The long turn ends without posting a reply (as an interrupted turn does)."""
    stopped = asyncio.Event()

    async def long_job(_event):
        await stopped.wait()
        _record({'final_response': '', 'messages': [], 'interrupted': True})

    async with _gateway(tmp_path, instance, extra) as (runner, send, outgoing, _adapter):
        executed = _stub_model(runner, monkeypatch, {'list the largest files': long_job})
        await send('1700000000.000100', 'D1', 'list the largest files')
        await async_wait_until(lambda: any(_thread(a) == '1700000000.000100' for a in _heartbeats(outgoing)),
                               timeout=10, what='the long turn shows its status')
        beats = len(_heartbeats(outgoing))
        await send('1700000000.000200', 'D1', 'also check /tmp')
        # The running turn keeps refreshing its status after the second message arrived.
        await async_wait_until(lambda: len(_heartbeats(outgoing)) > beats + 1, timeout=10,
                               what='a heartbeat after the second message')
        beats = len(_heartbeats(outgoing))
        await send('1700000000.000300', 'D1', '/stop')
        await async_wait_until(lambda: len(_heartbeats(outgoing)) > beats + 1, timeout=10,
                               what='a heartbeat after /stop')
        stopped.set()
        await async_wait_until(lambda: 'list the largest files' in executed, timeout=10, what='the long turn ran')
        await _settle(outgoing)
        return outgoing


@pytest.mark.asyncio
async def test_stop_after_a_second_top_level_dm_leaves_no_status_in_a_flat_dm_session(tmp_path, monkeypatch):
    """The reported case: reply_in_thread true, dm_top_level_threads_as_sessions false."""
    outgoing = await _overlapping_dm_turns_then_stop(
        tmp_path, monkeypatch, 'status-anchor-flat-session', FLAT_DM_SESSION)
    assert _left_working(_actions(outgoing)) == set()
    # The long turn's last frame is the clear on its own message; the queued message (dropped by
    # /stop, never run) never showed a status.
    first = [a for a in _actions(outgoing) if a.get('op') == 'typing' and _thread(a) == '1700000000.000100']
    assert first and first[-1].get('content') == ''
    assert not [a for a in _heartbeats(outgoing) if _thread(a) == '1700000000.000200']


@pytest.mark.asyncio
async def test_stop_after_a_follow_up_leaves_no_status_with_dm_threads_as_sessions(tmp_path, monkeypatch):
    """dm_top_level_threads_as_sessions true: each top-level DM keys its own session and its status
    metadata names its thread, so the same steps never reached the per-chat anchor."""
    outgoing = await _overlapping_dm_turns_then_stop(
        tmp_path, monkeypatch, 'status-anchor-thread-sessions', DM_THREAD_SESSIONS)
    assert _left_working(_actions(outgoing)) == set()


@pytest.mark.asyncio
async def test_queued_turns_each_show_and_clear_their_own_status(tmp_path, monkeypatch):
    """Two messages queue behind a running turn in one DM session. Each turn that answers clears its own
    status, and a queued turn shows its status on its own message when it starts, even though a newer
    message already reached the chat by then."""
    release = asyncio.Event()

    async def held(_event):
        await release.wait()
        _record({'final_response': 'first done', 'messages': []})
        return 'first done'

    async with _gateway(tmp_path, 'status-anchor-queued', FLAT_DM_SESSION) as (runner, send, outgoing, _adapter):
        executed = _stub_model(runner, monkeypatch, {'first': held})
        await send('1700000000.000400', 'D4', 'first')
        await async_wait_until(lambda: any(_thread(a) == '1700000000.000400' for a in _heartbeats(outgoing)),
                               timeout=10, what='the first turn shows its status')
        beats = len(_heartbeats(outgoing))
        await send('1700000000.000401', 'D4', 'second')
        await send('1700000000.000402', 'D4', 'third')
        await async_wait_until(lambda: len(_heartbeats(outgoing)) > beats + 1, timeout=10,
                               what='a heartbeat after the queued messages')
        release.set()
        await async_wait_until(lambda: executed == ['first', 'second', 'third'], timeout=20, what='all turns ran')
        await _settle(outgoing)
        actions = _actions(outgoing)
        assert _left_working(actions) == set()
        replies = [next(i for i, a in enumerate(actions) if a.get('op') == 'send' and a.get('content') == text)
                   for text in ('first done', 'reply second', 'reply third')]
        spans = zip(('1700000000.000400', '1700000000.000401', '1700000000.000402'), [0] + replies[:2], replies)
        for message_id, begin, end in spans:  # the heartbeats sent while that turn ran
            heartbeats = [a for a in actions[begin:end] if a.get('op') == 'typing' and a.get('content') != '']
            assert heartbeats and {_thread(a) for a in heartbeats} == {message_id}


@pytest.mark.asyncio
async def test_a_channel_thread_turn_keeps_its_thread_status(tmp_path, monkeypatch):
    """A turn in a channel thread names its thread in its own metadata: unchanged."""
    release = asyncio.Event()

    async def held(_event):
        await release.wait()
        _record({'final_response': '', 'messages': [], 'interrupted': True})

    async with _gateway(tmp_path, 'status-anchor-channel', FLAT_DM_SESSION) as (runner, send, outgoing, _adapter):
        executed = _stub_model(runner, monkeypatch, {'in thread': held})
        await send('1700000000.000501', 'C5', 'in thread', thread_id='1700000000.000500', chat_type='channel')
        await async_wait_until(lambda: any(_thread(a) == '1700000000.000500' for a in _heartbeats(outgoing)),
                               timeout=10, what='the thread turn shows its status')
        beats = len(_heartbeats(outgoing))
        await send('1700000000.000600', 'C5', 'elsewhere', thread_id='1700000000.000599', chat_type='channel')
        await async_wait_until(lambda: len(_heartbeats(outgoing)) > beats + 1, timeout=10,
                               what='heartbeats after the other thread spoke')
        release.set()
        await async_wait_until(lambda: 'elsewhere' in executed, timeout=20, what='both turns ran')
        await _settle(outgoing)
        assert _left_working(_actions(outgoing)) == set()
        mine = [a for a in _actions(outgoing) if a.get('op') == 'typing' and _thread(a) == '1700000000.000500']
        assert mine and mine[-1].get('content') == ''


def _dm_event(message_id, chat_id='D7'):
    from gateway.platforms.event import MessageEvent, MessageType
    from gateway.session import SessionSource
    source = SessionSource(platform=Platform.SLACK, chat_id=chat_id, chat_type='dm', user_id='U1', scope_id='T1')
    return MessageEvent(text='hi', source=source, message_type=MessageType.TEXT, message_id=message_id)


def _stub_adapter(extra):
    from gateway.relay.adapter import RelayAdapter
    from tests.gateway.relay.stub_connector import StubConnector
    from tests.gateway.relay.test_relay_slack_prompt_dm_root import _slack_desc

    stub = StubConnector(_slack_desc())
    return RelayAdapter(PlatformConfig(extra=extra), _slack_desc(), transport=stub), stub


@pytest.mark.asyncio
async def test_an_adapter_bracketed_turn_keeps_its_status_on_its_own_message():
    """A turn ``_process_message_background`` brackets (a command turn, such as a skill command) keeps
    refreshing and then clears its own message's status after a newer message reached the chat."""
    adapter, stub = _stub_adapter(FLAT_DM_SESSION)
    first = _dm_event('1700000000.000800')
    adapter._capture_scope(first)

    def beats():
        return [a for a in stub.sent if a['op'] == 'typing' and a.get('content') != '']

    async def handler(_event):
        await async_wait_until(beats, timeout=10, what='the turn shows its status')
        adapter._capture_scope(_dm_event('1700000000.000801'))  # a newer top-level message
        seen = len(beats())
        await async_wait_until(lambda: len(beats()) > seen, timeout=10, what='a refresh after it')
        # Returns no reply, as an interrupted turn does: nothing posted in its thread clears the status.

    adapter.set_message_handler(handler)
    await adapter._process_message_background(first, 'agent:main:slack:dm:T1:D7')
    assert {_thread(a) for a in stub.sent if a['op'] == 'typing'} == {'1700000000.000800'}
    assert _left_working(stub.sent) == set()


async def _adapter_with_two_inbounds(extra):
    """A relay adapter that saw the turn's message, then a newer top-level one in the same DM."""
    adapter, stub = _stub_adapter(extra)
    first = _dm_event('1700000000.000700')
    for event in (first, _dm_event('1700000000.000701')):
        adapter._capture_scope(event)
    return adapter, stub, first


@pytest.mark.asyncio
async def test_progress_restores_the_status_on_the_turns_own_message():
    """``_progress_restore_typing`` re-sets the status after a progress bubble: on this turn's message,
    also when the progress metadata names no thread (flat replies)."""
    from gateway.run_turn_progress import GatewayTurnProgressMixin

    adapter, stub, event = await _adapter_with_two_inbounds(FLAT_REPLIES)
    progress = GatewayTurnProgressMixin()
    progress._ctx = SimpleNamespace(source=event.source, event_message_id=event.message_id,
                                    _progress_metadata=None, _run_still_current=lambda: True)
    await progress._progress_restore_typing(SimpleNamespace(adapter=adapter))
    typing = [f for f in stub.sent if f['op'] == 'typing']
    assert [(f['metadata'] or {}).get('thread_id') for f in typing] == ['1700000000.000700']


@pytest.mark.asyncio
async def test_the_runners_end_of_turn_clear_targets_the_turns_own_message(tmp_path):
    """``_hmwa_stop_typing_for_turn`` (the runner's clear once the agent returned) clears this turn's
    status, not the status of the chat's newest message."""
    adapter, stub, event = await _adapter_with_two_inbounds(FLAT_DM_SESSION)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    runner.adapters[Platform.RELAY] = adapter
    runner._delivery_adapter_for = lambda _source: adapter
    await runner._hmwa_stop_typing_for_turn(event, event.source)
    clears = [f for f in stub.sent if f['op'] == 'typing' and f.get('content') == '']
    assert clears and {(f['metadata'] or {}).get('thread_id') for f in clears} == {'1700000000.000700'}
