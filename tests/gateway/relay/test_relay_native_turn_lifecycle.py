"""A relay chat message admitted natively (``admit_producer``) gets the adapter processing lifecycle.

``RelayAdapter._on_inbound`` commits a non-command message through ``admit_relay_delivery`` and the
session authority's drain runs it, so ``BasePlatformAdapter._process_message_background`` (the
typing refresh, ``on_processing_start`` and ``on_processing_complete``) never brackets it. The turn
executor owns that bracket for producer deliveries: opened when the admission starts executing
(a message queued behind a running turn is acknowledged when it runs), typing stopped when
execution ends, and the outcome reaction swapped once the turn settled and its reply was sent.

Real ``GatewayRunner`` + session authority + ``RelayAdapter`` + ``WebSocketRelayTransport`` against
a fixture connector; only the model call (``_handle_message_with_agent``) is stubbed.
"""
import asyncio
import json
from contextlib import asynccontextmanager

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session_results import execution_result, record_unexecuted_failure
from tests.waiting import async_wait_until

websockets = pytest.importorskip('websockets')

EYES, OK, FAIL = '👀', '✅', '❌'


@asynccontextmanager
async def _relay(runner, *, refusing_chats=()):
    """A real relay transport + adapter fronting Slack (``react`` advertised) against a fixture
    connector that answers every outbound frame; every ``send`` to a chat in ``refusing_chats`` is
    refused. Yields ``(send, outgoing, adapter)``."""
    from gateway.relay.adapter import RelayAdapter
    from gateway.relay.descriptor import CapabilityDescriptor
    from gateway.relay.ws_transport import WebSocketRelayTransport

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
                refused = frame['action'].get('op') == 'send' and frame['action'].get('chat_id') in refusing_chats
                result = {'success': False, 'error': 'refused'} if refused else {'success': True, 'message_id': 'sent'}
                await ws.send(json.dumps({'type': 'outbound_result', 'requestId': frame['requestId'],
                                          'result': result}) + '\n')

    async with websockets.serve(connector, '127.0.0.1', 0) as server:
        transport = WebSocketRelayTransport(
            f'ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/relay', 'slack', 'fixture-bot',
            gateway_id='fixture-gateway', upgrade_secret='fixture-secret', reconnect=False)
        adapter = RelayAdapter(PlatformConfig(enabled=True), descriptor, transport)
        adapter.gateway_runner = runner
        runner.adapters[Platform.RELAY] = adapter
        runner._wire_adapter_handlers(adapter)
        try:
            assert await adapter.connect()
            ws = await asyncio.wait_for(ready.get(), 5)

            async def send(message_id, chat_id, text, thread_id=None):
                source = {'platform': 'slack', 'chat_type': 'dm', 'chat_id': chat_id, 'user_id': 'U-' + chat_id}
                if thread_id is not None:
                    source['thread_id'] = thread_id  # a reply in that message's thread: its session
                event = {'text': text, 'message_id': message_id, 'source': source}
                await ws.send(json.dumps({'type': 'inbound', 'event': event, 'bufferId': 'buf-' + message_id}) + '\n')

            yield send, outgoing, adapter
        finally:
            await adapter.disconnect()


@asynccontextmanager
async def _gateway(tmp_path, instance, **relay):
    from gateway.session_authority import initialize_session_authority
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    authority = await initialize_session_authority(runner, profile_id='default', instance_id=instance)
    try:
        async with _relay(runner, **relay) as (send, outgoing, adapter):
            yield runner, authority, send, outgoing, adapter
    finally:
        authority.db.close()


def _stub_model(runner, monkeypatch, behaviours=None):
    """Replace the model turn: ``behaviours[text]`` (async, gets the event) decides the reply, else
    ``'reply <text>'``. Returns the list of executed texts."""
    executed = []

    async def turn(event, *_args, **_kwargs):
        executed.append(event.text)
        behaviour = (behaviours or {}).get(event.text)
        if behaviour is not None:
            return await behaviour(event)
        execution_result.get()['result'] = {'final_response': 'reply ' + event.text, 'messages': []}
        return 'reply ' + event.text

    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
    return executed


def _actions(outgoing):
    return [f['action'] for f in outgoing if f.get('type') == 'outbound']


def _reacts(outgoing, message_id):
    return [(a['emoji'], a['remove']) for a in _actions(outgoing)
            if a.get('op') == 'react' and a.get('message_id') == message_id]


def _index(outgoing, predicate):
    actions = _actions(outgoing)
    return next(i for i, a in enumerate(actions) if predicate(a))


def _last_index(outgoing, predicate):
    actions = _actions(outgoing)
    return max(i for i, a in enumerate(actions) if predicate(a))


def _heartbeats(outgoing, chat_id):
    return [a for a in _actions(outgoing)
            if a.get('op') == 'typing' and a.get('chat_id') == chat_id and a.get('content') != '']


def _is_reply(text):
    return lambda a: a.get('op') == 'send' and a.get('content') == text


def _settled(outgoing, message_id, final):
    return lambda: _reacts(outgoing, message_id)[-1:] == [final]


def _typing_cleared_after_last_heartbeat(outgoing, chat_id):
    actions = _actions(outgoing)
    typing = [i for i, a in enumerate(actions) if a.get('op') == 'typing' and a.get('chat_id') == chat_id]
    return bool(typing) and actions[typing[-1]].get('content') == ''


def _typing_refreshes():
    return [task for task in asyncio.all_tasks() if not task.done()
            and getattr(task.get_coro(), '__qualname__', '').endswith('._keep_typing')]


@pytest.mark.asyncio
async def test_toolless_chat_message_shows_typing_and_reactions_around_its_reply(tmp_path, monkeypatch):
    async with _gateway(tmp_path, 'relay-lifecycle-ok') as (runner, _authority, send, outgoing, _adapter):
        async def wait_for_typing(event):
            # The status is visible WHILE the turn runs (no tools, so nothing else sends one).
            await async_wait_until(lambda: _heartbeats(outgoing, 'D1'), timeout=10, what='typing during the turn')
            execution_result.get()['result'] = {'final_response': 'hi there', 'messages': []}
            return 'hi there'

        executed = _stub_model(runner, monkeypatch, {'hello': wait_for_typing})
        await send('1700000000.000100', 'D1', 'hello')
        await async_wait_until(_settled(outgoing, '1700000000.000100', (OK, False)), timeout=20,
                               what='the success reaction')
        assert executed == ['hello']
        assert _reacts(outgoing, '1700000000.000100') == [(EYES, False), (EYES, True), (OK, False)]
        reply = _index(outgoing, _is_reply('hi there'))
        assert _index(outgoing, lambda a: a.get('op') == 'react' and not a['remove']) < reply
        # The outcome follows the delivered reply.
        assert _last_index(outgoing, lambda a: a.get('op') == 'react') > reply
        await async_wait_until(lambda: _typing_cleared_after_last_heartbeat(outgoing, 'D1'), timeout=10,
                               what='the Slack status cleared')
        # No status refresh outlives the turn (a leaked one keeps Slack's "is working" up).
        await async_wait_until(lambda: not _typing_refreshes(), timeout=10, what='the typing refresh stopped')


async def _raise(_event):
    raise RuntimeError('model exploded')


async def _recorded_failure(_event):
    # The real agent-exception path (``run_turn_persistence``): a failed result plus the apology.
    return record_unexecuted_failure('Sorry, that failed.')


@pytest.mark.asyncio
@pytest.mark.parametrize('behaviour', [_raise, _recorded_failure], ids=['raised', 'recorded'])
async def test_a_failed_turn_swaps_in_the_failure_reaction_and_clears_typing(tmp_path, monkeypatch, behaviour):
    async with _gateway(tmp_path, 'relay-lifecycle-fail') as (runner, _authority, send, outgoing, _adapter):
        _stub_model(runner, monkeypatch, {'boom': behaviour})
        await send('1700000000.000200', 'D2', 'boom')
        await async_wait_until(_settled(outgoing, '1700000000.000200', (FAIL, False)), timeout=20,
                               what='the failure reaction')
        assert _reacts(outgoing, '1700000000.000200') == [(EYES, False), (EYES, True), (FAIL, False)]
        await async_wait_until(lambda: _typing_cleared_after_last_heartbeat(outgoing, 'D2'), timeout=10,
                               what='the Slack status cleared')


@pytest.mark.asyncio
async def test_a_refused_reply_is_a_failure(tmp_path, monkeypatch):
    async with _gateway(tmp_path, 'relay-lifecycle-refused', refusing_chats=('D25',)) as (
            runner, _authority, send, outgoing, _adapter):
        _stub_model(runner, monkeypatch)
        await send('1700000000.000250', 'D25', 'nope')
        await async_wait_until(_settled(outgoing, '1700000000.000250', (FAIL, False)), timeout=20,
                               what='the failure reaction')
        assert _reacts(outgoing, '1700000000.000250') == [(EYES, False), (EYES, True), (FAIL, False)]


@pytest.mark.asyncio
async def test_an_interrupted_turn_only_removes_the_in_progress_reaction(tmp_path, monkeypatch):
    async def interrupted(_event):
        execution_result.get()['result'] = {'final_response': '', 'messages': [], 'interrupted': True}

    async with _gateway(tmp_path, 'relay-lifecycle-stop') as (runner, _authority, send, outgoing, _adapter):
        executed = _stub_model(runner, monkeypatch, {'long job': interrupted})
        await send('1700000000.000300', 'D3', 'long job')
        await async_wait_until(lambda: (EYES, True) in _reacts(outgoing, '1700000000.000300'), timeout=20,
                               what='the in-progress reaction removed')
        # Settled: a later message in its thread runs, so the interrupted one's bracket is closed.
        await send('1700000000.000301', 'D3', 'after', thread_id='1700000000.000300')
        await async_wait_until(_settled(outgoing, '1700000000.000301', (OK, False)), timeout=20,
                               what='the next turn settled')
        assert executed == ['long job', 'after']
        assert _reacts(outgoing, '1700000000.000300') == [(EYES, False), (EYES, True)]
        await async_wait_until(lambda: _typing_cleared_after_last_heartbeat(outgoing, 'D3'), timeout=10,
                               what='the Slack status cleared')


@pytest.mark.asyncio
async def test_a_follow_up_queued_behind_a_running_turn_is_acknowledged_when_it_starts(tmp_path, monkeypatch):
    release = asyncio.Event()

    async def held(_event):
        await release.wait()
        execution_result.get()['result'] = {'final_response': 'first done', 'messages': []}
        return 'first done'

    async with _gateway(tmp_path, 'relay-lifecycle-queue') as (runner, authority, send, outgoing, _adapter):
        executed = _stub_model(runner, monkeypatch, {'first': held})
        await send('1700000000.000400', 'D4', 'first')
        await async_wait_until(lambda: _reacts(outgoing, '1700000000.000400') == [(EYES, False)], timeout=20,
                               what='the first message acknowledged')
        await send('1700000000.000401', 'D4', 'second', thread_id='1700000000.000400')
        await async_wait_until(
            lambda: authority.db._read_all("SELECT 1 FROM session_admissions WHERE request_id='1700000000.000401'"),
            timeout=20, what='the follow-up committed')
        # Committed behind the running turn, not started: no acknowledgement yet.
        assert _reacts(outgoing, '1700000000.000401') == []
        assert executed == ['first']
        release.set()
        await async_wait_until(_settled(outgoing, '1700000000.000401', (OK, False)), timeout=20,
                               what='the follow-up settled')
        assert executed == ['first', 'second']
        assert _reacts(outgoing, '1700000000.000400') == [(EYES, False), (EYES, True), (OK, False)]
        assert _reacts(outgoing, '1700000000.000401') == [(EYES, False), (EYES, True), (OK, False)]
        # It is acknowledged once it starts, after the turn ahead of it answered.
        assert _index(outgoing, _is_reply('first done')) < _index(
            outgoing, lambda a: a.get('op') == 'react' and a['message_id'] == '1700000000.000401')


@pytest.mark.asyncio
async def test_a_slash_command_keeps_its_single_adapter_lifecycle(tmp_path, monkeypatch):
    async with _gateway(tmp_path, 'relay-lifecycle-command') as (runner, _authority, send, outgoing, _adapter):
        _stub_model(runner, monkeypatch)
        await send('1700000000.000500', 'D5', '/help')
        await async_wait_until(_settled(outgoing, '1700000000.000500', (OK, False)), timeout=20,
                               what='the command lifecycle')
        assert _reacts(outgoing, '1700000000.000500') == [(EYES, False), (EYES, True), (OK, False)]


@pytest.mark.asyncio
async def test_pre_gateway_dispatch_still_fires_once_and_the_turn_is_acknowledged(tmp_path, monkeypatch):
    from hermes_cli.plugins import get_plugin_manager
    fired = []

    async def hook(event, **_kwargs):
        fired.append(event.message_id)

    manager = get_plugin_manager()
    manager._discovered = True
    manager._hooks.setdefault('pre_gateway_dispatch', []).append(hook)
    async with _gateway(tmp_path, 'relay-lifecycle-hook') as (runner, _authority, send, outgoing, _adapter):
        executed = _stub_model(runner, monkeypatch)
        await send('1700000000.000600', 'D6', 'hooked')
        await async_wait_until(_settled(outgoing, '1700000000.000600', (OK, False)), timeout=20,
                               what='the success reaction')
        assert executed == ['hooked']
        assert fired == ['1700000000.000600']
        assert _reacts(outgoing, '1700000000.000600') == [(EYES, False), (EYES, True), (OK, False)]


@pytest.mark.asyncio
async def test_a_scheduled_wake_on_the_chat_shows_no_reaction_or_typing(tmp_path, monkeypatch):
    from gateway.session_automation import admit_route_wake

    async with _gateway(tmp_path, 'relay-lifecycle-wake') as (runner, _authority, send, outgoing, adapter):
        executed = _stub_model(runner, monkeypatch)
        await send('1700000000.000700', 'D7', 'hello')
        await async_wait_until(_settled(outgoing, '1700000000.000700', (OK, False)), timeout=20,
                               what='the human turn settled')
        await async_wait_until(lambda: _typing_cleared_after_last_heartbeat(outgoing, 'D7'), timeout=10,
                               what='the Slack status cleared')
        before_reacts = [a for a in _actions(outgoing) if a.get('op') == 'react']
        before_typing = len(_heartbeats(outgoing, 'D7'))
        entry = next(e for e in runner.session_store._entries.values()
                     if e.origin is not None and e.origin.chat_id == 'D7')
        assert await admit_route_wake(runner, adapter, entry.origin, 'scheduled tick', 'loop') is True
        await async_wait_until(lambda: 'scheduled tick' in executed, timeout=20, what='the wake ran')
        await async_wait_until(lambda: any(_is_reply('reply scheduled tick')(a) for a in _actions(outgoing)),
                               timeout=20, what='the wake answered')
        # A bracket would have reacted and refreshed typing when the wake started, before its reply.
        assert [a for a in _actions(outgoing) if a.get('op') == 'react'] == before_reacts
        assert len(_heartbeats(outgoing, 'D7')) == before_typing


@pytest.mark.asyncio
async def test_a_cancel_while_the_start_hook_runs_leaks_no_typing():
    """A drain cancelled while on_processing_start is awaited stops typing and drops the bracket."""
    from gateway.platforms.base import BasePlatformAdapter
    from gateway.session_producer_lifecycle import producer_turn

    class Adapter(BasePlatformAdapter):
        name = property(lambda self: 'fake')

        def __init__(self):
            self._typing_paused = set()
            self._status_text = {}
            self.config = type('C', (), {'typing_indicator': True})()
            self.started = asyncio.Event()

        async def send_typing(self, chat_id, metadata=None):
            return None

        async def stop_typing(self, chat_id, metadata=None):
            return None

        async def on_processing_start(self, event):
            self.started.set()
            await asyncio.Event().wait()

        async def on_processing_complete(self, event, outcome):
            return None

        async def get_chat_info(self, *a, **k): ...
        async def connect(self): ...
        async def disconnect(self): ...
        async def send(self, *a, **k): ...

    adapter = Adapter()
    source = type('S', (), {'chat_id': 'C1', 'thread_id': None, 'platform': 'slack', 'chat_type': 'dm',
                            'user_id': 'U', 'scope_id': None, 'profile': None})()
    event = type('E', (), {'source': source, 'message_id': 'm1', 'raw_message': None, 'internal': False,
                           'reply_anchor_override': None})()
    runner = type('R', (), {'_adapter_for_source': lambda self, s: adapter})()
    authority = type('A', (), {'runner': runner, 'producer_turns': {'adm': None}})()

    async def drain():
        async with producer_turn(authority, {'admission_id': 'adm'}, event, {}):
            pass

    task = asyncio.create_task(drain())
    await adapter.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await async_wait_until(lambda: not _typing_refreshes(), timeout=2)
    assert authority.producer_turns == {}

