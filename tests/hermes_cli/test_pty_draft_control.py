"""Draft control exercises the real PTY session and ASGI WebSocket transport."""
import asyncio
from contextlib import suppress
import json
import os
import sys

import pytest
from starlette.websockets import WebSocket

from hermes_cli.pty_session import PtySession

PREFIX = b"\x00hermes-draft-v1:"


class Wire:
    def __init__(self):
        self.sent = []
        self.failure: BaseException | None = None
        self.incoming = asyncio.Queue()
        self.ws = WebSocket({"type": "websocket", "path": "/", "headers": []}, self.incoming.get, self.send)

    async def send(self, message):
        if self.failure is not None:
            raise self.failure
        self.sent.append(message)

    async def open(self):
        await self.incoming.put({"type": "websocket.connect"})
        await self.ws.accept()
        return self.ws

    def frames(self):
        return [json.loads(m['text']) for m in self.sent if m.get('text')]


class RecordingBridge:
    def __init__(self):
        self.writes = []

    async def write(self, raw):
        self.writes.append(raw)
        return True

    def close(self):
        pass


@pytest.mark.asyncio
async def test_reserved_frames_never_reach_real_session_terminal_writer():
    bridge = RecordingBridge()
    session = PtySession('key', bridge, buffer_cap=1024, read_timeout=.01)
    wire = Wire()
    ws = await wire.open()
    await session.attach(ws)
    for payload in [b'{broken', b'{"type":"shell.exec","path":"/tmp/x"}', b'[]', b'null']:
        assert await session.write(ws, PREFIX + payload)
    assert await session.write(ws, b'normal typing')
    assert bridge.writes == [b'normal typing']
    await session.close()


@pytest.mark.asyncio
async def test_private_controller_roundtrip_adds_pty_identity_without_rpc_or_terminal_input():
    bridge = RecordingBridge()
    session = PtySession('key', bridge, buffer_cap=1024, read_timeout=.01)
    browser, node = Wire(), Wire()
    await session.attach(await browser.open())
    node_ws = await node.open()
    control = session.draft
    assert not await control.claim(node_ws, credential='wrong', generation=1)
    assert await control.claim(node_ws, credential=control.credential, generation=1)
    binding = node.frames()[-1]
    assert binding['type'] == 'draft.refresh'
    await control.receive(node_ws, {**binding, 'type': 'draft.state',
                                   'session_id': 'native-sid', 'draft_id': 'native-draft', 'available': True})
    state = browser.frames()[-1]
    assert state['type'] == 'draft.state' and state['available']
    assert state['identity']['session_id'] == 'native-sid'
    assert state['identity']['pty_instance'] != session.key
    assert control.credential not in json.dumps(browser.frames())
    request = {'type': 'draft.attach', 'request_id': 'r1', 'expected': state['identity'],
               'path': '/private/upload.txt', 'method': 'shell.exec', 'params': {'command': 'bad'}}
    await session.write(browser.ws, PREFIX + json.dumps(request).encode())
    forwarded = node.frames()[-1]
    assert forwarded == {key: request[key] for key in ('type', 'request_id', 'expected', 'path')}
    result = {'type': 'draft.result', 'request_id': 'r1', 'identity': state['identity'], 'status': 'attached'}
    await control.receive(node_ws, result)
    assert browser.frames()[-1] == result
    assert bridge.writes == []
    await session.close()


@pytest.mark.asyncio
async def test_detach_replacement_and_reconnect_retire_authority_without_retargeting():
    session = PtySession('key', RecordingBridge(), buffer_cap=1024, read_timeout=.01)
    first, second, node, replacement = Wire(), Wire(), Wire(), Wire()
    await session.attach(await first.open())
    control = session.draft
    await control.claim(await node.open(), credential=control.credential, generation=1)

    async def state(wire, draft='draft'):
        await control.receive(wire.ws, {**control.binding(), 'type': 'draft.state',
                              'session_id': 'sid', 'draft_id': draft, 'available': True})

    await state(node)
    identity = first.frames()[-1]['identity']
    request = {'type': 'draft.attach', 'request_id': 'pending', 'expected': identity, 'path': '/upload'}
    result = {'type': 'draft.result', 'request_id': 'pending', 'identity': identity, 'status': 'attached'}
    await session.write(first.ws, PREFIX + json.dumps(request).encode())
    session.detach(first.ws)
    await asyncio.sleep(0)
    assert node.frames()[-1]['type'] == 'draft.disconnected'
    await session.attach(await second.open())
    assert control.controller is node.ws
    await state(node)
    fresh = second.frames()[-1]['identity']
    assert fresh['pty_instance'] == identity['pty_instance']
    assert fresh['connection_generation'] > identity['connection_generation']
    await control.receive(node.ws, result)
    assert not any(frame['type'] == 'draft.result' for frame in second.frames())
    before = list(node.frames())
    await session.write(first.ws, PREFIX + json.dumps({**request, 'expected': fresh}).encode())
    assert node.frames() == before
    await session.write(second.ws, PREFIX + json.dumps(request).encode())
    assert second.frames()[-1]['status'] == 'stale'
    assert node.frames() == before
    assert await control.claim(await replacement.open(), credential=control.credential, generation=2)
    assert not await control.claim(node.ws, credential=control.credential, generation=1)
    assert not await control.claim(node.ws, credential=control.credential, generation=2)
    await state(replacement, 'new-draft')
    frames = list(second.frames())
    await state(node, 'forged')
    await control.receive(node.ws, result)
    assert second.frames() == frames
    await session.close()
    assert not await control.claim(node.ws, credential=control.credential, generation=3)


@pytest.mark.asyncio
async def test_unavailable_and_stale_results_keep_request_identity_on_authority_loss():
    session = PtySession('key', RecordingBridge(), buffer_cap=1024, read_timeout=.01)
    browser, node = Wire(), Wire()
    await session.attach(await browser.open())
    control = session.draft
    await control.claim(await node.open(), credential=control.credential, generation=1)
    state = {**control.binding(), 'type': 'draft.state', 'session_id': 'sid', 'draft_id': 'draft', 'available': True}
    await control.receive(node.ws, state)
    identity = browser.frames()[-1]['identity']
    request = {'type': 'draft.attach', 'request_id': 'one', 'expected': identity, 'path': '/staged'}
    await session.write(browser.ws, PREFIX + json.dumps(request).encode())
    await control.disconnected(node.ws)
    assert browser.frames()[-1] == {'type': 'draft.state', 'identity': identity, 'available': False}
    assert any(frame.get('status') == 'unavailable' and frame.get('identity') == identity for frame in browser.frames())
    replacement = Wire()
    await control.claim(await replacement.open(), credential=control.credential, generation=2)
    await control.receive(replacement.ws, {**state, **control.binding()})
    await session.write(browser.ws, PREFIX + json.dumps(request).encode())
    assert browser.frames()[-1] == {'type': 'draft.result', 'request_id': 'one', 'identity': identity, 'status': 'stale'}
    await session.close()


@pytest.mark.asyncio
async def test_slow_replacement_cannot_overwrite_a_newer_controller_generation():
    blocked, release = asyncio.Event(), asyncio.Event()

    class SlowWire(Wire):
        async def send(self, message):
            if message.get('text') and json.loads(message['text']).get('available') is False:
                blocked.set()
                await release.wait()
            await super().send(message)

    session = PtySession('key', RecordingBridge(), buffer_cap=1024, read_timeout=.01)
    browser, first, second, third = SlowWire(), Wire(), Wire(), Wire()
    await session.attach(await browser.open())
    control = session.draft
    await control.claim(await first.open(), credential=control.credential, generation=1)
    await control.receive(first.ws, {**control.binding(), 'type': 'draft.state', 'session_id': 'sid', 'draft_id': 'draft', 'available': True})
    await second.open(); await third.open()
    replacing = asyncio.create_task(control.claim(second.ws, credential=control.credential, generation=2))
    await asyncio.wait_for(blocked.wait(), 3)
    newest = asyncio.create_task(control.claim(third.ws, credential=control.credential, generation=3))
    await asyncio.sleep(0)
    release.set()
    await asyncio.wait_for(asyncio.gather(replacing, newest), 3)
    assert control.controller is third.ws
    assert control.controller_generation == 3
    await session.close()


@pytest.mark.asyncio
async def test_malformed_identity_and_conflicting_retries_cannot_replace_pending_request():
    session = PtySession('key', RecordingBridge(), buffer_cap=1024, read_timeout=.01)
    browser, node = Wire(), Wire()
    await session.attach(await browser.open())
    control = session.draft
    await control.claim(await node.open(), credential=control.credential, generation=1)
    await control.receive(node.ws, {**control.binding(), 'type': 'draft.state', 'session_id': 'sid', 'draft_id': 'draft', 'available': True})
    identity = browser.frames()[-1]['identity']
    request = {'type': 'draft.attach', 'request_id': 'one', 'expected': identity, 'path': '/original'}
    before = list(browser.frames())
    for invalid in [None, [], {}, {**identity, 'connection_generation': True}]:
        await session.write(browser.ws, PREFIX + json.dumps({**request, 'expected': invalid}).encode())
    assert browser.frames() == before
    await session.write(browser.ws, PREFIX + json.dumps(request).encode())
    node_frames = list(node.frames())
    await session.write(browser.ws, PREFIX + json.dumps({**request, 'path': '/replacement'}).encode())
    assert node.frames() == node_frames
    await control.receive(node.ws, {'type': 'draft.result', 'request_id': 'one', 'identity': identity, 'status': 'attached', 'error': {'method': 'shell.exec'}})
    assert browser.frames()[-1] == {'type': 'draft.result', 'request_id': 'one', 'identity': identity, 'status': 'attached'}
    await session.close()


@pytest.mark.asyncio
async def test_broken_controller_transport_reports_unavailable_without_breaking_terminal():
    class BrokenWire(Wire):
        async def send(self, message):
            if message.get('text') and json.loads(message['text']).get('type') == 'draft.attach':
                raise OSError('peer lost')
            await super().send(message)

    bridge = RecordingBridge()
    session = PtySession('key', bridge, buffer_cap=1024, read_timeout=.01)
    browser, node = Wire(), BrokenWire()
    await session.attach(await browser.open())
    control = session.draft
    await control.claim(await node.open(), credential=control.credential, generation=1)
    await control.receive(node.ws, {**control.binding(), 'type': 'draft.state', 'session_id': 'sid', 'draft_id': 'draft', 'available': True})
    identity = browser.frames()[-1]['identity']
    request = {'type': 'draft.attach', 'request_id': 'one', 'expected': identity, 'path': '/upload'}
    assert await session.write(browser.ws, PREFIX + json.dumps(request).encode())
    assert any(frame.get('status') == 'unavailable' for frame in browser.frames())
    assert await session.write(browser.ws, b'editing')
    assert bridge.writes == [b'editing']
    await session.close()


@pytest.mark.linux_only
@pytest.mark.asyncio
@pytest.mark.parametrize('failure', [OSError('sidecar lost'), asyncio.CancelledError()])
async def test_broken_sidecar_cannot_abort_pty_cleanup(failure):
    from hermes_cli.pty_bridge import PtyBridge
    from hermes_cli.pty_draft_control import LIVE_CONTROLLERS

    ready = asyncio.Event()

    class BrowserWire(Wire):
        async def send(self, message):
            await super().send(message)
            if message.get('bytes'):
                ready.set()

    bridge = PtyBridge.spawn([sys.executable, '-u', '-c',
                              "import signal; print('ready', flush=True); signal.pause()"])
    session = PtySession('key', bridge, buffer_cap=1024, read_timeout=.01)
    browser, node = BrowserWire(), Wire()
    try:
        await session.attach(await browser.open())
        assert await session.draft.claim(await node.open(), credential=session.draft.credential, generation=1)
        await session.start()
        await asyncio.wait_for(ready.wait(), 3)
        assert bridge.is_alive()
        node.failure = failure
        if isinstance(failure, asyncio.CancelledError):
            with pytest.raises(asyncio.CancelledError):
                await session.close()
        else:
            await session.close()
        assert session._drain_task is not None and session._drain_task.done()
        assert bridge._proc.closed
        assert not bridge._proc.isalive()
        with pytest.raises(OSError):
            os.fstat(bridge._fd)
        assert session.draft.controller is None
        assert session.draft.credential not in LIVE_CONTROLLERS
    finally:
        # A red regression must not itself leave a child or drain behind.
        if session._drain_task is not None:
            session._drain_task.cancel()
            with suppress(asyncio.CancelledError):
                await session._drain_task
        await asyncio.to_thread(bridge.close)


@pytest.mark.asyncio
async def test_controller_replacement_survives_closed_old_sender():
    bridge = RecordingBridge()
    session = PtySession('key', bridge, buffer_cap=1024, read_timeout=.01)
    browser, first, second = Wire(), Wire(), Wire()
    control = session.draft
    try:
        await session.attach(await browser.open())
        assert await control.claim(await first.open(), credential=control.credential, generation=1)
        first.failure = OSError('peer gone')
        assert await control.claim(await second.open(), credential=control.credential, generation=2)
        assert second.frames()[-1] == {'type': 'draft.refresh', **control.binding()}
        state = {**control.binding(), 'type': 'draft.state', 'session_id': 'sid',
                 'draft_id': 'replacement-draft', 'available': True}
        await control.receive(second.ws, state)
        identity = browser.frames()[-1]['identity']
        request = {'type': 'draft.attach', 'request_id': 'new', 'expected': identity, 'path': '/new'}
        await session.write(browser.ws, PREFIX + json.dumps(request).encode())
        result = {'type': 'draft.result', 'request_id': 'new', 'identity': identity, 'status': 'attached'}
        before = browser.frames()
        # The old pub handler's late finally/frames cannot revoke its successor.
        await control.disconnected(first.ws)
        await control.receive(first.ws, {**state, 'draft_id': 'forged'})
        await control.receive(first.ws, result)
        assert not await control.claim(first.ws, credential=control.credential, generation=1)
        assert control.controller is second.ws
        assert browser.frames() == before
        assert second.frames()[-1] == request
        await control.receive(second.ws, result)
        assert browser.frames()[-1] == result
        assert await session.write(browser.ws, b'live typing')
        assert bridge.writes == [b'live typing']
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_child_eof_closes_browser_even_when_sidecar_cleanup_is_cancelled():
    class EofBridge(RecordingBridge):
        def read(self, timeout):
            return None

    session = PtySession('key', EofBridge(), buffer_cap=1024, read_timeout=.01)
    browser, node = Wire(), Wire()
    try:
        await session.attach(await browser.open())
        await session.draft.claim(await node.open(), credential=session.draft.credential, generation=1)
        node.failure = asyncio.CancelledError()
        await session.start()
        assert session._drain_task is not None
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(session._drain_task, 3)
        assert browser.sent[-1]['type'] == 'websocket.close'
        assert browser.sent[-1]['code'] == 4410
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('scope_key', ['session_id', 'draft_id'])
async def test_state_change_retires_obsolete_pending_without_breaking_current_retries(scope_key):
    session = PtySession('key', RecordingBridge(), buffer_cap=1024, read_timeout=.01)
    browser, node = Wire(), Wire()
    control = session.draft
    try:
        await session.attach(await browser.open())
        await control.claim(await node.open(), credential=control.credential, generation=1)
        state = {**control.binding(), 'type': 'draft.state', 'session_id': 'sid',
                 'draft_id': 'draft', 'available': True}
        await control.receive(node.ws, state)
        old_identity = browser.frames()[-1]['identity']
        request = {'type': 'draft.attach', 'request_id': 'retry', 'expected': old_identity, 'path': '/upload'}
        await session.write(browser.ws, PREFIX + json.dumps(request).encode())
        await control.receive(node.ws, {**state, 'available': False})
        await control.receive(node.ws, state)
        await session.write(browser.ws, PREFIX + json.dumps(request).encode())
        assert node.frames()[-2:] == [request, request]
        assert control.pending['retry'][0] == request
        # Only a new identity invalidates in-flight work; availability/typing does not.
        await control.receive(node.ws, {**state, scope_key: 'next'})
        assert not control.pending
        current_identity = browser.frames()[-1]['identity']
        current = {**request, 'request_id': 'current', 'expected': current_identity}
        await session.write(browser.ws, PREFIX + json.dumps(current).encode())
        before = browser.frames()
        await control.receive(node.ws, {'type': 'draft.result', 'request_id': 'retry',
                                       'identity': old_identity, 'status': 'attached'})
        assert browser.frames() == before
        await session.write(browser.ws, PREFIX + json.dumps(request).encode())
        assert browser.frames()[-1]['status'] == 'stale'
        assert node.frames()[-1] == current
        result = {'type': 'draft.result', 'request_id': 'current', 'identity': current_identity, 'status': 'attached'}
        await control.receive(node.ws, result)
        assert browser.frames()[-1] == result
        assert not control.pending
    finally:
        await session.close()
