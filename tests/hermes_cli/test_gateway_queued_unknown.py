"""A finite queued viewer detaches when its predecessor becomes unknown."""
import asyncio
import json

import pytest

from hermes_cli.gateway_chat_view import GatewayChatView


@pytest.mark.asyncio
@pytest.mark.parametrize('before_ack', [False, True])
@pytest.mark.parametrize('stream_json', [False, True])
async def test_finite_follower_does_not_wait_forever_after_worker_loss(before_ack, stream_json, capsys):
    from hermes_cli.stream_json import StreamJsonEmitter

    rendered = asyncio.Event()

    class Events(asyncio.Queue):
        async def get(self):
            event = await super().get()
            rendered.set()
            return event

    class Peer:
        events = Events()
        calls = []

        async def rpc(self, method, **params):
            self.calls.append((method, params))
            assert method == 'prompt.submit'
            self.events.put_nowait({'method': 'event', 'params': {'session_id': 's',
                'type': 'session.info', 'admission_id': 'predecessor', 'payload': {'pending': [
                    {'admission_id': 'predecessor', 'status': 'unknown'},
                    {'admission_id': 'ours', 'status': 'queued'}]}}})
            if before_ack:
                await rendered.wait()
            return {'admission_id': 'ours'}

    peer = Peer()
    emitter = StreamJsonEmitter(model='m', session_id='s') if stream_json else None
    view = GatewayChatView(peer, {'stored_session_id': 's'}, quiet=True, emitter=emitter)
    assert await asyncio.wait_for(view.run('queued', oneshot=True), 5) == 3
    assert len(peer.calls) == 1
    assert peer.calls[0][1]['text'] == 'queued'
    output = capsys.readouterr()
    assert 'accepted input is retained' in output.err and 'do not resend' in output.err
    if stream_json:
        records = [json.loads(line) for line in output.out.splitlines()]
        assert records[-1]['type'] == 'result' and records[-1]['exit_code'] == 3


@pytest.mark.asyncio
async def test_successor_unknown_does_not_replace_completed_prompt():
    class Peer:
        events = asyncio.Queue()

        async def rpc(self, method, **params):
            if method == 'prompt.receipt':
                return {'result': {'completed': True}}
            assert method == 'prompt.submit'
            for kind, admission, payload in [
                ('message.complete', 'ours', {'outcome': 'completed', 'text': 'done'}),
                ('session.info', 'next', {'pending': [{'admission_id': 'next', 'status': 'unknown'}]}),
            ]:
                self.events.put_nowait({'method': 'event', 'params': {'session_id': 's',
                    'type': kind, 'admission_id': admission, 'payload': payload}})
            return {'admission_id': 'ours'}

    view = GatewayChatView(Peer(), {'stored_session_id': 's'}, quiet=True)
    assert await asyncio.wait_for(view.run('work', oneshot=True), 5) == 0
