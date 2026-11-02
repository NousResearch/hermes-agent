"""A text-to-speech-style consumer over real loopback WebSockets, without audio.

Only the provider HTTP transport is simulated; routing, SDK SSE decoding, the
public plugin API and consumer TCP/WebSocket framing are real. No telephony,
STT, TTS, remote model, or third-party service is exercised here.
"""

import asyncio
from contextlib import suppress
import json
from pathlib import Path
import os

import httpx
from openai import AsyncOpenAI
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from tests.agent import test_plugin_llm_stream as stream_fixtures

context = stream_fixtures.context


QUESTION = 'Proč mají listy na podzim jiné barvy?'
FIRST_SENTENCE = 'Na podzim ubývá chlorofylu.'
LAST_SENTENCE = ' Vyniknou tak další barviva.'


class SentenceWire(httpx.AsyncByteStream):
    """The provider cannot finish until the test explicitly opens its barrier."""

    def __init__(self):
        self.release = asyncio.Event()
        self.waiting = asyncio.Event()
        self.closed = asyncio.Event()
        self.completed = asyncio.Event()

    @staticmethod
    def event(text=None, finish=None):
        delta = {} if text is None else {
            'content': text,
            'reasoning_content': 'private-reasoning',
        }
        payload = dict(id='provider-private-id', object='chat.completion.chunk',
                       created=0, model='fast-model', choices=[
                           dict(index=0, delta=delta, finish_reason=finish)])
        return ('data: ' + json.dumps(payload) + '\n\n').encode()

    async def __aiter__(self):
        yield self.event('Na podzim ubývá ')
        yield self.event('chlorofylu.')
        self.waiting.set()
        await self.release.wait()
        yield self.event(LAST_SENTENCE)
        yield self.event(finish='stop')
        self.completed.set()
        yield b'data: [DONE]\n\n'

    async def aclose(self):
        self.closed.set()


class JsonSocket:
    def __init__(self, socket):
        self.socket = socket

    async def send_json(self, value):
        await self.socket.send(json.dumps(value))

    async def receive_json(self):
        return json.loads(await self.socket.recv())


async def speak_sentences(ctx, socket, question):
    """A minimal test-only chunking adapter, not a production voice service."""
    pending = ''
    async with ctx.llm.astream(
        [{'role': 'user', 'content': question}], task='realtime',
    ) as stream:
        async for event in stream:
            if event.type == 'text_delta':
                pending += event.text
                while '.' in pending:
                    sentence, pending = pending.split('.', 1)
                    await socket.send_json({'type': 'speech_text', 'text': sentence + '.'})
            elif event.type == 'end':
                assert event.finish_reason == 'stop'
                assert event.audit['task'] == 'realtime'
    if pending:
        await socket.send_json({'type': 'speech_text', 'text': pending})
    await socket.send_json({'type': 'done'})


def test_loopback_consumer_interrupts_owned_request_and_next_call_completes(context, monkeypatch):
    home = Path(os.environ['HERMES_HOME'])
    (home / 'SOUL.md').write_text('private-owner-persona', encoding='utf-8')
    (home / 'memories').mkdir(exist_ok=True)
    (home / 'memories' / 'MEMORY.md').write_text('private-owner-memory', encoding='utf-8')

    async def exercise():
        wires = [SentenceWire(), SentenceWire()]
        requests = []

        def handle(request):
            wire = wires[len(requests)]
            requests.append(json.loads(request.content))
            return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=wire)

        client = AsyncOpenAI(api_key='fixture', base_url='http://fixture.invalid/v1',
                             http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        # Preserve the real PluginContext, task ownership, route/model selection,
        # request construction and SDK. Replace only SDK client construction.
        monkeypatch.setattr('agent.auxiliary_client._to_async_client',
                            lambda sync, model, **kw: (client, model))

        async def websocket(connection):
            socket = JsonSocket(connection)
            producer = None
            try:
                async for message in connection:
                    command = json.loads(message)
                    if command['type'] == 'question':
                        assert producer is None
                        producer = asyncio.create_task(
                            speak_sentences(context, socket, command['text']))
                    elif command['type'] == 'interrupt':
                        assert producer is not None
                        producer.cancel()
                        with suppress(asyncio.CancelledError):
                            await producer
                        await socket.send_json({'type': 'interrupted'})
            finally:
                if producer is not None:
                    producer.cancel()
                    with suppress(asyncio.CancelledError):
                        await producer

        try:
            async with serve(websocket, '127.0.0.1', 0) as server:
                port = next(iter(server.sockets)).getsockname()[1]
                async with connect(f'ws://127.0.0.1:{port}/speech') as connection:
                    socket = JsonSocket(connection)
                    await socket.send_json({'type': 'question', 'text': QUESTION})
                    first = await asyncio.wait_for(socket.receive_json(), 5)
                    await asyncio.wait_for(wires[0].waiting.wait(), 5)
                    assert first == {'type': 'speech_text', 'text': FIRST_SENTENCE}
                    # Receipt crossed a real TCP/WebSocket boundary while the
                    # upstream completion was still impossible, not just delayed.
                    assert not wires[0].release.is_set()
                    assert not wires[0].completed.is_set()
                    await socket.send_json({'type': 'interrupt'})
                    await asyncio.wait_for(wires[0].closed.wait(), 2)
                    assert await asyncio.wait_for(socket.receive_json(), 5) == {'type': 'interrupted'}
                    assert not wires[0].release.is_set()
                    assert not wires[0].completed.is_set()
                    assert not client.is_closed()

                async with connect(f'ws://127.0.0.1:{port}/speech') as connection:
                    socket = JsonSocket(connection)
                    await socket.send_json({'type': 'question', 'text': QUESTION})
                    assert await asyncio.wait_for(socket.receive_json(), 5) == first
                    await asyncio.wait_for(wires[1].waiting.wait(), 5)
                    assert not wires[1].closed.is_set()
                    wires[1].release.set()
                    assert await asyncio.wait_for(socket.receive_json(), 5) == {
                        'type': 'speech_text', 'text': LAST_SENTENCE}
                    assert await asyncio.wait_for(socket.receive_json(), 5) == {'type': 'done'}
                    assert wires[1].completed.is_set()
                    assert wires[1].closed.is_set()
                    assert not client.is_closed()

            assert len(requests) == 2
            for request in requests:
                assert request['model'] == 'fast-model'
                assert request['stream'] is True
                assert request['messages'] == [{'role': 'user', 'content': QUESTION}]
                assert 'tools' not in request
                assert 'private-owner' not in json.dumps(request)
        finally:
            await client.close()

    asyncio.run(exercise())
