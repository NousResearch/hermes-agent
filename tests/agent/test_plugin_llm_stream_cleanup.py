"""Offline regressions for plugin stream resource and exception boundaries."""

import asyncio

import httpx
import pytest
from openai import AsyncOpenAI

from tests.agent import test_plugin_llm_stream as stream_fixtures

context = stream_fixtures.context
Wire = stream_fixtures.Wire


@pytest.mark.parametrize('carrier', ['direct', 'cause', 'context', 'args'])
def test_close_failure_has_no_authenticated_exception_chain(context, monkeypatch, carrier):
    from agent.plugin_llm_stream import PluginLlmStreamError

    async def exercise():
        class CloseFailure(Wire):
            async def aclose(self):
                await super().aclose()
                error = httpx.CloseError('fixture close failed', request=requests[0])
                if carrier == 'args':
                    raise RuntimeError(error)
                if carrier == 'cause':
                    raise RuntimeError('wrapped close') from error
                if carrier == 'context':
                    try:
                        raise error
                    except httpx.CloseError:
                        raise RuntimeError('wrapped close')
                raise error

        requests = []
        wire = CloseFailure()
        def handle(request):
            requests.append(request)
            return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=wire)
        client = AsyncOpenAI(api_key='private-fixture-token', base_url='http://fixture.invalid/v1',
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        monkeypatch.setattr('agent.auxiliary_client._to_async_client', lambda sync, model, **kw: (client, model))
        stream = context.llm.astream([{'role': 'user', 'content': 'hi'}], task='realtime')
        await stream.__aenter__()
        await anext(stream)
        assert requests[0].headers['authorization'] == 'Bearer private-fixture-token'
        try:
            with pytest.raises(PluginLlmStreamError) as caught:
                await stream.aclose()
            assert caught.value.__context__ is None
            assert caught.value.__cause__ is None
            assert not any(isinstance(arg, BaseException) for arg in caught.value.args)
            assert wire.closed
        finally:
            await client.close()
    asyncio.run(exercise())


@pytest.mark.parametrize('fault', ['runtime', 'httpx'])
@pytest.mark.parametrize('action', ['close', 'cancel_reader'])
def test_cancelled_read_failure_still_closes_wire_and_releases_permit(context, monkeypatch, fault, action):
    from agent.plugin_llm_stream import PluginLlmStreamError

    async def exercise():
        class CancelFailure(Wire):
            async def __aiter__(self):
                yield b'data: {"id":"fixture","object":"chat.completion.chunk","created":0,"model":"fixture","choices":[{"index":0,"delta":{"content":"first"},"finish_reason":null}]}\n\n'
                self.waiting.set()
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    if fault == 'httpx':
                        raise httpx.ReadError('private read failure', request=requests[0])
                    raise RuntimeError('read failed on cancellation')

        requests = []
        wire = CancelFailure()
        semaphore = asyncio.Semaphore(1)
        def handle(request):
            requests.append(request)
            return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=wire)
        client = AsyncOpenAI(api_key='fixture-private', base_url='http://fixture.invalid/v1',
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        monkeypatch.setattr('agent.auxiliary_client._to_async_client', lambda sync, model, **kw: (client, model))
        monkeypatch.setattr('agent.auxiliary_client._acquire_async_aux_semaphore', lambda task: semaphore)
        stream = context.llm.astream([{'role': 'user', 'content': 'hi'}], task='realtime')
        await stream.__aenter__()
        await anext(stream)
        pending = asyncio.create_task(anext(stream))
        await asyncio.wait_for(wire.waiting.wait(), 3)
        try:
            if action == 'close':
                with pytest.raises((RuntimeError, PluginLlmStreamError)):
                    await stream.aclose()
            else:
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            assert wire.closed
            await asyncio.wait_for(semaphore.acquire(), 2)
            semaphore.release()
            await stream.aclose()
        finally:
            await asyncio.gather(pending, return_exceptions=True)
            await client.close()
    asyncio.run(exercise())


@pytest.mark.parametrize('primary', [ValueError, asyncio.CancelledError])
def test_exit_cleanup_does_not_replace_primary_failure(context, monkeypatch, primary):
    async def exercise():
        class CloseFailure(Wire):
            async def aclose(self):
                await super().aclose()
                raise httpx.CloseError('close failure', request=requests[0])
        requests = []
        wire = CloseFailure()
        def handle(request):
            requests.append(request)
            return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=wire)
        client = AsyncOpenAI(api_key='fixture-private', base_url='http://fixture.invalid/v1',
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        monkeypatch.setattr('agent.auxiliary_client._to_async_client', lambda sync, model, **kw: (client, model))
        try:
            with pytest.raises(primary):
                async with context.llm.astream([{'role': 'user', 'content': 'hi'}], task='realtime') as stream:
                    await anext(stream)
                    raise primary('primary')
            assert wire.closed
        finally:
            await client.close()
    asyncio.run(exercise())


def test_concurrent_close_waits_for_bounded_wire_cleanup(context, monkeypatch):
    async def exercise():
        class SlowClose(Wire):
            async def aclose(self):
                self.closing.set()
                try:
                    await self.finish_close.wait()
                finally:
                    await super().aclose()
        wire = SlowClose()
        wire.closing, wire.finish_close = asyncio.Event(), asyncio.Event()
        semaphore = asyncio.Semaphore(1)
        client = AsyncOpenAI(api_key='fixture', base_url='http://fixture.invalid/v1',
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(
                lambda req: httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=wire))))
        monkeypatch.setattr('agent.auxiliary_client._to_async_client', lambda sync, model, **kw: (client, model))
        monkeypatch.setattr('agent.auxiliary_client._acquire_async_aux_semaphore', lambda task: semaphore)
        stream = context.llm.astream([{'role': 'user', 'content': 'hi'}], task='realtime')
        await stream.__aenter__()
        await anext(stream)
        first = asyncio.create_task(stream.aclose())
        await asyncio.wait_for(wire.closing.wait(), 3)
        second = asyncio.create_task(stream.aclose())
        try:
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(first, 5)
            await asyncio.wait_for(second, 3)
            assert wire.closed
            await asyncio.wait_for(semaphore.acquire(), 2)
            semaphore.release()
            await stream.aclose()
            assert not client.is_closed()
        finally:
            wire.finish_close.set()
            await asyncio.gather(first, second, return_exceptions=True)
            await client.close()
    asyncio.run(exercise())
