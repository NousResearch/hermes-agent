"""Plugin streams exercise the real host resolver and SDK without remote inference."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from openai import AsyncOpenAI


@pytest.fixture
def context(tmp_path, monkeypatch):
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    import agent.auxiliary_client as ac

    home = Path(__import__('os').environ['HERMES_HOME'])
    (home / 'config.yaml').write_text(
        'model:\n  provider: custom\n  model: active-model\n'
        '  base_url: http://fixture.invalid/v1\n'
        'auxiliary:\n  realtime:\n    provider: custom\n    model: fast-model\n'
        '    base_url: http://fixture.invalid/v1\n', encoding='utf-8')
    monkeypatch.setenv('OPENAI_API_KEY', 'test-only-not-a-credential')
    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name='stream-fixture'), manager)
    ctx.register_auxiliary_task('realtime', display_name='Realtime', description='Fixture inference')
    monkeypatch.setattr('hermes_cli.plugins.get_plugin_auxiliary_tasks', lambda: list(manager._aux_tasks.values()))
    ac._client_cache.clear()
    monkeypatch.setattr(ac, '_openai_http_client_kwargs', lambda *args, **kw: {})
    yield ctx
    ac._client_cache.clear()


class Wire(httpx.AsyncByteStream):
    def __init__(self):
        self.release = asyncio.Event()
        self.waiting = asyncio.Event()
        self.closed = False
        self.closed_event = asyncio.Event()

    async def __aiter__(self):
        def event(choices, **kw):
            value = dict(id='fixture', object='chat.completion.chunk', created=0,
                         model='fast-model', choices=choices, **kw)
            return ('data: ' + json.dumps(value) + '\n\n').encode()
        yield event([dict(index=0, delta=dict(content='first'), finish_reason=None)])
        self.waiting.set()
        await self.release.wait()
        yield event([dict(index=0, delta={}, finish_reason='length')])
        yield event([], usage=dict(prompt_tokens=3, completion_tokens=1, total_tokens=4))
        yield b'data: [DONE]\n\n'

    async def aclose(self):
        self.closed = True
        self.closed_event.set()
        self.release.set()


def test_async_first_delta_precedes_completion_barrier(context, monkeypatch):
    import agent.auxiliary_client as ac

    async def exercise():
        wire = Wire()
        requests = []
        def handle(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=wire)
        client = AsyncOpenAI(api_key='fixture', base_url='http://fixture.invalid/v1',
                             http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        # Patch the SDK construction seam, NOT task/model resolution or the facade.
        monkeypatch.setattr('agent.auxiliary_client._to_async_client', lambda sync, model, **kw: (client, model))
        async with context.llm.astream([{'role': 'user', 'content': 'hi', '_db_persisted': True}], task='realtime') as stream:
            first = await anext(stream)
            assert first.type == 'text_delta' and first.text == 'first'
            assert not wire.release.is_set()
            wire.release.set()
            tail = [event async for event in stream]
        assert tail[-1].type == 'end'
        assert tail[-1].finish_reason == 'length'
        assert tail[-1].usage.total_tokens == 4
        assert tail[-1].audit['task'] == 'realtime'
        assert requests[0]['model'] == 'fast-model'
        assert requests[0]['stream'] is True
        assert '_db_persisted' not in requests[0]['messages'][0]
        assert wire.closed
        assert not client.is_closed()
        await client.close()
    asyncio.run(exercise())


def test_deadline_includes_blocked_host_resolution(context, monkeypatch):
    import threading
    import agent.auxiliary_client as ac
    release = threading.Event()
    finished = threading.Event()
    original = ac._plan_aux_call
    def blocked(*args, **kw):
        try:
            release.wait(5)
            return original(*args, **kw)
        finally:
            finished.set()
    monkeypatch.setattr(ac, '_plan_aux_call', blocked)
    async def exercise():
        started = asyncio.get_running_loop().time()
        try:
            with pytest.raises(TimeoutError):
                async with context.llm.astream([{'role': 'user', 'content': 'hi'}], task='realtime', timeout=0.1):
                    pytest.fail('must not open after routing exceeds deadline')
            assert asyncio.get_running_loop().time() - started < 2
        finally:
            release.set()
    asyncio.run(exercise())
    assert finished.wait(5)


def test_aggregating_host_adapter_is_refused_before_create(context, monkeypatch):
    from agent.plugin_llm_stream import PluginLlmStreamError
    import agent.auxiliary_client as ac
    from types import SimpleNamespace
    called = []
    adapter = ac.AsyncAnthropicAuxiliaryClient.__new__(ac.AsyncAnthropicAuxiliaryClient)
    async def create(**kw):
        called.append(kw)
        return SimpleNamespace(choices=[])
    adapter.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
    monkeypatch.setattr(ac, '_resolve_call_client', lambda *args, **kw: ac._ResolvedAuxRoute(
        adapter, 'claude-fixture', 'anthropic', 'anthropic'))
    async def exercise():
        with pytest.raises(PluginLlmStreamError, match='not supported'):
            async with context.llm.astream([{'role': 'user', 'content': 'hi'}], task='realtime'):
                pytest.fail('must refuse aggregating transport')
    asyncio.run(exercise())
    assert called == []


def test_explicit_close_cancels_blocked_read_not_shared_client(context, monkeypatch):
    async def exercise():
        wire = Wire()
        client = AsyncOpenAI(api_key='fixture', base_url='http://fixture.invalid/v1',
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(
                lambda req: httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=wire))))
        monkeypatch.setattr('agent.auxiliary_client._to_async_client', lambda sync, model, **kw: (client, model))
        async with context.llm.astream([{'role': 'user', 'content': 'hi'}], task='realtime') as stream:
            await anext(stream)
            pending = asyncio.create_task(anext(stream))
            await asyncio.wait_for(wire.waiting.wait(), 3)
            await stream.aclose()
            with pytest.raises(asyncio.CancelledError):
                await pending
        assert wire.closed and not client.is_closed()
        await client.close()
    asyncio.run(exercise())


def test_deadline_closes_abandoned_stream_without_another_pull(context, monkeypatch):
    async def exercise():
        wire = Wire()
        client = AsyncOpenAI(api_key='fixture', base_url='http://fixture.invalid/v1',
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(
                lambda req: httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=wire))))
        monkeypatch.setattr('agent.auxiliary_client._to_async_client', lambda sync, model, **kw: (client, model))
        async with context.llm.astream([{'role': 'user', 'content': 'hi'}], task='realtime', timeout=2) as stream:
            await anext(stream)
            await asyncio.wait_for(wire.closed_event.wait(), 5)
            with pytest.raises(TimeoutError):
                await anext(stream)
        assert not client.is_closed()
        await client.close()
    asyncio.run(exercise())


def test_close_during_unopened_request_cancels_dispatch(context, monkeypatch):
    async def exercise():
        entered = asyncio.Event()
        cancelled = asyncio.Event()
        never = asyncio.Event()
        async def handle(req):
            entered.set()
            try:
                await never.wait()
            finally:
                cancelled.set()
        client = AsyncOpenAI(api_key='fixture', base_url='http://fixture.invalid/v1',
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        monkeypatch.setattr('agent.auxiliary_client._to_async_client', lambda sync, model, **kw: (client, model))
        stream = context.llm.astream([{'role': 'user', 'content': 'hi'}], task='realtime')
        opening = asyncio.create_task(stream.__aenter__())
        await asyncio.wait_for(entered.wait(), 3)
        try:
            await stream.aclose()
            await asyncio.wait_for(cancelled.wait(), 2)
            with pytest.raises(asyncio.CancelledError):
                await opening
        finally:
            opening.cancel()
            await client.close()
    asyncio.run(exercise())


def test_active_agent_turn_is_explicitly_unsupported(context, monkeypatch):
    from agent.plugin_llm_stream import PluginLlmStreamError
    monkeypatch.setattr('agent.relay_runtime.active_turn', lambda: object())
    async def forbidden(*args):
        pytest.fail('must refuse before routing when managed streaming is unavailable')
    monkeypatch.setattr('agent.plugin_llm_stream._prepare_bounded', forbidden)
    async def exercise():
        with pytest.raises(PluginLlmStreamError, match='managed agent turn'):
            async with context.llm.astream([{'role': 'user', 'content': 'hi'}]):
                pytest.fail('must not bypass Relay')
    asyncio.run(exercise())


def test_profile_change_while_reading_closes_owned_stream(context, monkeypatch, tmp_path):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from agent.plugin_llm_stream import PluginLlmStreamError
    async def exercise():
        wire = Wire()
        client = AsyncOpenAI(api_key='fixture', base_url='http://fixture.invalid/v1',
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(
                lambda req: httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=wire))))
        monkeypatch.setattr('agent.auxiliary_client._to_async_client', lambda sync, model, **kw: (client, model))
        async with context.llm.astream([{'role': 'user', 'content': 'hi'}], task='realtime') as stream:
            await anext(stream)
            token = set_hermes_home_override(tmp_path / 'other-profile')
            try:
                with pytest.raises(PluginLlmStreamError, match='profile changed'):
                    await anext(stream)
                assert wire.closed
            finally:
                reset_hermes_home_override(token)
        await client.close()
    asyncio.run(exercise())


def test_stream_event_types_are_public():
    from agent.plugin_llm import PluginLlmStreamEvent, PluginLlmStreamError
    assert PluginLlmStreamEvent('text_delta', text='one').text == 'one'
    assert issubclass(PluginLlmStreamError, RuntimeError)


@pytest.mark.parametrize('kwargs', [dict(provider='foreign'), dict(model='foreign'),
    dict(profile='foreign'), dict(agent_id='foreign'), dict(task='foreign')])
def test_stream_trust_rejects_before_host_resolution(context, monkeypatch, kwargs):
    from agent.plugin_llm import PluginLlmTrustError
    monkeypatch.setattr('agent.auxiliary_client._plan_aux_call', lambda *a, **k: pytest.fail('must not route'))
    with pytest.raises(PluginLlmTrustError):
        context.llm.astream([{'role': 'user', 'content': 'hi'}], **kwargs)


@pytest.mark.parametrize('finish', ['stop', 'length', 'tool_calls', None])
def test_terminal_reason_is_genuine_and_payload_is_text_only(context, monkeypatch, finish):
    from agent.plugin_llm_stream import PluginLlmStreamError
    async def exercise():
        choices = [dict(index=0, delta=dict(content='visible', reasoning_content='hidden',
            tool_calls=[dict(index=0, id='call', type='function', function=dict(name='never_execute', arguments='{}'))]),
            finish_reason=finish)]
        payload = dict(id='fixture', object='chat.completion.chunk', created=0, model='fast-model', choices=choices)
        wire = ('data: ' + json.dumps(payload) + '\n\ndata: [DONE]\n\n').encode()
        client = AsyncOpenAI(api_key='fixture', base_url='http://fixture.invalid/v1',
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(
                lambda req: httpx.Response(200, headers={'content-type': 'text/event-stream'}, content=wire))))
        monkeypatch.setattr('agent.auxiliary_client._to_async_client', lambda sync, model, **kw: (client, model))
        async with context.llm.astream([{'role': 'user', 'content': 'hi'}], task='realtime') as stream:
            event = await anext(stream)
            assert event.text == 'visible'
            assert 'hidden' not in repr(event) and 'never_execute' not in repr(event)
            if finish is None:
                with pytest.raises(PluginLlmStreamError, match='without a finish reason'):
                    await anext(stream)
            else:
                terminal = await anext(stream)
                assert terminal.type == 'end' and terminal.finish_reason == finish
                with pytest.raises(StopAsyncIteration):
                    await anext(stream)
        await client.close()
    asyncio.run(exercise())


def test_bound_output_closes_without_success_terminal(context, monkeypatch):
    from agent.plugin_llm_stream import PluginLlmStreamError
    async def exercise():
        wire = Wire()
        client = AsyncOpenAI(api_key='fixture', base_url='http://fixture.invalid/v1',
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(
                lambda req: httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=wire))))
        monkeypatch.setattr('agent.auxiliary_client._to_async_client', lambda sync, model, **kw: (client, model))
        async with context.llm.astream([{'role': 'user', 'content': 'hi'}], task='realtime', max_output_chars=3) as stream:
            with pytest.raises(PluginLlmStreamError, match='max_output_chars'):
                await anext(stream)
            assert wire.closed
        await client.close()
    asyncio.run(exercise())


def test_closing_one_stream_does_not_cancel_another(context, monkeypatch):
    async def exercise():
        first, second = Wire(), Wire()
        wires = iter([first, second])
        client = AsyncOpenAI(api_key='fixture', base_url='http://fixture.invalid/v1',
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(
                lambda req: httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=next(wires)))))
        monkeypatch.setattr('agent.auxiliary_client._to_async_client', lambda sync, model, **kw: (client, model))
        async with context.llm.astream([{'role': 'user', 'content': 'one'}], task='realtime') as a:
            async with context.llm.astream([{'role': 'user', 'content': 'two'}], task='realtime') as b:
                await anext(a)
                await anext(b)
                await a.aclose()
                assert first.closed and not second.closed and not client.is_closed()
                second.release.set()
                assert (await anext(b)).finish_reason == 'length'
        await client.close()
    asyncio.run(exercise())


def test_profile_change_before_open_refuses_provider(context, monkeypatch, tmp_path):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from agent.plugin_llm_stream import PluginLlmStreamError
    stream = context.llm.astream([{'role': 'user', 'content': 'hi'}], task='realtime')
    monkeypatch.setattr('agent.auxiliary_client._plan_aux_call', lambda *a, **k: pytest.fail('must not route'))
    async def exercise():
        token = set_hermes_home_override(tmp_path / 'other-profile')
        try:
            with pytest.raises(PluginLlmStreamError, match='profile changed'):
                async with stream:
                    pytest.fail('must not open')
        finally:
            reset_hermes_home_override(token)
    asyncio.run(exercise())


def test_provider_error_after_visible_delta_is_not_retried_or_exposed(context, monkeypatch):
    from agent.plugin_llm import PluginLlmStreamError
    class ErrorWire(Wire):
        async def __aiter__(self):
            yield b'data: {"id":"x","object":"chat.completion.chunk","created":0,"model":"fast-model","choices":[{"index":0,"delta":{"content":"first"},"finish_reason":null}]}\n\n'
            yield b'data: {"error":{"message":"private provider detail","type":"fixture_error"}}\n\n'
    async def exercise():
        wire = ErrorWire()
        requests = []
        def handle(req):
            requests.append(req)
            return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=wire)
        client = AsyncOpenAI(api_key='fixture', base_url='http://fixture.invalid/v1',
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        monkeypatch.setattr('agent.auxiliary_client._to_async_client', lambda sync, model, **kw: (client, model))
        async with context.llm.astream([{'role': 'user', 'content': 'hi'}], task='realtime') as stream:
            assert (await anext(stream)).text == 'first'
            with pytest.raises(PluginLlmStreamError) as error:
                await anext(stream)
            assert 'private provider detail' not in str(error.value)
            assert not hasattr(error.value, 'request')
            assert error.value.__context__ is None
        assert len(requests) == 1 and wire.closed
        await client.close()
    asyncio.run(exercise())


def test_host_task_concurrency_wait_obeys_deadline(context, monkeypatch):
    async def exercise():
        semaphore = asyncio.Semaphore(0)
        monkeypatch.setattr('agent.auxiliary_client._acquire_async_aux_semaphore', lambda task: semaphore)
        async def forbidden(*args):
            pytest.fail('must wait for the host task permit before routing')
        monkeypatch.setattr('agent.plugin_llm_stream._prepare_bounded', forbidden)
        with pytest.raises(TimeoutError):
            async with context.llm.astream([{'role': 'user', 'content': 'hi'}], task='realtime', timeout=0.1):
                pytest.fail('must not dispatch without permit')
        assert semaphore.locked()
    asyncio.run(exercise())


def test_deadline_interrupts_blocked_next(context, monkeypatch):
    async def exercise():
        wire = Wire()
        client = AsyncOpenAI(api_key='fixture', base_url='http://fixture.invalid/v1',
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(
                lambda req: httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=wire))))
        monkeypatch.setattr('agent.auxiliary_client._to_async_client', lambda sync, model, **kw: (client, model))
        async with context.llm.astream([{'role': 'user', 'content': 'hi'}], task='realtime', timeout=0.1) as stream:
            assert (await anext(stream)).text == 'first'
            started = asyncio.get_running_loop().time()
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(anext(stream), 3)
            assert asyncio.get_running_loop().time() - started < 2
        assert wire.closed
        assert not client.is_closed()
        await client.close()
    asyncio.run(exercise())
