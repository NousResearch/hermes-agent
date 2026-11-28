"""Strict stream routing and dispatch-bound validation, using host planning."""

import asyncio
from pathlib import Path

import httpx
import pytest
from openai import AsyncOpenAI

from tests.agent import test_plugin_llm_stream as stream_fixtures

context = stream_fixtures.context


def test_unavailable_primary_does_not_dispatch_configured_fallback(context, monkeypatch):
    import agent.auxiliary_client as ac

    home = Path(__import__('os').environ['HERMES_HOME'])
    (home / 'config.yaml').write_text(
        'auxiliary:\n  realtime:\n    provider: anthropic\n    model: claude-fixture\n'
        '    fallback_chain:\n      - provider: custom\n        model: fallback-model\n'
        '        base_url: http://fixture.invalid/v1\n', encoding='utf-8')
    resolutions, requests = [], []
    async def exercise():
        def handle(request):
            requests.append(request)
            return httpx.Response(200, headers={'content-type': 'text/event-stream'}, content=b'data: [DONE]\n\n')
        client = AsyncOpenAI(api_key='fixture', base_url='http://fixture.invalid/v1',
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        def get_client(provider, model=None, **kw):
            resolutions.append(provider)
            return (None, model) if provider == 'anthropic' else (client, model)
        monkeypatch.setattr(ac, '_get_cached_client', get_client)
        monkeypatch.setattr(ac, '_to_async_client', lambda sync, model, **kw: (client, model))
        try:
            with pytest.raises(RuntimeError, match='unavailable|not supported'):
                async with context.llm.astream([{'role': 'user', 'content': 'hi'}], task='realtime'):
                    pass
            assert requests == []
            assert resolutions == ['anthropic']
        finally:
            await client.close()
    asyncio.run(exercise())


@pytest.mark.parametrize('name', ['timeout', 'max_tokens', 'max_output_chars'])
@pytest.mark.parametrize('value', [0, -1, True, float('inf'), float('nan'), 'bad'])
def test_invalid_bounds_rejected_before_host_planning(context, monkeypatch, name, value):
    plans = []
    def forbidden(*args, **kw):
        plans.append(kw)
        raise AssertionError('invalid bound reached provider planning')
    monkeypatch.setattr('agent.auxiliary_client._plan_aux_call', forbidden)
    async def exercise():
        with pytest.raises(ValueError, match=name):
            async with context.llm.astream([{'role': 'user', 'content': 'hi'}], task='realtime', **{name: value}):
                pass
        assert plans == []
    asyncio.run(exercise())


@pytest.mark.parametrize('value', [0, -1, True, float('inf'), float('nan'), 'bad'])
def test_invalid_configured_timeout_rejected_before_planning(context, monkeypatch, value):
    home = Path(__import__('os').environ['HERMES_HOME'])
    scalar = {0: '0', -1: '-1', True: 'true'}.get(value, str(value))
    if scalar in ('inf', 'nan'):
        scalar = '.' + scalar
    (home / 'config.yaml').write_text('auxiliary:\n  realtime:\n    timeout: ' + scalar + '\n', encoding='utf-8')
    plans = []
    def forbidden(*args, **kw):
        plans.append(kw)
        raise AssertionError('invalid default reached provider planning')
    monkeypatch.setattr('agent.auxiliary_client._plan_aux_call', forbidden)
    async def exercise():
        with pytest.raises(ValueError, match='timeout'):
            async with context.llm.astream([{'role': 'user', 'content': 'hi'}], task='realtime'):
                pass
        assert plans == []
    asyncio.run(exercise())
