"""Live policy and task ownership must still authorize deferred inference."""

import asyncio
import os
from pathlib import Path

import httpx
import pytest
import hermes_yaml as yaml
from openai import AsyncOpenAI

from tests.agent import test_plugin_llm_stream as stream_fixtures

context = stream_fixtures.context


MESSAGES = [{'role': 'user', 'content': 'hi'}]


def configure(**policy):
    path = Path(os.environ['HERMES_HOME']) / 'config.yaml'
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    config['plugins'] = {'entries': {'stream-fixture': {'llm': policy}}}
    path.write_text(yaml.safe_dump(config), encoding='utf-8')


def sdk_client(monkeypatch, requests):
    def handle(request):
        requests.append(request)
        return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                              content=b'data: [DONE]\n\n')
    client = AsyncOpenAI(api_key='fixture', base_url='http://fixture.invalid/v1',
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
    monkeypatch.setattr('agent.auxiliary_client._to_async_client',
                        lambda sync, model, **kw: (client, model))
    return client


def test_held_stream_rechecks_revoked_model_override(context, monkeypatch):
    from agent.plugin_llm import PluginLlmTrustError

    async def exercise():
        requests = []
        client = sdk_client(monkeypatch, requests)
        configure(allow_model_override=True)
        stream = context.llm.astream(MESSAGES, task='realtime', model='override')
        configure(allow_model_override=False)
        try:
            with pytest.raises(PluginLlmTrustError):
                async with stream:
                    pytest.fail('revoked override reached inference')
            assert requests == []
        finally:
            await client.close()
    asyncio.run(exercise())


@pytest.mark.parametrize('wait_at', ['routing', 'permit'])
@pytest.mark.parametrize('revocation', ['provider_flag', 'model_flag', 'provider_allowlist',
    'model_allowlist', 'agent_id', 'profile', 'builtin_task', 'owned_task', 'home_scope'])
def test_revocation_during_wait_denies_dispatch_and_releases_permit(
        context, monkeypatch, wait_at, revocation):
    import threading
    import agent.auxiliary_client as ac
    from agent.plugin_llm import PluginLlmTrustError
    from agent.plugin_llm_stream import PluginLlmStreamError

    async def exercise():
        requests = []
        client = sdk_client(monkeypatch, requests)
        reached = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        policy = dict(allow_provider_override=True, allow_model_override=True,
            allow_agent_id_override=True, allow_profile_override=True, allow_task_override=True,
            allowed_providers=['custom'], allowed_models=['override'])
        kwargs = {'task': 'realtime'}
        if revocation.startswith('provider'):
            kwargs['provider'] = 'custom'
        elif revocation.startswith('model'):
            kwargs['model'] = 'override'
        elif revocation in ('agent_id', 'profile'):
            kwargs[revocation] = 'fixture-override'
        elif revocation == 'builtin_task':
            kwargs['task'] = 'vision'
        configure(**policy)
        stream = context.llm.astream(MESSAGES, **kwargs)

        class Permit(asyncio.Semaphore):
            async def acquire(self):
                if wait_at == 'permit':
                    reached.set()
                return await super().acquire()
        permit = Permit(0 if wait_at == 'permit' else 1)
        monkeypatch.setattr(ac, '_acquire_async_aux_semaphore', lambda task: permit)
        original = ac._plan_aux_call
        if wait_at == 'routing':
            def blocked(*args, **kw):
                result = original(*args, **kw)
                loop.call_soon_threadsafe(reached.set)
                if not release.wait(5):
                    raise TimeoutError('test did not release resolver')
                return result
            monkeypatch.setattr(ac, '_plan_aux_call', blocked)
        opening = asyncio.create_task(stream.__aenter__())
        try:
            await asyncio.wait_for(reached.wait(), 5)
            if revocation == 'home_scope':
                monkeypatch.setenv('HERMES_HOME', str(Path(os.environ['HERMES_HOME']) / 'other-profile'))
            elif revocation == 'owned_task':
                context._manager._aux_tasks.clear()
            elif revocation.endswith('_allowlist'):
                policy['allowed_' + revocation.split('_')[0] + 's'] = ['different']
            else:
                flag = {'provider_flag': 'provider', 'model_flag': 'model',
                        'builtin_task': 'task'}.get(revocation, revocation)
                policy['allow_' + flag + '_override'] = False
            if revocation != 'home_scope':
                configure(**policy)
            release.set()
            if wait_at == 'permit':
                permit.release()
            error = PluginLlmStreamError if revocation == 'home_scope' else PluginLlmTrustError
            with pytest.raises(error):
                await opening
            assert requests == []
            assert not permit.locked(), 'denial must release the acquired permit'
            assert stream._closed
        finally:
            release.set()
            if not opening.done():
                opening.cancel()
            await asyncio.gather(opening, return_exceptions=True)
            await stream.aclose()
            await client.close()
    asyncio.run(exercise())
