"""Read-only candidate probes. Network denied by OS sandbox; all remote evidence synthetic.

Assertions express desired properties (not assertions that defects exist).
The SDK/fallback probes fake only the innermost HTTP transport, not Bot.send_message.
"""
import asyncio
import gc
import importlib.util
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import weakref

import httpx
import pytest
import pytest_asyncio

from telegram.error import TimedOut

from gateway.live_todo import _surfaces
from hermes_cli.plugins import get_plugin_manager
from tools.todo_tool import TodoStore

# Byte-preserved reviewer assertions; only the repository locator is portable.
HOST = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('review_candidate_harness', HOST / 'tests/gateway/test_todo_progress_integration.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


@pytest_asyncio.fixture
async def manager(tmp_path, monkeypatch):
    home = tmp_path / 'synthetic-profile'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    topics = ['7', '8', '71', '72', '73',
              *(str(topic) for topic in range(100, 132)),
              '941', '942', '951', '952', '961', '962']
    scope = {
        'routes': [dict(profile='default', platform='telegram', chat_id='-100', thread_id=topic)
                   for topic in topics],
        'task_resources': [],
    }
    import yaml
    (home / 'config.yaml').write_text(yaml.safe_dump({
        'plugins': {'enabled': ['hermes-telegram-experience'], 'entries': {
            'hermes-telegram-experience': {'settings': {'enabled': True, 'scope': scope}},
        }},
    }))
    manager = get_plugin_manager()
    manager.discover_and_load()
    assert manager._live_todo_registration.active
    yield manager
    manager.unload()
    await asyncio.sleep(0)
    sources = [v for k, v in tuple(_surfaces.items()) if k[0] == str(home)]
    for source in sources:
        await source.finish()
        _surfaces.pop(source.binding.surface, None)  # only synthetic fixture state
    await asyncio.sleep(0)


def new_lane(bot, topic='7'):
    adapter = h._telegram_adapter(bot)
    harness = h.GatewayHarness(adapter)
    agent = h.ToolAgent(TodoStore())
    ctx, source = h._owner(harness, agent, chat_id='-100', thread_id=topic, current=lambda: True)
    return adapter, harness, agent, ctx, source


@pytest.mark.asyncio
async def test_real_worker_commit_handoff_and_snapshot_stability(manager):
    bot = h.StubTelegramBot()
    adapter, harness, agent, ctx, source = new_lane(bot)
    callback = agent.tool_progress_callback
    threads = []
    def traced_callback(*args, **kwargs):
        if args[0] == 'tool.completed':
            threads.append(threading.get_ident())
        return callback(*args, **kwargs)
    agent.tool_progress_callback = traced_callback
    task = asyncio.create_task(source.run())
    await asyncio.to_thread(h._execute_todo, agent, 'worker-1', {'id': 'a', 'content': 'from worker', 'status': 'pending'})
    await h._wait_until(lambda: source.message_id is not None)
    assert threads and all(t != threading.get_ident() for t in threads)
    await asyncio.to_thread(h._execute_todo, agent, 'worker-2', {'id': 'a', 'content': 'latest worker', 'status': 'completed'})
    await h._wait_until(lambda: len(bot.edited) == 1)
    assert bot.sent[0]['text'] == 'Conversation steps\n○ from worker'
    assert bot.edited[0]['text'] == 'Conversation steps\n✓ latest worker'
    assert len(bot.sent) == 1 and bot.edited[0]['message_id'] == int(source.message_id)
    await source.finish()
    await task


class WireFixture(httpx.AsyncBaseTransport):
    def __init__(self, source, calls, first=None):
        self.source = source
        self.calls = calls
        self.first = first

    async def handle_async_request(self, request):
        self.calls.append({'host': request.url.host, 'method': request.method,
                           'admitted': self.source.admitted(), 'registration_active': self.source.registration.active})
        if self.first is not None:
            entered, release = self.first
            entered.set()
            await release.wait()
            raise httpx.ConnectTimeout('synthetic connection never established', request=request)
        # Explicit synthetic Telegram receipt; no network is available.
        return httpx.Response(200, request=request, json={'ok': True, 'result': {
            'message_id': 301, 'date': 1, 'chat': {'id': -100, 'type': 'supergroup'}, 'text': 'synthetic'}})

    async def aclose(self):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize('pause', ['fallback-lock', 'connect-retry'])
async def test_effective_unload_prevents_later_real_sdk_transport_attempt(manager, monkeypatch, pause):
    from plugins.platforms.telegram import adapter as adapter_module
    from plugins.platforms.telegram import telegram_network
    monkeypatch.setattr(adapter_module, 'resolve_proxy_url', lambda *a, **kw: None)
    monkeypatch.setattr(telegram_network, '_resolve_proxy_url', lambda *a, **kw: None)
    adapter, harness, agent, ctx, source = new_lane(h.StubTelegramBot())
    ips = ['149.154.166.110', '149.154.167.220']
    monkeypatch.setattr(adapter, '_fallback_ips', lambda: ips)
    general, updates = await adapter._build_ptb_requests()  # production construction; no initialize/getMe/poll
    transport = general._client._transport
    assert isinstance(transport, telegram_network.TelegramFallbackTransport)
    app = (adapter_module.Application.builder().token('123:synthetic-review-only')
           .request(general).get_updates_request(updates).build())
    bot = app.bot  # the same ExtBot construction as production; never initialize/poll
    # Open against the actual SDK client, not the temporary harness client.
    await source.finish()
    adapter._bot = bot
    source = harness._run_agent_create_todo_progress_owner(h._display({}), ctx)
    assert source.client is bot
    calls = []
    entered, release = asyncio.Event(), asyncio.Event()
    await transport._primary.aclose()
    transport._primary = WireFixture(source, calls)
    transport._fallbacks = {
        ips[0]: WireFixture(source, calls, (entered, release) if pause == 'connect-retry' else None),
        ips[1]: WireFixture(source, calls),
    }
    if pause == 'fallback-lock':
        await transport._fallback_lock.acquire()
    task = asyncio.create_task(source.run())
    try:
        await asyncio.to_thread(h._execute_todo, agent, 'transport', {'id': 'a', 'content': 'SDK boundary probe', 'status': 'pending'})
        if pause == 'fallback-lock':
            await h._wait_until(lambda: source.inflight)
            assert calls == []
        else:
            await asyncio.wait_for(entered.wait(), 5)
        # Real native unload, not source.close or a mocked capability.
        assert manager.unload('hermes-telegram-experience')
        assert not source.admitted() and not source.registration.active
        if pause == 'fallback-lock':
            transport._fallback_lock.release()
        else:
            release.set()
        await asyncio.wait_for(task, 5)
        print('SDK_WIRE_TRACE', pause, json.dumps(calls), 'outcome', source.last_outcome)
        assert not [c for c in calls if not c['admitted']], 'A new inner HTTP attempt occurred after effective native unload'
    finally:
        if transport._fallback_lock.locked():
            transport._fallback_lock.release()
        release.set()
        await source.finish()
        await general.shutdown()
        await updates.shutdown()


@pytest.mark.asyncio
async def test_unknown_quarantine_does_not_pin_entire_completed_turn(manager):
    class UncertainBot(h.StubTelegramBot):
        async def send_message(self, **kwargs):
            self.sent.append(kwargs)
            raise TimedOut('synthetic ambiguous dispatch')

    async def complete_unknown(topic):
        adapter, harness, agent, ctx, source = new_lane(UncertainBot(), topic)
        task = asyncio.create_task(source.run())
        await asyncio.to_thread(h._execute_todo, agent, 'unknown-' + topic, {'id': 'a', 'content': 'synthetic retained task data', 'status': 'pending'})
        await task
        await source.finish()
        assert source.unknown and not source.inflight
        assert source not in source.registration.sources and source not in adapter._live_todo_sources
        assert _surfaces[source.binding.surface] is source
        return weakref.ref(ctx), weakref.ref(agent), source.binding.surface

    refs = [await complete_unknown(topic) for topic in ('71', '72', '73')]
    await asyncio.sleep(0)
    gc.collect()
    retained = [key for ctxref, agentref, key in refs if ctxref() is not None or agentref() is not None]
    print('UNKNOWN_RETAINED_TURNS', len(retained), retained)
    # Causality control: remove only these synthetic tombstones and collect again.
    for _, _, key in refs:
        _surfaces.pop(key, None)
    gc.collect()
    assert all(ctxref() is None and agentref() is None for ctxref, agentref, _ in refs)
    print('UNKNOWN_REMOVAL_CONTROL', 'all synthetic contexts and agents released')
    assert retained == [], 'Settled unknown tombstones retain complete TurnContext/agent, not just fencing metadata'


@pytest.mark.asyncio
@pytest.mark.parametrize('phase', ['factory', 'cleanup'])
async def test_optional_ui_exception_cannot_abort_normal_turn_lifecycle(manager, monkeypatch, phase):
    bot = h.StubTelegramBot()
    adapter, harness, agent, ctx, source = new_lane(bot)
    error = None
    if phase == 'factory':
        await source.finish()
        def broken_factory(handle):
            raise RuntimeError('injected presentation construction failure')
        monkeypatch.setattr(manager._live_todo_registration, 'factory', broken_factory)
        try:
            harness._run_agent_create_todo_progress_owner(h._display({}), ctx)
        except Exception as exc:
            error = exc
        assert error is None, 'Optional consumer factory exception escapes the real gateway setup boundary: ' + repr(error)
    else:
        original_close = source.consumer.close
        def broken_close():
            raise RuntimeError('injected presentation close failure')
        monkeypatch.setattr(source.consumer, 'close', broken_close)
        tasks = [asyncio.create_task(asyncio.sleep(3600)) for _ in range(3)]
        try:
            try:
                await harness._run_agent_cleanup_turn_tasks(ctx, progress_task=None, log_task=None,
                    interrupt_monitor=tasks[0], _notify_task=tasks[1], tracking_task=tasks[2], stream_task=None)
            except Exception as exc:
                error = exc
            assert harness.released == [(ctx.session_key, ctx.run_generation)], 'Optional close failure bypasses session release: ' + repr(error)
        finally:
            monkeypatch.setattr(source.consumer, 'close', original_close)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await source.finish()
