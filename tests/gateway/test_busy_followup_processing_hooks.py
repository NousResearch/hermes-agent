"""A busy follow-up gets the adapter processing hooks once, when it runs.

The first message is bracketed by ``BasePlatformAdapter._process_message_background``
(``on_processing_start`` / ``on_processing_complete``). A message sent while that turn runs goes
through ``dispatch_shared_busy`` and the authority's drain runs it, so no adapter task brackets it:
the executor must, exactly once, and not before the queued turn starts executing.
Real GatewayRunner, adapter, SessionAuthority, drain and ``execute_admission`` over SQLite; only
the agent turn is a stand-in.
"""
import asyncio

import pytest

from tests.gateway.test_shared_busy_receive_loop import Adapter, _statuses, _wait_for


class HookAdapter(Adapter):
    def __init__(self):
        super().__init__()
        self.hooks = []

    async def on_processing_start(self, event):
        self.hooks.append((event.message_id, 'start'))

    async def on_processing_complete(self, event, outcome):
        self.hooks.append((event.message_id, 'complete', outcome))


@pytest.mark.asyncio
async def test_busy_followup_fires_processing_hooks_once(monkeypatch):
    from gateway.config import GatewayConfig, Platform
    from gateway.platforms.event import MessageEvent, ProcessingOutcome
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_results import execution_result

    monkeypatch.setenv('TELEGRAM_ALLOWED_USERS', 'fixture-user')
    runner = GatewayRunner(GatewayConfig())
    adapter = HookAdapter()
    adapter.gateway_runner = runner
    runner.adapters = {Platform.TELEGRAM: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='busy-chat', chat_type='dm', user_id='fixture-user')
    gate = asyncio.Event()

    async def turn(event, source, quick_key, run_generation):
        if event.text == 'one':
            await gate.wait()
        execution_result.get()['result'] = {'final_response': 'reply to ' + event.text, 'messages': []}
        return 'reply to ' + event.text
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)

    await adapter.handle_message(MessageEvent(text='one', source=source, message_id='m1'))
    await _wait_for(lambda: _statuses(authority).get('one') == 'started')
    await asyncio.wait_for(adapter.handle_message(MessageEvent(text='two', source=source, message_id='m2')), 5)
    await _wait_for(lambda: _statuses(authority).get('two') == 'queued')
    assert ('m2', 'start') not in adapter.hooks, 'a queued message is acknowledged when it runs, not when queued'
    gate.set()
    await _wait_for(lambda: 'reply to two' in adapter.sent and not adapter._background_tasks, 15)
    await _wait_for(lambda: ('m2', 'complete', ProcessingOutcome.SUCCESS) in adapter.hooks)
    assert sorted(adapter.hooks, key=lambda h: (h[0], h[1] != 'start')) == [
        ('m1', 'start'), ('m1', 'complete', ProcessingOutcome.SUCCESS),
        ('m2', 'start'), ('m2', 'complete', ProcessingOutcome.SUCCESS)]
    assert adapter.hooks.index(('m2', 'start')) < adapter.hooks.index(('m2', 'complete', ProcessingOutcome.SUCCESS))
