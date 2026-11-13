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
    from hermes_agent.gateway.config import GatewayConfig, Platform
    from hermes_agent.gateway.platforms.event import MessageEvent, ProcessingOutcome
    from hermes_agent.gateway.run import GatewayRunner
    from hermes_agent.gateway.session_authority import initialize_session_authority
    from hermes_agent.gateway.session_results import execution_result

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


async def _busy_hook_owner(monkeypatch, adapter):
    from hermes_agent.gateway.config import GatewayConfig, Platform
    from hermes_agent.gateway.run import GatewayRunner
    from hermes_agent.gateway.session_authority import initialize_session_authority
    from hermes_agent.gateway.session_results import execution_result

    monkeypatch.setenv('TELEGRAM_ALLOWED_USERS', 'fixture-user')
    runner = GatewayRunner(GatewayConfig())
    adapter.gateway_runner = runner
    runner.adapters = {Platform.TELEGRAM: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter)
    gate = asyncio.Event()

    async def turn(event, source, quick_key, run_generation):
        if event.text == 'one':
            await gate.wait()
        execution_result.get()['result'] = {'final_response': 'reply to ' + event.text, 'messages': []}
        return 'reply to ' + event.text
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
    return authority, adapter.build_source(chat_id='busy-chat', chat_type='dm', user_id='fixture-user'), gate


@pytest.mark.asyncio
async def test_busy_followup_success_waits_for_its_reply_and_a_failed_send_reports_failure(monkeypatch):
    """The follow-up's observer sends its reply after settlement: SUCCESS only once that send
    succeeded, FAILURE when it failed (a thumbs-up on an unanswered message misleads the user)."""
    from hermes_agent.gateway.platforms.base import SendResult
    from hermes_agent.gateway.platforms.event import MessageEvent, ProcessingOutcome

    send_gate, send_entered = asyncio.Event(), asyncio.Event()

    class HeldSendAdapter(HookAdapter):
        async def send(self, chat_id, content, reply_to=None, metadata=None):
            if content == 'reply to two':
                send_entered.set()  # settlement (and the drain's own completion report) came first
                await send_gate.wait()
                self.sent.append(content)
                return SendResult(success=False, error='network down')
            return await super().send(chat_id, content, reply_to=reply_to, metadata=metadata)

    adapter = HeldSendAdapter()
    authority, source, gate = await _busy_hook_owner(monkeypatch, adapter)
    await adapter.handle_message(MessageEvent(text='one', source=source, message_id='m1'))
    await _wait_for(lambda: _statuses(authority).get('one') == 'started')
    await asyncio.wait_for(adapter.handle_message(MessageEvent(text='two', source=source, message_id='m2')), 5)
    gate.set()
    await asyncio.wait_for(send_entered.wait(), 15)
    assert [h for h in adapter.hooks if h[0] == 'm2'] == [('m2', 'start')], 'completed before its reply was sent'
    send_gate.set()
    await _wait_for(lambda: not adapter._background_tasks, 15)
    await _wait_for(lambda: any(h[0] == 'm2' and h[1] == 'complete' for h in adapter.hooks))
    assert [h for h in adapter.hooks if h[0] == 'm2'] == [('m2', 'start'), ('m2', 'complete', ProcessingOutcome.FAILURE)]


@pytest.mark.asyncio
async def test_busy_followup_released_by_a_pause_keeps_its_lifecycle_when_it_runs(monkeypatch):
    """A pause releases the follow-up's observer with a notice while its row stays queued. The turn
    still owes its lifecycle: when the queue moves on it gets its start and completion hooks."""
    from hermes_agent.gateway.platforms.event import MessageEvent, ProcessingOutcome
    from hermes_agent.gateway.session_contract import SessionRef

    adapter = HookAdapter()
    authority, source, gate = await _busy_hook_owner(monkeypatch, adapter)
    await adapter.handle_message(MessageEvent(text='one', source=source, message_id='m1'))
    await _wait_for(lambda: _statuses(authority).get('one') == 'started')
    await asyncio.wait_for(adapter.handle_message(MessageEvent(text='two', source=source, message_id='m2')), 5)
    await _wait_for(lambda: _statuses(authority).get('two') == 'queued')
    (session_id,) = authority.sessions
    authority._pause(SessionRef(authority.profile_id, session_id), 'unknown_execution')
    await _wait_for(lambda: len(adapter.sent) == 1)  # the observer's pause notice
    gate.set()
    await _wait_for(lambda: _statuses(authority).get('two') == 'terminal', 15)
    await _wait_for(lambda: ('m2', 'complete', ProcessingOutcome.SUCCESS) in adapter.hooks)
    assert [h for h in adapter.hooks if h[0] == 'm2'] == [('m2', 'start'), ('m2', 'complete', ProcessingOutcome.SUCCESS)]

@pytest.mark.asyncio
async def test_rewritten_busy_followup_preserves_processing_hooks(monkeypatch):
    """A pre_gateway_dispatch rewrite must retain a busy follow-up's executor-owned lifecycle."""
    from hermes_agent.gateway.platforms.event import MessageEvent, ProcessingOutcome

    async def rewrite(name, **kwargs):
        if name == 'pre_gateway_dispatch' and kwargs['event'].text == 'two':
            return [{'action': 'rewrite', 'text': 'rewritten two'}]
        return []

    monkeypatch.setattr('hermes_cli.lifecycle.ainvoke_hook', rewrite)
    adapter = HookAdapter()
    authority, source, gate = await _busy_hook_owner(monkeypatch, adapter)
    await adapter.handle_message(MessageEvent(text='one', source=source, message_id='m1'))
    await _wait_for(lambda: _statuses(authority).get('one') == 'started')
    await asyncio.wait_for(adapter.handle_message(
        MessageEvent(text='two', source=source, message_id='m2')), 5)
    await _wait_for(lambda: _statuses(authority).get('rewritten two') == 'queued')
    assert not any(h[0] == 'm2' for h in adapter.hooks)
    gate.set()
    await _wait_for(lambda: 'reply to rewritten two' in adapter.sent and not adapter._background_tasks, 15)
    assert [h for h in adapter.hooks if h[0] == 'm2'] == [
        ('m2', 'start'), ('m2', 'complete', ProcessingOutcome.SUCCESS)]
