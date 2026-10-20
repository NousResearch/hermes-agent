"""An internal wake never interrupts or steers an admitted user turn (N44, sequence 1).

Main's busy handler queued every internal synthetic event (Kanban wakes, /loop ticks, process
notifications) behind the running turn. Under the session authority a busy adapter hands every
non-command to ``dispatch_shared_busy``; the authority declines internal input, so the wake fell
through to the classic busy path, which in the default ``interrupt`` mode ended the user's turn.
Real GatewayRunner, DiscordAdapter, SessionAuthority and drain over SQLite; the user turn is a
stand-in holding a real running-agent slot.
"""
import asyncio

import pytest

from tests.waiting import async_wait_until


@pytest.mark.asyncio
async def test_kanban_wake_during_an_admitted_turn_queues_without_interrupting(monkeypatch):
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_results import execution_result
    from gateway.wake import deliver_wake
    from hermes_state_runtime import list_session_admissions
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '42')
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='fixture-token', typing_indicator=False))
    sent = []

    async def send(chat_id, content, reply_to=None, metadata=None):
        sent.append(content)
        return SendResult(success=True, message_id=f's{len(sent)}')
    monkeypatch.setattr(adapter, 'send', send)
    adapter.gateway_runner = runner  # as GatewayRunner._create_adapter binds every adapter
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='42', chat_type='dm', user_id='42')
    key = runner.session_store._generate_session_key(source)
    running, finish = asyncio.Event(), asyncio.Event()
    interrupts, executed = [], []

    class Agent:
        def interrupt(self, message=None, *args, **kwargs):
            interrupts.append(message)

    async def turn(event, source, quick_key, run_generation):
        executed.append(event.text)
        if event.text == 'A':
            runner._session_state(quick_key).turn.agent = Agent()  # the built, running agent
            running.set()
            await finish.wait()
        captured = execution_result.get()
        if captured is not None:
            captured['result'] = {'final_response': 'reply ' + event.text, 'messages': []}
        return 'reply ' + event.text
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)

    await adapter.handle_message(MessageEvent(text='A', source=source, message_id='A'))
    await asyncio.wait_for(running.wait(), 10)
    wake = '[Kanban task completed] t_1'
    await asyncio.wait_for(deliver_wake(adapter, text=wake, source=source), 10)  # accepted, no raise
    assert interrupts == [], 'the internal wake interrupted the admitted user turn'
    finish.set()
    await async_wait_until(lambda: wake in executed, what='the wake turn to run')
    owner = runner.session_store.peek_session_id(key)

    def rows():
        return list_session_admissions(authority.db, session_id=owner, pending_only=False)
    await async_wait_until(lambda: [r['status'] for r in rows()] == ['terminal', 'terminal'],
                           what='both admissions to settle')
    # The wake ran as its own FIFO turn after the user's, never inside or instead of it.
    assert [(r['principal_id'].split(':')[0], r['outcome']) for r in rows()] == [
        ('messaging', 'completed'), ('automation', 'completed')]
    assert executed == ['A', wake]


@pytest.mark.asyncio
async def test_idle_loop_tick_runs_as_a_fifo_turn_not_beside_it(monkeypatch):
    """N44 sequence 2: an idle /loop tick started a classic turn outside the FIFO, so a drain
    claiming a user row meanwhile handed that row's text to the tick's turn and settled it
    ``completed`` without running it. The tick must commit as its own FIFO admission."""
    from types import SimpleNamespace

    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_ingress import executing_admission
    from gateway.session_results import execution_result
    from hermes_cli import loops
    from hermes_state_runtime import list_session_admissions
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '42')
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='fixture-token', typing_indicator=False))
    adapter.gateway_runner = runner

    async def send(chat_id, content, reply_to=None, metadata=None):
        from gateway.platforms.base import SendResult
        return SendResult(success=True, message_id='sent')
    monkeypatch.setattr(adapter, 'send', send)
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='42', chat_type='dm', user_id='42')
    key = runner.session_store._generate_session_key(source)
    started = []

    async def turn(event, source, quick_key, run_generation):
        started.append((event.text, executing_admission.get()))
        captured = execution_result.get()  # None outside the drain
        if captured is not None:
            captured['result'] = {'final_response': 'ok', 'messages': []}
        return 'ok'
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
    await adapter.handle_message(MessageEvent(text='hello', source=source, message_id='m1'))

    def rows():
        owner = runner.session_store.peek_session_id(key)
        return list_session_admissions(authority.db, session_id=owner, pending_only=False) if owner else []
    await async_wait_until(lambda: [r['status'] for r in rows()] == ['terminal'] and not runner._is_session_running(key),
                           what='the first turn to settle and release the session')
    owner = runner.session_store.peek_session_id(key)

    class Loop:
        state = SimpleNamespace(ticks_fired=1)

        def __init__(self, session_id=None):
            pass

        def is_due(self, now):
            return True

        def fire_tick(self):
            return 'loop tick prompt'

        def abandon_tick(self):
            raise AssertionError('the tick was not admitted')
    monkeypatch.setattr(loops, 'LoopManager', Loop)
    monkeypatch.setattr(loops, 'goal_blocks_loop_tick', lambda sid: False)
    route = {'platform': 'discord', 'chat_id': '42', 'chat_type': 'dm', 'user_id': '42'}
    await runner._loop_wakeup_fire_one(owner, SimpleNamespace(awaiting_response=False, next_due_at=0, route=route),
                                       1e12, set())
    await async_wait_until(lambda: len(started) >= 2, what='the loop tick turn to start')
    assert started == [('hello', True), ('loop tick prompt', True)], 'the tick ran outside the FIFO'
    await async_wait_until(lambda: [r['status'] for r in rows()] == ['terminal', 'terminal'],
                           what='both admissions to settle')
