"""A native Bot message spends its loop budget on arrival, not again during FIFO execution."""
import asyncio

import pytest
import pytest_asyncio

from tests.waiting import async_wait_until


@pytest_asyncio.fixture
async def bot_owner(tmp_path, monkeypatch):
    from gateway.bot_loop_guard import BotLoopGuard, BotLoopGuardSettings
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setenv('DISCORD_ALLOW_BOTS', 'all')
    monkeypatch.setenv('DISCORD_ALLOWED_USERS', 'human')
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    runner._busy_input_mode = 'queue'
    runner._bot_loop_guard = BotLoopGuard(
        settings=lambda: BotLoopGuardSettings(max_events=2, window_seconds=300, cooldown_seconds=600),
        clock=lambda: 1000.0,
    )
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='fixture-token', typing_indicator=False))
    adapter.gateway_runner, runner.adapters = runner, {Platform.DISCORD: adapter}

    async def send(chat_id, content, **kwargs):
        return SendResult(success=True, message_id='reply')

    monkeypatch.setattr(adapter, 'send', send)
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='chat', chat_type='group', user_id='peer')
    source.is_bot = True
    try:
        yield runner, adapter, authority, source
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('busy', [False, True], ids=['idle', 'queued-behind-running-turn'])
async def test_fifo_spends_one_budget_unit_per_bot_message(bot_owner, monkeypatch, busy):
    from gateway.platforms.event import MessageEvent
    from gateway.session_ingress import executing_admission
    from gateway.session_results import execution_result
    from hermes_state_runtime import list_session_admissions

    runner, adapter, authority, source = bot_owner
    turns = []
    running, finish = asyncio.Event(), asyncio.Event()

    async def turn(event, *_args):
        turns.append((event.message_id, executing_admission.get()))
        if event.message_id == 'first' and busy:
            running.set()
            await finish.wait()
        execution_result.get()['result'] = {'final_response': 'received', 'messages': []}
        return 'received'

    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
    await adapter.handle_message(MessageEvent(text='first', source=source, message_id='first'))
    if busy:
        await asyncio.wait_for(running.wait(), 10)
    else:
        await async_wait_until(lambda: turns == [('first', True)], what='first Bot turn')
        await async_wait_until(lambda: not runner._is_session_running(
            runner.session_store._generate_session_key(source)), what='first turn to settle')
    await adapter.handle_message(MessageEvent(text='second', source=source, message_id='second'))
    if busy:
        session_id = runner.session_store.peek_session_id(runner.session_store._generate_session_key(source))
        await async_wait_until(lambda: len(list_session_admissions(authority.db, session_id=session_id)) == 2,
                               what='second Bot message committed behind the running turn')
    finish.set()
    await async_wait_until(lambda: len(turns) == 2, what='second Bot turn within the two-message budget')
    await async_wait_until(lambda: not runner._is_session_running(
        runner.session_store._generate_session_key(source)), what='second turn to settle')
    assert turns == [('first', True), ('second', True)]
    assert await runner._hm_admit_event(MessageEvent(text='third', source=source, message_id='third')) is None
    assert runner._bot_loop_guard.blocked(runner._bot_loop_guard_conversation(source))


@pytest.mark.asyncio
async def test_fifo_execution_rechecks_bot_authorization(bot_owner, monkeypatch):
    from gateway.platforms.event import MessageEvent
    from gateway.session_authority import SessionAuthority
    from gateway.session_contract import SessionRef
    from gateway.session_results import execution_result
    from hermes_state_runtime import list_session_admissions

    runner, adapter, authority, source = bot_owner
    turns = []
    schedule = SessionAuthority._schedule
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)

    async def turn(event, *_args):
        turns.append(event.message_id)
        execution_result.get()['result'] = {'final_response': 'received', 'messages': []}
        return 'received'

    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
    await adapter.handle_message(MessageEvent(text='queued', source=source, message_id='queued'))
    key = runner.session_store._generate_session_key(source)
    await async_wait_until(lambda: runner.session_store.peek_session_id(key), what='native session creation')
    session_id = runner.session_store.peek_session_id(key)
    await async_wait_until(lambda: list_session_admissions(authority.db, session_id=session_id),
                           what='Bot input committed to SQLite')
    monkeypatch.setenv('DISCORD_ALLOW_BOTS', 'none')
    monkeypatch.setattr(SessionAuthority, '_schedule', schedule)
    authority._schedule(SessionRef(authority.profile_id, session_id))
    await async_wait_until(lambda: authority.sessions[session_id].task.done(),
                           what='revoked Bot input to pause the FIFO')
    assert turns == []
    assert [row['status'] for row in list_session_admissions(authority.db, session_id=session_id)] == ['queued']
