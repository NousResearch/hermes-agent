"""An adapter slot is not proof a heartbeat reached execution."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from hermes_cli.heartbeat import HeartbeatManager, HeartbeatState, save_heartbeat
from evals.heartbeat_idle_wire import WireAdapter


@pytest.mark.asyncio
async def test_cancelled_admission_is_refunded_but_started_execution_is_not():
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._run_in_executor_with_context = asyncio.to_thread
    adapter = WireAdapter(PlatformConfig(enabled=True, typing_indicator=False), Platform.TELEGRAM)
    adapter.wire = []
    runner._adapter_for_source = lambda source: adapter
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='42', user_id='42')
    key = build_session_key(source)
    watch = {key: (source, 'cancel-test')}
    started = asyncio.Event()

    async def model(**kwargs):
        started.set()
        await asyncio.Event().wait()

    runner._run_agent = model
    runner.hooks = SimpleNamespace(emit=AsyncMock())
    runner._hmwa_resolve_session = AsyncMock(return_value=(source, SimpleNamespace(session_id="cancel-test"), key))
    runner._hmwa_prepare_turn = AsyncMock(return_value=(
        runner._PreparedTurn([], "", "check", "check", None, None), {}))
    runner._clear_session_env = lambda tokens: None

    async def handler(event):
        return await runner._handle_message_with_agent(event, source, key, 1)

    adapter.set_message_handler(handler)
    await runner._warm_goals_session_db('test')
    for cancel_before_start in (True, False):
        save_heartbeat('cancel-test', HeartbeatState(prompt='check', interval_seconds=60, created_at=1))
        await runner._heartbeat_poll_once(watch)
        task = adapter._session_tasks[key]
        if not cancel_before_start:
            await asyncio.wait_for(started.wait(), 2)
        await adapter.cancel_session_processing(key)
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)  # task callbacks settle the exact attempt
        assert HeartbeatManager('cancel-test').state.fire_count == int(not cancel_before_start)


@pytest.mark.asyncio
async def test_runner_rejection_does_not_consume_tick_or_overwrite_replacement():
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._run_in_executor_with_context = asyncio.to_thread
    adapter = WireAdapter(PlatformConfig(enabled=True, typing_indicator=False), Platform.TELEGRAM)
    adapter.wire = []
    runner._adapter_for_source = lambda source: adapter
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='42', user_id='42')
    key = build_session_key(source)
    watch = {key: (source, 'rejected-test')}
    await runner._warm_goals_session_db('test')

    async def denied(event):
        return 'not authorized'

    adapter.set_message_handler(denied)
    save_heartbeat('rejected-test', HeartbeatState(prompt='check', interval_seconds=60, created_at=1))
    await runner._heartbeat_poll_once(watch)
    await asyncio.gather(*adapter._background_tasks)
    await asyncio.sleep(0)
    assert HeartbeatManager('rejected-test').state.fire_count == 0

    async def replaced(event):
        HeartbeatManager('rejected-test').set('replacement', 120)
        return None

    adapter.set_message_handler(replaced)
    await runner._heartbeat_poll_once(watch)
    await asyncio.gather(*adapter._background_tasks)
    await asyncio.sleep(0)
    state = HeartbeatManager('rejected-test').state
    assert state.prompt == 'replacement' and state.fire_count == 0
