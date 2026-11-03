"""A user message admitted while a classic fallback wake holds the route's slot waits queued and runs
as its own admitted turn; it is never handed to the wake's busy path and booked completed unrun (N82).
Real GatewayRunner, DiscordAdapter, SessionAuthority and drain over SQLite; the turn is a stand-in."""
import asyncio

import pytest

from tests.waiting import async_wait_until


@pytest.mark.asyncio
@pytest.mark.parametrize('stop', [False, True])
async def test_message_during_classic_wake_runs_after_it_not_inside_it(monkeypatch, stop):
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_ingress import executing_admission
    from gateway.session_results import execution_result
    from gateway.wake import deliver_wake
    from hermes_state_runtime import list_session_admissions
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '42')
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='fixture-token', typing_indicator=False))

    async def send(chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id='sent')
    monkeypatch.setattr(adapter, 'send', send)
    adapter.gateway_runner = runner
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='42', chat_type='dm', user_id='42')
    key = runner.session_store._generate_session_key(source)
    release = asyncio.Event()
    started = []
    wake = '[Kanban task completed] t_9'

    class Agent:
        def interrupt(self, *args, **kwargs):
            pass

    async def turn(event, source, quick_key, run_generation):
        started.append((event.text, executing_admission.get()))
        if event.text == wake:
            runner._session_state(quick_key).turn.agent = Agent()
            await release.wait()
        captured = execution_result.get()
        if captured is not None:
            captured['result'] = {'final_response': 'ok', 'messages': []}
        return 'ok'
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)

    await runner._handle_reset_command(MessageEvent(text='/new', source=source))  # fresh route: no native row
    fallback = asyncio.create_task(deliver_wake(adapter, text=wake, source=source))
    await async_wait_until(lambda: started, what='the classic fallback wake to hold the slot')
    assert started == [(wake, False)]
    await adapter.handle_message(MessageEvent(text='B', source=source, message_id='B'))
    sid = runner.session_store.peek_session_id(key)

    def rows():
        return [(r['request_id'], r['status'], r['outcome'])
                for r in list_session_admissions(authority.db, session_id=sid, pending_only=False)]
    await async_wait_until(lambda: rows(), what='B to be admitted')
    if stop:
        await adapter.handle_message(MessageEvent(text='/stop', source=source, message_id='stop'))
    release.set()
    await fallback
    if stop:
        await async_wait_until(lambda: rows() == [('B', 'terminal', 'cancelled')], what='B to be cancelled')
        assert ('B', True) not in started and ('B', False) not in started
        return
    await async_wait_until(lambda: ('B', True) in started, timeout=15, what='B to run as its own admitted turn')
    await async_wait_until(lambda: rows() == [('B', 'terminal', 'completed')], what='B to settle')
