"""Messaging /new keeps a queued automation wake and runs it on the fresh session, as main's #114456
kept a parked internal wake across /new; it is never stranded queued on the abandoned owner (N83).
Real GatewayRunner, DiscordAdapter, SessionAuthority and drain over SQLite; the turn is a stand-in."""
import asyncio

import pytest

from tests.waiting import async_wait_until


@pytest.mark.asyncio
@pytest.mark.parametrize('settles_mid_reset', [False, True])
async def test_new_moves_a_queued_wake_to_the_fresh_session(monkeypatch, settles_mid_reset):
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_automation import admit_route_wake
    from gateway.session_results import execution_result
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
    running, finish = asyncio.Event(), asyncio.Event()
    executed = []

    class Agent:
        def interrupt(self, *args, **kwargs):
            pass

    async def turn(event, source, quick_key, run_generation):
        executed.append((event.text, runner.session_store.peek_session_id(quick_key)))
        if event.text == 'A':
            runner._session_state(quick_key).turn.agent = Agent()
            running.set()
            await finish.wait()
        captured = execution_result.get()
        if captured is not None:
            captured['result'] = {'final_response': 'reply', 'messages': []}
        return 'reply'
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
    await adapter.handle_message(MessageEvent(text='A', source=source, message_id='A'))
    await asyncio.wait_for(running.wait(), 10)
    old = runner.session_store.peek_session_id(key)
    assert await admit_route_wake(runner, adapter, source, 'WAKE', 'kanban-wake')
    if settles_mid_reset:
        # The old turn settles after the rotation, before the wake moves: its drain wakes up there.
        live = authority.sessions[authority.logical_owner(old)]
        blocked = []

        class ClaimLock(asyncio.Lock):
            async def acquire(self):
                if finish.is_set() and asyncio.current_task() is live.task:
                    blocked.append(self.locked())  # the old drain's next claim, after the old turn settled
                return await super().acquire()
        live.mutation_lock = ClaimLock()
        reset = runner.async_session_store.reset_session

        def old_statuses(wake):
            return [r['status'] for r in list_session_admissions(authority.db, session_id=str(old), pending_only=False)
                    if r['principal_id'].startswith('automation:') == wake]

        async def reset_then_settle(session_key):
            entry = await reset(session_key)
            finish.set()
            # The window the old drain must NOT claim in: it waits behind /new, or (unlocked) has
            # already taken the wake off the old FIFO.
            await async_wait_until(lambda: any(blocked) or old_statuses(True) != ['queued'],
                                   timeout=10, what='the old drain to try its next claim')
            assert old_statuses(False) == ['terminal']
            return entry
        monkeypatch.setattr(runner.async_session_store, 'reset_session', reset_then_settle)
    await adapter.handle_message(MessageEvent(text='/new', source=source, message_id='new'))
    new = runner.session_store.peek_session_id(key)
    assert new != old
    finish.set()
    await async_wait_until(lambda: any(text == 'WAKE' for text, _ in executed), timeout=15,
                           what='the wake to run after /new')
    rows = list_session_admissions(authority.db, session_id=old, pending_only=False)
    assert not [r for r in rows if r['status'] == 'queued'], rows
    assert [sid for text, sid in executed if text == 'WAKE'] == [new], 'the wake must run once, on the fresh session'
