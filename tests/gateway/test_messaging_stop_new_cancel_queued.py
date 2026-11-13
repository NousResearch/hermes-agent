"""Messaging /stop and /new drop the durable queued follow-up, as main dropped the adapter slot.

Under session authority every busy follow-up is admitted to the durable FIFO, so clearing the
adapter's pending slot no longer reached it: /stop let the drain RUN the message the user meant
to cancel, and /new stranded it queued on the abandoned session. Real GatewayRunner busy
dispatch, real DiscordAdapter, real authority over a real SessionDB; only the turn executor is
recorded.
"""
import asyncio

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize('command', ['/stop', '/new'])
async def test_busy_stop_and_new_cancel_the_queued_messaging_followup(tmp_path, monkeypatch, command):
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_authority import SessionAuthority, initialize_session_authority
    from gateway.session_ingress import admit_message
    from gateway.session_ingress_context import native_callback
    from hermes_constants import get_hermes_home
    from hermes_state_runtime import list_session_admissions
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '42')
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='fixture-token', typing_indicator=False))
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='42', chat_type='dm', user_id='42')

    # Turn A is running; follow-up B is committed to the durable FIFO behind it.
    follow_up = MessageEvent(text='B', source=source, message_id='B')
    with native_callback(runner, follow_up, get_hermes_home()):
        waiter = asyncio.create_task(admit_message(authority, follow_up))
        for _ in range(200):
            if authority.waiters:
                break
            await asyncio.sleep(0.01)
    key = runner.session_store._generate_session_key(source)
    owner = runner.session_store.peek_session_id(key)
    assert [r['status'] for r in list_session_admissions(authority.db, session_id=owner)] == ['queued']

    class RunningAgent:
        interrupted = []

        def interrupt(self, message=None):
            self.interrupted.append(message)

    runner._session_state(key).turn.agent = RunningAgent()
    with native_callback(runner, MessageEvent(text=command, source=source), get_hermes_home()):
        reply = await runner._handle_message(MessageEvent(text=command, source=source))
    assert reply is not None

    # B is settled silently (no pause notice, no reply) and never runs.
    assert await asyncio.wait_for(waiter, 5) is None
    rows = list_session_admissions(authority.db, session_id=owner, pending_only=False)
    assert [(r['request_id'], r['status'], r['outcome']) for r in rows] == [('B', 'terminal', 'cancelled')]
    executed = []

    async def handle(event):
        executed.append(event.text)
        return 'ran'
    monkeypatch.setattr(runner, '_handle_message', handle)
    from gateway.session_contract import SessionRef
    await SessionAuthority._drain(authority, SessionRef(authority.profile_id, owner))
    assert executed == []
