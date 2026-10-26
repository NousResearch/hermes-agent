"""Messaging /stop and /new reach the authority's claimed turn and its queued follow-up.

N45: a /stop while the claimed turn still builds its agent (the slot holds the pending sentinel)
must latch on the authority, so the agent the turn adopts starts interrupted and never runs tools.
N53: an idle /stop or /new (between turns: slot released, next row not yet claimed) must still
cancel the queued messaging follow-up. Real GatewayRunner, DiscordAdapter, SessionAuthority and
drain over a real SessionDB; only agent construction/conversation is a stand-in.
"""
import asyncio

import pytest
from tests.waiting import async_wait_until


def _owner(monkeypatch):
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.run import GatewayRunner
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '42')
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='fixture-token', typing_indicator=False))
    sent = []

    async def send(chat_id, content, reply_to=None, metadata=None):
        from gateway.platforms.base import SendResult
        sent.append(content)
        return SendResult(success=True, message_id=f's{len(sent)}')
    monkeypatch.setattr(adapter, 'send', send)
    runner.adapters = {Platform.DISCORD: adapter}
    return runner, adapter, sent


async def _until(predicate, timeout=10):
    await async_wait_until(predicate, timeout=timeout)


@pytest.mark.asyncio
async def test_stop_during_agent_construction_latches_so_the_turn_runs_no_tools(monkeypatch):
    from gateway.platforms.event import MessageEvent
    from gateway.run_turn_runner import TurnRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_results import execution_result
    from gateway.turn_context import TurnContext
    from hermes_state_runtime import list_session_admissions

    runner, adapter, sent = _owner(monkeypatch)
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='42', chat_type='dm', user_id='42')
    key = runner.session_store._generate_session_key(source)
    constructing, built = asyncio.Event(), asyncio.Event()
    tools_ran = []

    class Agent:
        interrupted = False

        def interrupt(self, *args, **kwargs):
            self.interrupted = True

        def clear_interrupt(self):
            self.interrupted = False

    async def turn(event, source, quick_key, run_generation):
        # Agent construction: the slot holds the pending sentinel until the agent is adopted.
        constructing.set()
        await built.wait()
        ctx = TurnContext(session_id=runner.session_store.peek_session_id(key), session_key=key, source=source)
        owner, sid, generation = TurnRunner(runner, ctx)._approval_owner
        agent = Agent()
        owner.adopt_agent(sid, generation, agent)  # what TurnRunner's agent wiring does
        if agent.interrupted:
            result = {'final_response': '', 'messages': [], 'interrupted': True}
        else:
            tools_ran.append(event.text)  # run_conversation: the turn's tools execute
            result = {'final_response': 'done', 'messages': []}
        execution_result.get()['result'] = result
        return result['final_response'] or None
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)

    await adapter.handle_message(MessageEvent(text='A', source=source, message_id='A'))
    await asyncio.wait_for(constructing.wait(), 10)
    await adapter.handle_message(MessageEvent(text='/stop', source=source, message_id='stop'))
    await _until(lambda: any('Stopped' in text for text in sent))
    built.set()
    owner_id = runner.session_store.peek_session_id(key)
    await _until(lambda: [r['status'] for r in list_session_admissions(
        authority.db, session_id=owner_id, pending_only=False)] == ['terminal'])
    rows = list_session_admissions(authority.db, session_id=owner_id, pending_only=False)
    assert tools_ran == [], 'the stopped turn still ran its tools'
    assert [(r['request_id'], r['outcome']) for r in rows] == [('A', 'interrupted')]


@pytest.mark.asyncio
@pytest.mark.parametrize('command', ['/stop', '/new'])
async def test_idle_stop_and_new_cancel_the_queued_followup(monkeypatch, command):
    from gateway.platforms.event import MessageEvent
    from gateway.session_authority import SessionAuthority, initialize_session_authority
    from gateway.session_contract import SessionRef
    from gateway.session_ingress import admit_message
    from gateway.session_ingress_context import native_callback
    from hermes_constants import get_hermes_home
    from hermes_state_runtime import list_session_admissions

    runner, adapter, _sent = _owner(monkeypatch)
    # The gap between turns: the slot is released and the drain has not claimed B yet.
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    monkeypatch.setattr(runner, '_read_user_config', lambda: {'approvals': {'destructive_slash_confirm': False}})
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='42', chat_type='dm', user_id='42')
    follow_up = MessageEvent(text='B', source=source, message_id='B')
    with native_callback(runner, follow_up, get_hermes_home()):
        waiter = asyncio.create_task(admit_message(authority, follow_up))
        await _until(lambda: authority.waiters)
    key = runner.session_store._generate_session_key(source)
    owner = runner.session_store.peek_session_id(key)
    assert not runner._is_session_running(key)

    stop = MessageEvent(text=command, source=source)
    with native_callback(runner, stop, get_hermes_home()):
        assert await runner._handle_message(stop) is not None
    assert await asyncio.wait_for(waiter, 5) is None, 'the cancelled follow-up must get no reply'
    rows = list_session_admissions(authority.db, session_id=owner, pending_only=False)
    assert [(r['request_id'], r['status'], r['outcome']) for r in rows] == [('B', 'terminal', 'cancelled')]
    executed = []

    async def handle(event):
        executed.append(event.text)
        return 'ran'
    monkeypatch.setattr(runner, '_handle_message', handle)
    await SessionAuthority._drain(authority, SessionRef(authority.profile_id, owner))
    assert executed == []
