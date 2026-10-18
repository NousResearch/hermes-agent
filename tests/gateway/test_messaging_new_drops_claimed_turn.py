"""A /new between the drain's claim and the turn's session lookup drops the claimed turn.

Main's /new cancelled the adapter task running the turn. Under the session authority the turn runs
in the drain, so without a fence it resolved the route's fresh session and ran there, where no owner
adopts its agent and the Stop that /new sent is lost. That holds for a viewer's submit into a chat
too. A move that sends no Stop (/resume) keeps main's behaviour, and a compression is no move even
with a Stop latched (the turn reaches the tip, where adoption fires it). Real GatewayRunner,
DiscordAdapter, SessionAuthority, drain and session resolver over a real SessionDB; preparation is
a stand-in.
"""
import asyncio

import pytest

from tests.waiting import async_wait_until


@pytest.mark.asyncio
@pytest.mark.parametrize('case', ['new', 'viewer_new', 'resume', 'stop_compression'])
async def test_route_move_after_claim(monkeypatch, case):
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal, Submission
    from gateway.session_ingress import executing_admission
    from gateway.session_results import execution_result
    from hermes_state_runtime import list_session_admissions
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '42')
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='fixture-token', typing_indicator=False))

    sent = []

    async def send(chat_id, content, reply_to=None, metadata=None):
        sent.append(content)
        return SendResult(success=True, message_id='s')
    monkeypatch.setattr(adapter, 'send', send)
    runner.adapters = {Platform.DISCORD: adapter}
    monkeypatch.setattr(runner, '_read_user_config', lambda: {'approvals': {'destructive_slash_confirm': False}})
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='42', chat_type='dm', user_id='42')
    key, store, db = runner.session_store._generate_session_key(source), runner.session_store, authority.db
    entered, gate = asyncio.Event(), asyncio.Event()
    ran_on = []
    resolve = runner._hmwa_resolve_session

    async def held_resolve(event, source):  # hold the claimed turn before its session lookup
        if executing_admission.get() and not entered.is_set():
            entered.set()
            await gate.wait()
        return await resolve(event, source)
    monkeypatch.setattr(runner, '_hmwa_resolve_session', held_resolve)

    async def prepare(event, source, session_entry, *args):
        ran_on.append(session_entry.session_id)
        execution_result.get()['result'] = {'final_response': 'ok', 'messages': []}
        return None, None
    monkeypatch.setattr(runner, '_hmwa_prepare_turn', prepare)

    if case == 'viewer_new':
        ref = authority.register(source)
        viewer = Principal('authenticated-viewer', 'default', frozenset({'session:submit'}), 'viewer')
        submit = asyncio.create_task(authority.submit(viewer, Submission('A', ref, {'text': 'A'}, 'queue')))
    else:
        await adapter.handle_message(MessageEvent(text='A', source=source, message_id='A'))
    await asyncio.wait_for(entered.wait(), 10)
    claimed = store.peek_session_id(key)
    if case in ('new', 'viewer_new'):  # the reply comes once the reset finished (a background task here)
        await adapter.handle_message(MessageEvent(text='/new', source=source, message_id='new'))
        await async_wait_until(lambda: sent, what='the /new reply')
    elif case == 'resume':  # an idle /resume before the turn's busy guard: no Stop is sent
        db.create_session('R', source='discord')
        assert store.switch_session(key, 'R', expected_session_id=claimed) is not None
    else:  # a compression continuation, and a /stop in the same window
        db.create_session('tip', source='discord', parent_session_id=claimed)
        db.end_session(claimed, end_reason='compression')
        assert store.advance_compression_session(key, claimed, 'tip') is not None
        await adapter.handle_message(MessageEvent(text='/stop', source=source, message_id='stop'))
        assert claimed in authority.pending_stops  # latched for this claim, so the check runs
    await async_wait_until(lambda: store.peek_session_id(key) != claimed, what='the route to move')
    moved = store.peek_session_id(key)
    gate.set()

    def settled():
        rows = list_session_admissions(db, session_id=claimed, pending_only=False)
        return rows and rows[0]['status'] == 'terminal' and rows[0]
    await async_wait_until(settled, timeout=10, what='the claimed turn to settle')
    if case in ('new', 'viewer_new'):
        assert ran_on == [], f'the claimed turn ran on the fresh session {moved}'
        assert settled()['outcome'] == 'interrupted'
    else:
        assert ran_on == [moved], f'the claimed turn was dropped instead of running on {moved}'
    if case == 'viewer_new':
        await asyncio.wait_for(submit, 10)
