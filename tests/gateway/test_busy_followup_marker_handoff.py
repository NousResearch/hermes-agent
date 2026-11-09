"""A busy follow-up's crash marker survives until its reply is in the delivery ledger (N49).

The first message's adapter lifecycle adopts its turn's durable active-turn marker and releases
it only once the reply is ledgered, so a kill between the terminal commit and the ledger write
leaves a marked turn whose persisted reply the next boot sends. A follow-up sent while the
session is busy goes through ``dispatch_shared_busy`` instead and must get the same hand-off.
Real GatewayRunner, adapter, SessionAuthority, drain and ``execute_admission`` over SQLite; only
the agent turn is a stand-in that marks the turn active like ``_hmwa_prepare_turn`` does.
"""
import asyncio

import pytest

from tests.gateway.test_shared_busy_receive_loop import Adapter, _statuses, _wait_for


@pytest.mark.asyncio
async def test_busy_followup_keeps_its_marker_until_the_reply_is_ledgered(monkeypatch):
    from gateway.config import GatewayConfig, Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_results import execution_result

    monkeypatch.setenv('TELEGRAM_ALLOWED_USERS', 'fixture-user')
    runner = GatewayRunner(GatewayConfig())
    adapter = Adapter()
    adapter.gateway_runner = runner  # as GatewayRunner._create_adapter binds every adapter
    runner.adapters = {Platform.TELEGRAM: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='busy-chat', chat_type='dm', user_id='fixture-user')
    key = runner.session_store._generate_session_key(source)
    gate = asyncio.Event()

    async def turn(event, source, quick_key, run_generation):
        await runner._mark_durable_active_turn(event, quick_key)
        if event.text == 'one':
            await gate.wait()
        execution_result.get()['result'] = {'final_response': 'reply to ' + event.text, 'messages': []}
        return 'reply to ' + event.text
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)

    marker_at_ledger = {}
    record = adapter._record_delivery_obligation

    async def ledger(event, session_key, text, *args, **kwargs):
        # A kill right here must still leave this turn marked: the reply is not yet owed by a row.
        marker_at_ledger[text] = runner.session_store._entries[key].active_turn_token
        return await record(event, session_key, text, *args, **kwargs)
    monkeypatch.setattr(adapter, '_record_delivery_obligation', ledger)

    await adapter.handle_message(MessageEvent(text='one', source=source, message_id='m1'))
    await _wait_for(lambda: _statuses(authority).get('one') == 'started')
    await asyncio.wait_for(adapter.handle_message(MessageEvent(text='two', source=source, message_id='m2')), 5)
    gate.set()
    await _wait_for(lambda: 'reply to two' in adapter.sent and not adapter._background_tasks, 15)
    assert adapter.sent == ['reply to one', 'reply to two']
    assert marker_at_ledger['reply to one'] is not None
    assert marker_at_ledger['reply to two'] is not None, 'the busy follow-up released its marker before the ledger write'
    await _wait_for(lambda: runner.session_store._entries[key].active_turn_token is None)
