"""Native FIFO identity survives a compression that moves the physical route to a child."""
import asyncio

import pytest


@pytest.mark.asyncio
async def test_native_fifo_identity_is_stable_across_compression(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority, SessionAuthority
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
    events = [MessageEvent(text=name, source=source, message_id=name) for name in ('first', 'second')]
    receipts = []
    for event in events:
        with native_callback(runner, event, get_hermes_home()):
            receipts.append(await authority.admit_native(event))
    key = runner.session_store._generate_session_key(source)
    parent = runner.session_store.peek_session_id(key)
    assert receipts[0].ref.session_id == parent == receipts[1].ref.session_id

    # Compression (the real publication path) moves the route to a child before the drain runs.
    db = authority.db
    db.append_message(parent, 'user', 'first')
    db.append_message(parent, 'assistant', 'ok')
    child = parent + '-child'
    db.publish_compression_child(parent_session_id=parent, child_session_id=child, source='discord',
                                 messages=[{'role': 'user', 'content': 'summary'}],
                                 require_compression_lease=False)
    assert runner.session_store.advance_compression_session(key, parent, child) is not None
    assert runner.session_store.peek_session_id(key) == child

    executed = []
    async def handle(event):
        executed.append(event.text)
        # The turn's execution callbacks must still reach the logical owner's stream.
        from gateway.run_turn_runner import TurnRunner
        from gateway.turn_context import TurnContext
        ctx = TurnContext(session_id=runner.session_store.peek_session_id(key), session_key=key, source=source)
        assert TurnRunner(runner, ctx)._approval_owner[1] == parent
        return 'ok'
    monkeypatch.setattr(runner, '_handle_message', handle)
    ref = receipts[0].ref
    await SessionAuthority._drain(authority, ref)
    assert executed == ['first', 'second'], executed
    rows = list_session_admissions(db, session_id=parent, pending_only=False)
    assert [(r['request_id'], r['status']) for r in rows] == [('first', 'terminal'), ('second', 'terminal')]

    # An exact retry of the first event is the same admission, and a third follows the same FIFO.
    with native_callback(runner, events[0], get_hermes_home()):
        assert (await authority.admit_native(events[0])).admission_id == receipts[0].admission_id
    third = MessageEvent(text='third', source=source, message_id='third')
    with native_callback(runner, third, get_hermes_home()):
        receipt = await authority.admit_native(third)
    assert receipt.ref.session_id == parent
    await SessionAuthority._drain(authority, ref)
    assert executed == ['first', 'second', 'third']
    await asyncio.sleep(0)
