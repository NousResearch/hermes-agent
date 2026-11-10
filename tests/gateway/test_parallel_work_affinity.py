"""Reply affinity, custody refusal and cold recovery for native parallel work."""
import asyncio

import pytest

from gateway.session import SessionEntry
from gateway.session_envelope import check_native_route, restore_native
from tests.gateway.test_parallel_native_ingress import admissions, event, setup_owner
from tests.waiting import async_wait_until


@pytest.mark.asyncio
async def test_followups_queue_only_on_selected_work_and_can_continue_final(monkeypatch):
    entered = []
    gates = {name: asyncio.Event() for name in ('a', 'b')}

    async def produce(authority, ref, row):
        text = row['payload']['text']
        entered.append((text, ref.session_id))
        if text in gates:
            await gates[text].wait()
        return text.upper()

    _runner, authority, adapter = await setup_owner(monkeypatch, produce)
    try:
        for mid, text in [('m1', 'a'), ('m2', 'b')]:
            await adapter.handle_message(event(adapter, mid, text))
        await async_wait_until(lambda: len(entered) == 2, what='two live native works')
        await adapter.handle_message(event(adapter, 'm3', 'a followup', reply='m1'))
        await async_wait_until(lambda: len(admissions(authority)) == 3, what='followup committed')
        assert len(entered) == 2
        assert next(row for row in admissions(authority) if row['request_id'] == 'm3')['target_session_id'] == entered[0][1]
        gates['b'].set()
        await async_wait_until(lambda: 'B' in adapter.sent, what='sibling unaffected')
        gates['a'].set()
        await async_wait_until(lambda: 'A FOLLOWUP' in adapter.sent, what='followup answered')
        final = next(mid for _, text, _, mid in adapter.deliveries if text == 'B')
        await adapter.handle_message(event(adapter, 'm4', 'continue b', reply=final))
        await async_wait_until(lambda: 'CONTINUE B' in adapter.sent, what='final anchor continued')
        assert dict(entered)['continue b'] == dict(entered)['b']
    finally:
        for gate in gates.values():
            gate.set()
        await async_wait_until(lambda: not adapter._background_tasks, what='delivery cleanup')
        authority.db.close()


@pytest.mark.asyncio
async def test_capacity_refuses_before_custody_and_ambiguous_controls_are_inert(monkeypatch):
    gate = asyncio.Event()

    async def produce(authority, ref, row):
        await gate.wait()
        return row['payload']['text'].upper()

    _runner, authority, adapter = await setup_owner(monkeypatch, produce)
    try:
        for mid, text in [('m1', 'a'), ('m2', 'b')]:
            await adapter.handle_message(event(adapter, mid, text))
        await async_wait_until(lambda: len(admissions(authority)) == 2, what='both admitted')
        for mid, text in [('m3', '/stop'), ('m4', 'yes'), ('m5', '/new')]:
            await adapter.handle_message(event(adapter, mid, text))
        assert len(admissions(authority)) == 2
        assert all(row['status'] == 'started' for row in admissions(authority))
        assert sum('Reply to the request' in text for text in adapter.sent) == 3
        await adapter.handle_message(event(adapter, 'm6', 'overflow'))
        await async_wait_until(lambda: any('send this again' in text for text in adapter.sent), what='explicit custody refusal')
        assert len(admissions(authority)) == 2
        with authority.db._read_ctx() as conn:
            assert conn.execute('SELECT COUNT(*) FROM parallel_work').fetchone()[0] == 2
    finally:
        gate.set()
        await async_wait_until(lambda: not adapter._background_tasks, what='delivery cleanup')
        authority.db.close()


@pytest.mark.asyncio
async def test_other_sender_or_room_cannot_target_work(monkeypatch):
    gate = asyncio.Event()

    async def produce(authority, ref, row):
        await gate.wait()
        return 'OK'

    _runner, authority, adapter = await setup_owner(monkeypatch, produce)
    try:
        await adapter.handle_message(event(adapter, 'm1', 'a'))
        await async_wait_until(lambda: len(admissions(authority)) == 1, what='work admitted')
        await adapter.handle_message(event(adapter, 'm2', '/stop', reply='m1', user='bob'))
        assert any('originating sender' in text for text in adapter.sent)
        await adapter.handle_message(event(adapter, 'm3', 'continue', reply='m1', chat='elsewhere'))
        assert any('Reply to the request' in text for text in adapter.sent)
        assert len(admissions(authority)) == 1
        assert admissions(authority)[0]['status'] == 'started'
    finally:
        gate.set()
        await async_wait_until(lambda: not adapter._background_tasks, what='delivery cleanup')
        authority.db.close()


@pytest.mark.asyncio
async def test_private_route_survives_native_codec_and_cold_entry_recovery(monkeypatch):
    from gateway.session_authority import SessionAuthority
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)

    async def produce(*args):
        raise AssertionError('recovery must not execute unknown work')

    runner, authority, adapter = await setup_owner(monkeypatch, produce)
    try:
        incoming = event(adapter, 'm1', 'a')
        from gateway.session_parallel import route_input
        assert await route_input(runner, incoming) is None
        receipt = await authority.admit_native(incoming)
        row = admissions(authority)[0]
        restored = restore_native(row['payload'], runner)
        route = runner._session_key_for_source(restored.source)
        assert route == runner._session_key_for_source(incoming.source)
        assert restored.source.user_id == 'alice' and restored.source.chat_id == 'room'
        entry = runner.session_store.lookup_by_session_key(route)
        cold = SessionEntry.from_dict(entry.to_dict())
        assert cold.origin.parallel_work is None
        available = runner._restored_source(cold)
        assert runner._session_key_for_source(available) == route
        await check_native_route(runner, row['payload'], receipt.ref.session_id, available, adapter)
        authority.db._execute_write(lambda conn: conn.execute(
            "UPDATE session_admissions SET status='unknown' WHERE admission_id=?", (receipt.admission_id,)))
        authority.sessions.clear()
        result = await authority.recover_native_sessions([(receipt.ref.session_id, available, adapter)])
        assert result == {receipt.ref.session_id: 'unknown_execution'}
    finally:
        authority.db.close()
