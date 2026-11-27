"""/stop (and /new: the same hook) reaches a messaging input whose admission passed the route fence and is
still waiting on the SQLite writer: it commits before /stop lists the queue, so /stop cancels it and it never
runs after the Stop."""
import asyncio
import sqlite3
import threading

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session_authority import SessionAuthority, initialize_session_authority
from persistence_assembly import close_stores_under, store_location
from tests.waiting import async_wait_until

X = '15551234567@s.whatsapp.net'


@pytest.mark.asyncio
async def test_stop_cancels_an_input_that_passed_the_fence_and_waits_on_the_writer(tmp_path, monkeypatch):
    from gateway.session_ingress import admit_message
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
    monkeypatch.setenv('WHATSAPP_ALLOWED_USERS', '15551234567')
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)  # no drain: rows stay as admitted
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={'session_path': str(tmp_path / 'wa')}))
    runner.adapters = {Platform.WHATSAPP: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='stop-wait')
    source = adapter.build_source(chat_id=X, user_id=X, chat_type='dm')
    route = runner._session_key_for_source(source)
    await authority.admit_native(MessageEvent(text='text x0', source=source, message_id='x0'))

    holder = sqlite3.connect(store_location(authority.persistence.scope), isolation_level=None, check_same_thread=False)
    passed = threading.Event()
    real_write, real_gate = SessionAuthority._admit_native_write, SessionAuthority.gate_refusal

    def write(self, **kwargs):
        if kwargs.get('request_id') == 'x1':
            holder.execute('BEGIN IMMEDIATE')  # another process holds the writer from here
        return real_write(self, **kwargs)

    def gate(self, *args, **kwargs):
        refusal = real_gate(self, *args, **kwargs)
        if holder.in_transaction:
            passed.set()
        return refusal
    monkeypatch.setattr(SessionAuthority, '_admit_native_write', write)
    monkeypatch.setattr(SessionAuthority, 'gate_refusal', gate)
    x1 = stop = None
    try:
        x1 = asyncio.create_task(admit_message(authority, MessageEvent(text='text x1', source=source,
                                                                        message_id='x1')))
        assert await asyncio.to_thread(passed.wait, 5), 'x1 never reached its admission gate'
        epoch = authority.route_epoch(route)
        stop = asyncio.create_task(runner._hm_stop_owner_turn(route))
        await async_wait_until(lambda: authority.route_epoch(route) == epoch + 1, what='the route fence moved')
        holder.execute('COMMIT')
        await asyncio.wait_for(stop, 10)
        assert await asyncio.wait_for(x1, 10) is None  # dropped silently, as an input the fence refused
        rows = {row.request_id: row.status for sid in authority.sessions
                for row in authority.persistence.admissions.list(session_id=sid, pending_only=False)}
        assert rows == {'x0': 'terminal', 'x1': 'terminal'}, rows
    finally:
        if holder.in_transaction:
            holder.execute('COMMIT')
        holder.close()
        for task in (x1, stop):
            if task is not None:
                task.cancel()
        close_stores_under(authority.persistence.scope.profile_dir)
