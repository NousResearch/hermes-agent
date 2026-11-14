"""A Slack thread reply's watermark write never blocks the gateway event loop on storage.

Every ordinary reply in a Slack thread persists the thread's watermark (session metadata, so a
restart can rehydrate the replies it missed). That write is a full routing-index save
(``replace_gateway_routing_entries`` + the fsynced ``sessions.json`` mirror) and ran synchronously
on the event loop inside the adapter's message handler, holding ``SessionStore._lock`` across the
storage I/O, so every loop-side reader of that lock waited for it too.

Real SlackAdapter ``_handle_slack_message`` + real SessionStore/SessionDB; only the store's write
choke point is slowed, and the stall is measured as loop-tick lag.
"""
import asyncio
import threading
import time
from unittest.mock import AsyncMock, patch

import pytest

from tests.gateway.test_slack_ignore_other_user_mentions import CHANNEL_ID, _redirect_cache, adapter

SLOW_S = 2.0
MAX_LAG_S = 0.2
THREAD = '1700000000.000010'
__all__ = ['_redirect_cache', 'adapter']  # imported pytest fixtures (``_redirect_cache`` is autouse)


@pytest.mark.asyncio
async def test_slack_thread_watermark_write_keeps_the_loop_ticking(adapter, tmp_path, monkeypatch):  # health: allow F811 -- pytest injects the imported adapter fixture by parameter name
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from hermes_state import SessionDB
    monkeypatch.setattr('hermes_state.DEFAULT_DB_PATH', tmp_path / 'state.db')
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    adapter.set_session_store(store)
    source = adapter._thread_session_source(CHANNEL_ID, THREAD, 'U_HUMAN', '', 'group')
    key = adapter._build_thread_session_key(CHANNEL_ID, THREAD, 'U_HUMAN')  # non-DM threads key as 'group'
    await asyncio.to_thread(store.get_or_create_session, source)  # the thread already has a session
    adapter._thread_rehydration_checked.add(adapter._thread_rehydration_key(CHANNEL_ID, THREAD, 'U_HUMAN', ''))
    adapter._mentioned_threads.add(THREAD)  # the bot is part of this thread
    write, slow, writes = SessionDB._execute_write, threading.Event(), []

    def slow_write(self, *args, **kwargs):
        if slow.is_set():
            writes.append(threading.get_ident())
            time.sleep(SLOW_S)  # health: allow HX013 -- the slow storage write IS the stall under test
        return write(self, *args, **kwargs)
    monkeypatch.setattr(SessionDB, '_execute_write', slow_write)
    gaps = []

    async def tick():
        last = time.monotonic()
        while True:
            await asyncio.sleep(0.02)  # health: allow HX013 -- the ticker's period is the event-loop stall measurement
            now = time.monotonic()
            gaps.append(now - last)
            last = now
    event = {'channel': CHANNEL_ID, 'channel_type': 'channel', 'user': 'U_HUMAN', 'text': 'follow-up',
             'ts': '1700000000.000020', 'thread_ts': THREAD}
    started = time.monotonic()
    slow.set()
    ticker = asyncio.create_task(tick())
    await asyncio.sleep(0.05)  # health: allow HX013 -- the ticker is running before the handler starts
    try:
        with patch.object(adapter, '_resolve_user_name', new=AsyncMock(return_value='human')), \
                patch.object(adapter, '_fetch_thread_context', new=AsyncMock(return_value=None)), \
                patch.object(adapter, '_fetch_thread_parent_text', new=AsyncMock(return_value='')):
            await asyncio.wait_for(adapter._handle_slack_message(event), 30)
            await asyncio.sleep(0.1)  # health: allow HX013 -- one more tick after the handler returns
    finally:
        slow.clear()
        ticker.cancel()
        await asyncio.to_thread(store.close_all_db_handles)
    adapter.handle_message.assert_awaited_once()
    assert writes, 'the thread watermark was not persisted'
    assert store.get_session_metadata(key, f'slack_thread_watermark:{CHANNEL_ID}:{THREAD}') == event['ts']
    lag = max(gaps) if gaps else time.monotonic() - started
    assert lag < MAX_LAG_S, f'event loop stalled {lag:.2f}s behind the Slack thread watermark write'


@pytest.mark.asyncio
async def test_slack_assistant_lifecycle_seed_writes_off_the_loop_thread(adapter, tmp_path, monkeypatch):  # health: allow F811 -- pytest injects the imported adapter fixture by parameter name
    """``assistant_thread_started`` / ``app_home_opened`` seed a DM route (a routing-index write)."""
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    monkeypatch.setattr('hermes_state.DEFAULT_DB_PATH', tmp_path / 'state.db')
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    adapter.set_session_store(store)
    adapter._app.client.assistant_threads_setSuggestedPrompts = AsyncMock()
    real, threads = store.get_or_create_session, []

    def recorded(*args, **kwargs):
        threads.append(threading.get_ident())
        return real(*args, **kwargs)
    monkeypatch.setattr(store, 'get_or_create_session', recorded)
    try:
        await adapter._handle_assistant_thread_lifecycle_event({
            'type': 'assistant_thread_started', 'team_id': 'T1',
            'assistant_thread': {'channel_id': 'D1', 'thread_ts': '171.000', 'user_id': 'U1'}})
        await adapter._handle_app_home_opened({'type': 'app_home_opened', 'tab': 'messages', 'team': 'T1',
                                               'channel': 'D1', 'user': 'U1'})
    finally:
        await asyncio.to_thread(store.close_all_db_handles)
    assert len(threads) == 2, threads
    assert threading.get_ident() not in threads, 'a Slack lifecycle seed wrote the routing index on the loop thread'
