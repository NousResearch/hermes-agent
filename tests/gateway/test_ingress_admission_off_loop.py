"""Ingress admissions commit off the owner loop: a SQLite writer held by another process must
not freeze every session, timer and socket while a producer waits for its ACK (andrexibiza-21,
JoaoMarcos44-R7). One row per ingress admission site this lane owns (see SWEEP_owner-loop-writes)."""
import asyncio
import sqlite3
import threading
import time

import pytest


async def _msgraph(tmp_path):
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.msgraph_webhook import MSGraphWebhookAdapter
    from gateway.run import GatewayRunner
    from gateway.session_authority import SessionAuthority, initialize_session_authority
    runner = GatewayRunner()
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='off-loop')
    authority._schedule = lambda ref: None  # the admission write is the subject, not the turn
    adapter = MSGraphWebhookAdapter(PlatformConfig(enabled=True, extra={
        'host': '127.0.0.1', 'client_state': 'owned', 'accepted_resources': ['users/owned']}))
    runner.adapters[Platform.MSGRAPH_WEBHOOK] = adapter
    runner._wire_adapter_handlers(adapter)
    event = adapter._build_message_event({'id': 'n1', 'subscriptionId': 's', 'resource': 'users/owned/x'}, 'id:n1')
    assert isinstance(authority, SessionAuthority)
    return authority, lambda: adapter._admit_notification(event)


@pytest.mark.asyncio
@pytest.mark.parametrize('site', [_msgraph], ids=['msgraph_webhook._admit_notification'])
async def test_ingress_admission_never_blocks_the_owner_loop_on_a_held_writer(tmp_path, monkeypatch, site):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    authority, admit = await site(tmp_path)
    held, release = threading.Event(), threading.Event()

    def hold_writer():
        conn = sqlite3.connect(authority.db.db_path, timeout=30)
        conn.execute('BEGIN IMMEDIATE')
        held.set()
        release.wait(30)
        conn.rollback()
        conn.close()
    holder = threading.Thread(target=hold_writer, daemon=True)
    holder.start()
    assert await asyncio.to_thread(held.wait, 10)
    try:
        task = asyncio.create_task(admit())
        ticks, start = 0, time.monotonic()
        while time.monotonic() - start < 1.0:
            await asyncio.sleep(0.05)
            ticks += 1
        # A loop parked in a blocking BEGIN IMMEDIATE runs no ticks until the writer lets go.
        assert ticks >= 10, f'owner loop stalled behind the held writer ({ticks} ticks in 1s)'
        assert not task.done(), 'the admission must wait for the writer, not skip the commit'
    finally:
        release.set()
    receipt = await asyncio.wait_for(task, 30)
    assert receipt.status == 'queued'  # committed before the ACK, as before
    await asyncio.to_thread(holder.join, 10)
    await asyncio.to_thread(authority.db.close)
