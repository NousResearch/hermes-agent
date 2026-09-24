"""Reused durable source cancellation before lock/slot and after real SDK dispatch."""
import asyncio
import importlib.util
from pathlib import Path
import sys
from unittest.mock import MagicMock

import pytest

# gateway/conftest installs a stand-in before collection. These cases require
# the installed SDK, not that optional-dependency mock; no host module loaded yet.
for name in tuple(sys.modules):
    if (name == 'telegram' or name.startswith('telegram.')) and isinstance(sys.modules[name], MagicMock):
        del sys.modules[name]

spec = importlib.util.spec_from_file_location('repair_sdk_harness', Path(__file__).parents[1]/'plugins/test_kanban_cards_transport.py')
assert spec and spec.loader
sdk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sdk)
h = sdk.h
rig = h.rig


async def until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.002)
    await asyncio.wait_for(wait(), 8)


@pytest.mark.asyncio
@pytest.mark.parametrize('pause', ['lock', 'slot', 'receipt'])
async def test_reused_source_cancel_has_only_current_attempt_evidence(rig, monkeypatch, pause):
    monkeypatch.setattr(sdk.adapter_module, 'resolve_proxy_url', lambda *a, **k: None)
    monkeypatch.setattr(sdk.telegram_network, '_resolve_proxy_url', lambda *a, **k: None)
    monkeypatch.setenv('HERMES_TELEGRAM_DISABLE_FALLBACK_IPS', 'true')
    first = sdk.Socket('receipt')
    second = sdk.Socket('receipt')
    sockets = []
    async def connect(*args, **kwargs):
        sock = first if not sockets else second if len(sockets) == 1 else sdk.Socket()
        sockets.append(sock)
        return sock.reader, sock
    monkeypatch.setattr(sdk.wire.asyncio, 'open_connection', connect)
    general, updates = await rig.adapter._build_ptb_requests()
    rig.adapter._bot = sdk.adapter_module.Application.builder().token('123:synthetic').request(general).get_updates_request(updates).build().bot
    held, release = asyncio.Event(), asyncio.Event()
    blocker = None
    try:
        await h.tick(rig, False)
        await asyncio.wait_for(first.entered.wait(), 8)
        source = next(iter(rig.manager._task_card_registration.sources))
        old = h.receipt(rig)
        new = h.advance(rig)
        await h.tick(rig, False)
        if pause == 'lock':
            async def hold():
                async with rig.adapter._chat_send_lock('-100'):
                    held.set()
                    await release.wait()
            blocker = asyncio.create_task(hold())
            await asyncio.sleep(0)
        elif pause == 'slot':
            # Clock boundary only: a real shared slot remains unavailable.
            rig.adapter._telegram_chat_outbound_slot_until['-100'] = asyncio.get_running_loop().time() + 60
        first.respond()
        if pause == 'lock':
            await asyncio.wait_for(held.wait(), 8)
        await until(lambda: h.receipt(rig).attempt_count == 2)
        assert source.lease is not None and source.lease.desired_revision == new
        if pause == 'receipt':
            await asyncio.wait_for(second.entered.wait(), 8)
        else:
            assert source.last_outcome is None
        await source.finish()
        row = h.receipt(rig)
        data = [b''.join(s.writes) for s in sockets if s.writes]
        assert row.state == 'unknown' and row.destination_message_id == '701'
        assert row.desired_revision == new and row.delivered_revision == old.attempted_revision
        assert row.failure_count == 1 and not source.admitted()
        assert sum(b'/editMessageText' in p for p in data) == int(pause == 'receipt')
        assert source.last_outcome is not None and source.last_outcome.status.value == 'unknown'
        release.set()
        if blocker: await blocker
        rig.adapter._telegram_chat_outbound_slot_until['-100'] = 0
        # Recover using the installed consumer and a new source, same known ID.
        if pause != 'receipt': second.respond()
        await h.tick(rig)
        row = h.receipt(rig)
        data = [b''.join(s.writes) for s in sockets if s.writes]
        assert row.state == 'sent' and row.delivered_revision == new and row.failure_count == 0
        assert sum(b'/sendMessage' in p for p in data) == 1
        assert b'/editMessageText' in data[-1] and b'message_id=701' in data[-1]
        assert not source.admitted()
    finally:
        release.set()
        if blocker: await blocker
        for sock in sockets: sock.release.set(); sock.respond()
        await general.shutdown()
        await updates.shutdown()
