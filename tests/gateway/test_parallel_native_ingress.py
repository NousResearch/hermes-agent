"""Independent requests in one Discord room keep native single-writer ownership."""
import asyncio
from datetime import datetime

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session_authority import initialize_session_authority
from hermes_state_runtime import list_session_admissions
from tests.gateway.test_shared_busy_receive_loop import Adapter
from tests.waiting import async_wait_until


class DiscordFixture(Adapter):
    def __init__(self):
        super().__init__()
        self.platform = Platform.DISCORD
        self.deliveries = []

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        result = await super().send(chat_id, content, reply_to, metadata)
        self.deliveries.append((chat_id, content, reply_to, result.message_id))
        return result


async def setup_owner(monkeypatch, producer):
    import gateway.session_finite as finite
    monkeypatch.setenv('DISCORD_ALLOWED_USERS', 'alice,bob')
    monkeypatch.setattr(finite, 'execute_finite_admission', producer)
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordFixture()
    adapter.gateway_runner = runner
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='parallel')
    runner._wire_adapter_handlers(adapter)
    monkeypatch.setattr(runner, '_effective_busy_input_mode', lambda source: 'parallel')
    return runner, authority, adapter


def event(adapter, mid, text, *, reply=None, user='alice', chat='room'):
    source = adapter.build_source(chat_id=chat, chat_type='channel', user_id=user)
    return MessageEvent(text=text, source=source, message_id=mid, reply_to_message_id=reply,
                        timestamp=datetime(2026, 1, 1))


def admissions(authority):
    return [row for sid in authority.sessions
            for row in list_session_admissions(authority.db, session_id=sid, pending_only=False)]


@pytest.mark.asyncio
async def test_same_room_independent_requests_overlap_and_replay_once(monkeypatch):
    entered, releases = {}, {name: asyncio.Event() for name in ('one', 'two')}

    async def produce(authority, ref, row):
        name = row['payload']['text']
        entered[name] = ref.session_id
        await releases[name].wait()
        return name.upper()

    _runner, authority, adapter = await setup_owner(monkeypatch, produce)
    try:
        await adapter.handle_message(event(adapter, 'm1', 'one'))
        await async_wait_until(lambda: 'one' in entered, what='first work started')
        await adapter.handle_message(event(adapter, 'm2', 'two'))
        await async_wait_until(lambda: 'two' in entered, timeout=3, what='second work overlaps first')
        assert entered['one'] != entered['two']
        await adapter.handle_message(event(adapter, 'm1', 'one'))
        assert len(admissions(authority)) == 2
        releases['two'].set()
        await async_wait_until(lambda: 'TWO' in adapter.sent, what='second work replied first')
        assert 'ONE' not in adapter.sent
        releases['one'].set()
        await async_wait_until(lambda: not adapter._background_tasks, what='all native deliveries finished')
        assert [(content, anchor) for _, content, anchor, _ in adapter.deliveries] == [('TWO', 'm2'), ('ONE', 'm1')]
    finally:
        for gate in releases.values():
            gate.set()
        await async_wait_until(lambda: not adapter._background_tasks, what='adapter cleanup')
        authority.db.close()
