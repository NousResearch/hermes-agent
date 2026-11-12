"""Producers that pin epoch 0 for a stable retry fingerprint still record local time (not 1970)."""
from datetime import UTC, datetime

import pytest

from hermes_agent.gateway.config import GatewayConfig, Platform, PlatformConfig
from hermes_agent.gateway.platforms.base import SendResult
from hermes_agent.gateway.platforms.event import MessageEvent
from hermes_agent.gateway.run import GatewayRunner
from hermes_agent.gateway.session_authority import initialize_session_authority
from hermes_agent.gateway.session_ingress import admit_message
from hermes_agent.gateway.session_results import execution_result

CHAT = '15551234567@s.whatsapp.net'


@pytest.mark.asyncio
async def test_epoch_zero_producer_timestamp_runs_with_local_time_and_keeps_retry_identity(tmp_path, monkeypatch):
    from hermes_agent.plugins.platforms.whatsapp.adapter import WhatsAppAdapter
    monkeypatch.setenv('WHATSAPP_ALLOWED_USERS', '15551234567')
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={'session_path': str(tmp_path / 'wa')}))
    adapter.gateway_runner, runner.adapters = runner, {Platform.WHATSAPP: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='stamp')
    runner._wire_adapter_handlers(adapter)
    stamps = []

    async def send(chat_id, content, **_):
        return SendResult(success=True, message_id='s')

    async def turn(event, *_args, **_kwargs):
        stamps.append(event.timestamp)
        execution_result.get()['result'] = {'final_response': 'ok', 'messages': []}
        return 'ok'

    monkeypatch.setattr(adapter, 'send', send)
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
    source = adapter.build_source(chat_id=CHAT, user_id=CHAT, chat_type='dm')
    pinned = lambda: MessageEvent(text='hi', source=source, message_id='p1',
                                  timestamp=datetime.fromtimestamp(0, UTC))
    try:
        before = datetime.now()
        assert await admit_message(authority, pinned()) == 'ok'
        assert len(stamps) == 1 and stamps[0].timestamp() >= before.timestamp(), stamps  # was 1970-01-01 (persisted as 0)
        # The redelivered provider id is still the same admission: no second turn.
        first = await authority.admit_native(pinned())
        assert first.status == 'terminal' and len(stamps) == 1
    finally:
        authority.db.close()
