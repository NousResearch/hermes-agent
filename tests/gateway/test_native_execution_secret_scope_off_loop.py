"""Admitted native turns hydrate profile secret sources off the owner loop (main #99519 parity)."""
import threading

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner
from gateway.session_authority import initialize_session_authority
from gateway.session_results import execution_result
from tests.waiting import async_wait_until

CHAT = '15551234567@s.whatsapp.net'


@pytest.mark.asyncio
async def test_native_execution_and_settled_delivery_hydrate_secrets_off_loop(tmp_path, monkeypatch):
    from hermes_cli import env_loader
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
    monkeypatch.setenv('WHATSAPP_ALLOWED_USERS', '15551234567')
    hydrations = []
    real_hydrate = env_loader.hydrate_profile_secret_sources

    def hydrate(home, *args, **kwargs):
        # An external secret CLI (1Password, vault) blocks here; the turn's scope must never run it on
        # the loop thread. The authorization check's own scope is main's code (same on main).
        import traceback
        stack = ''.join(traceback.format_stack())
        if '_under_authorization_profile' not in stack:
            hydrations.append(threading.current_thread() is loop_thread)
        return real_hydrate(home, *args, **kwargs)

    monkeypatch.setattr(env_loader, 'hydrate_profile_secret_sources', hydrate)
    loop_thread = threading.current_thread()
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={
        'text_batch_delay_seconds': 0, 'session_path': str(tmp_path / 'wa')}))
    adapter.gateway_runner, runner.adapters = runner, {Platform.WHATSAPP: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='secrets')
    runner._wire_adapter_handlers(adapter)
    sent = []

    async def send(chat_id, content, **_):
        sent.append(content)
        return SendResult(success=True, message_id='s')

    async def turn(event, *_args, **_kwargs):
        execution_result.get()['result'] = {'final_response': 'ok', 'messages': []}
        return 'ok'

    monkeypatch.setattr(adapter, 'send', send)
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
    try:
        event = await adapter._build_message_event({'messageId': 'm1', 'chatId': CHAT, 'senderId': CHAT,
                                                    'body': 'hello', 'nativeType': 'conversation'})
        hydrations.clear()  # admission-side scope (_async_profile_runtime_scope) is already off-loop
        adapter._enqueue_text_event(event)
        await async_wait_until(lambda: sent and not adapter._background_tasks, timeout=20, what='the reply')
        assert sent == ['ok']
        assert hydrations and True not in hydrations, hydrations
    finally:
        authority.db.close()
