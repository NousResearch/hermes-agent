"""Under the session authority a busy bare "yes" still answers a blocking approval (#46866)."""
import asyncio

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session_authority import SessionAuthority, initialize_session_authority
from hermes_state_runtime import list_session_admissions

CHAT = '15551234567@s.whatsapp.net'


class _RunningAgent:
    def __init__(self):
        self.interrupts = []

    def interrupt(self, *args):
        self.interrupts.append(args)


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['interrupt', 'queue'])
async def test_busy_plaintext_yes_resolves_blocking_approval_not_interrupt_or_queue(tmp_path, monkeypatch, mode):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
    from tools.approval import _gateway_queues
    from tools.approval_gateway_wait import _ApprovalEntry
    monkeypatch.setenv('WHATSAPP_ALLOWED_USERS', '15551234567')
    # No drain: an admitted "yes" would then wait on a turn, which is exactly the regression.
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={'session_path': str(tmp_path / 'wa')}))
    adapter.gateway_runner, runner.adapters = runner, {Platform.WHATSAPP: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='approval')
    runner._wire_adapter_handlers(adapter)
    sent = []

    async def send(chat_id, content, **_):
        sent.append(content)
        return SendResult(success=True, message_id='s')

    monkeypatch.setattr(adapter, 'send', send)
    monkeypatch.setattr(runner, '_effective_busy_input_mode', lambda source: mode)
    source = adapter.build_source(chat_id=CHAT, user_id=CHAT, chat_type='dm')
    key = runner._session_key_for_source(source)
    agent = _RunningAgent()
    runner._session_state(key).turn.agent = agent
    entry = _ApprovalEntry({'command': 'rm -rf build'})
    _gateway_queues.setdefault(key, []).append(entry)
    try:
        handled = asyncio.ensure_future(runner._handle_message(MessageEvent(text='yes', source=source, message_id='y1')))
        done, _ = await asyncio.wait({handled}, timeout=10)
        handled.cancel()
        assert done, '"yes" was admitted as a turn behind the blocked approval'
        assert entry.event.is_set() and entry.result == 'once'
        assert agent.interrupts == []
        admitted = [row for sid in authority.sessions
                    for row in list_session_admissions(authority.db, session_id=sid, pending_only=False)]
        assert admitted == [], admitted
    finally:
        _gateway_queues.pop(key, None)
        authority.db.close()
