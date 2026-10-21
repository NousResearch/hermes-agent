"""A message whose first token is a filesystem path is text, even when the adapter typed it COMMAND
on its leading ``/``: native admission queues it and replays it as text (#141221)."""
import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session_authority import initialize_session_authority
from gateway.session_envelope import restore_native
from gateway.session_ingress import admit_message
from hermes_state_runtime import list_session_admissions

CHAT = '15551234567@s.whatsapp.net'


@pytest.mark.asyncio
async def test_path_text_typed_command_is_admitted_as_text(tmp_path, monkeypatch):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
    monkeypatch.setenv('WHATSAPP_ALLOWED_USERS', '15551234567')
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={'session_path': str(tmp_path / 'wa')}))
    adapter.gateway_runner, runner.adapters = runner, {Platform.WHATSAPP: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='path')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id=CHAT, user_id=CHAT, chat_type='dm')
    text = '/Users/me/notes.md is broken'
    try:
        receipt = await authority.admit_native(MessageEvent(text=text, source=source, message_id='p1',
                                                            message_type=MessageType.COMMAND))
        row, = list_session_admissions(authority.db, session_id=receipt.ref.session_id, pending_only=False)
        replay = restore_native(row['payload'], runner)
        assert (replay.text, replay.message_type, replay.get_command()) == (text, MessageType.TEXT, None)
        # A real slash command still never enters the FIFO as a turn.
        reply = await admit_message(authority, MessageEvent(text='/new', source=source, message_id='c1',
                                                            message_type=MessageType.COMMAND))
        assert 'invalid_params' in reply
    finally:
        authority.db.close()
