"""An edited message re-sends its message id with new text: native admission takes it as its own
input, never as a changed replay of the original send (``admission_conflict``, #140845)."""
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session_authority import initialize_session_authority
from hermes_state_runtime import list_session_admissions

CHAT = '15551234567@s.whatsapp.net'


@pytest.mark.asyncio
async def test_edit_of_an_admitted_message_is_its_own_admission(tmp_path, monkeypatch):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
    monkeypatch.setenv('WHATSAPP_ALLOWED_USERS', '15551234567')
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={'session_path': str(tmp_path / 'wa')}))
    adapter.gateway_runner, runner.adapters = runner, {Platform.WHATSAPP: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='edit')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id=CHAT, user_id=CHAT, chat_type='dm')
    sent = datetime(2026, 10, 18, 2, 5, tzinfo=UTC)  # an edit keeps its message's send time

    def send(text, revision=None):
        return authority.admit_native(MessageEvent(text=text, source=source, message_id='m1', timestamp=sent,
                                                   edit_revision=revision))
    try:
        first = await send('helo')
        edit = await send('hello', '1800000010')
        # A redelivered edit is still a replay of that edit, and a second edit is a third input.
        again = await send('hello', '1800000010')
        second = await send('hello!', '1800000020')
        assert again.admission_id == edit.admission_id
        assert len({first.admission_id, edit.admission_id, second.admission_id}) == 3
        rows = list_session_admissions(authority.db, session_id=first.ref.session_id, pending_only=False)
        assert [row['payload']['text'] for row in rows] == ['helo', 'hello', 'hello!']
    finally:
        authority.db.close()


def test_telegram_edit_carries_its_edit_date_as_revision():
    from plugins.platforms.telegram.adapter import TelegramAdapter
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token='***', extra={})
    sent = datetime(2026, 10, 18, 2, 5, tzinfo=UTC)

    def message(edit_date):
        return SimpleNamespace(
            message_id=7, text='hello', caption=None, entities=[], caption_entities=[], message_thread_id=None,
            is_topic_message=False, reply_to_message=None, date=sent, edit_date=edit_date, forum_topic_created=None,
            chat=SimpleNamespace(id=111, type='private', full_name='A', title=None, is_forum=False),
            from_user=SimpleNamespace(id=111, full_name='A', first_name='A', is_bot=False))

    original = adapter._build_message_event(message(None), MessageType.TEXT)
    edited = adapter._build_message_event(message(datetime(2026, 10, 18, 2, 6, tzinfo=UTC)), MessageType.TEXT)
    assert original.edit_revision is None
    assert edited.message_id == original.message_id == '7'
    assert edited.edit_revision == str(int(datetime(2026, 10, 18, 2, 6, tzinfo=UTC).timestamp()))
