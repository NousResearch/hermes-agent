"""An unmentioned contact card in an observed group keeps its contents in the transcript."""

import asyncio
from types import SimpleNamespace

from tests.gateway.test_telegram_group_gating import _FakeSessionStore, _group_message, _make_adapter


def test_observed_contact_card_is_saved_with_its_rendered_contents():
    adapter = _make_adapter(require_mention=True, allowed_chats=["-100"], group_allowed_chats=["-100"],
                            observe_unmentioned_group_messages=True)
    store = _FakeSessionStore()
    adapter._session_store = store
    msg = _group_message(text=None)
    msg.contact = SimpleNamespace(first_name="Ada", last_name="Lovelace", phone_number="+15550001234",
                                  user_id=None, vcard=None)

    asyncio.run(adapter._handle_contact_message(SimpleNamespace(update_id=7, message=msg, effective_message=None),
                                                None))

    adapter._message_handler.assert_not_awaited()
    (_sid, message, _skip_db), = store.messages
    assert message["observed"] is True
    assert "+15550001234" in message["content"] and "Ada Lovelace" in message["content"]
