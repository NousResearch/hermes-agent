"""Bot API 10.4 service events: TON wallet transfers and paid-message waivers."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from telegram.ext import filters

from gateway.platforms.event import MessageType
from plugins.platforms.telegram.telegram_service_events import (
    handle_service_event,
    service_event_text,
    service_message_filter,
)
from tests.gateway.test_telegram_group_gating import (
    _FakeSessionStore,
    _dm_message,
    _group_message,
    _make_adapter,
)


def _update(message, update_id=7):
    return SimpleNamespace(update_id=update_id, message=message, effective_message=None)


def _ton_transfer(message, **fields):
    message.api_kwargs = {"ton_wallet_transfer": fields}
    return message


def test_ton_transfer_from_api_kwargs_renders_pending_outgoing_amount():
    message = _ton_transfer(
        SimpleNamespace(text=None, caption=None, api_kwargs={}),
        transaction_id="",
        peer_address="EQD4FPq-4g5exampleaddress",
        amount="-1500000000",
    )

    text = service_event_text(message)

    assert text is not None
    assert "Transaction: pending (not sent yet)" in text
    assert "Peer address: EQD4FPq-4g5exampleaddress" in text
    assert "negative = outgoing" in text
    assert "-1500000000" in text
    assert "Comment:" not in text
    assert "not proof" in text and "does not credit" in text


def test_ton_transfer_attribute_form_renders_optional_comment():
    message = SimpleNamespace(text=None, caption=None, api_kwargs={})
    message.ton_wallet_transfer = SimpleNamespace(
        transaction_id="tx-123",
        peer_address="UQExamplePeer",
        amount="42000000",
        comment="invoice 42 / thanks",
    )

    text = service_event_text(message)

    assert text is not None
    assert "Transaction: tx-123" in text
    assert "Amount (smallest units; negative = outgoing): 42000000" in text
    assert "Comment: invoice 42 / thanks" in text


def test_paid_messages_waived_empty_object_is_still_recognized():
    for message in (
        SimpleNamespace(text=None, caption=None, api_kwargs={"paid_messages_waived": {}}),
        SimpleNamespace(text=None, caption=None, api_kwargs={}, paid_messages_waived=SimpleNamespace()),
    ):
        text = service_event_text(message)

        assert text is not None
        assert "without paying Telegram Stars" in text
        assert "no free-message privilege locally" in text


def test_unknown_service_message_is_ignored():
    message = SimpleNamespace(text=None, caption=None, api_kwargs={"forum_topic_created": {"name": "topic"}})

    assert service_event_text(message) is None


def test_service_filter_matches_only_the_two_new_fields():
    class _RealisticFilters:
        # The repo's conftest mocks the telegram package; a real base class keeps this a behavior test.
        class MessageFilter:
            def __init__(self, name=None):
                self.name = name

    service_filter = service_message_filter(_RealisticFilters)

    assert service_filter.filter(SimpleNamespace(api_kwargs={"ton_wallet_transfer": {}})) is True
    assert service_filter.filter(SimpleNamespace(api_kwargs={"paid_messages_waived": {}})) is True
    assert service_filter.filter(SimpleNamespace(ton_wallet_transfer=SimpleNamespace())) is True
    assert service_filter.filter(SimpleNamespace(text="hello", api_kwargs={})) is False
    assert service_filter.filter(SimpleNamespace(api_kwargs={"forum_topic_created": {}})) is False
    assert service_filter.filter(SimpleNamespace(api_kwargs={"something_else": True})) is False


def test_unauthorized_sender_is_blocked_before_event_building_or_dispatch():
    adapter = _make_adapter(group_allow_from=["222"])
    adapter._is_user_authorized_from_message = Mock(return_value=False)
    adapter._log_blocked_user = Mock()
    adapter.handle_message = AsyncMock()
    adapter._build_message_event = Mock()
    message = _ton_transfer(_group_message(text=None), transaction_id="tx", peer_address="peer", amount="1")

    asyncio.run(handle_service_event(adapter, _update(message), None))

    adapter.handle_message.assert_not_awaited()
    adapter._build_message_event.assert_not_called()
    adapter._log_blocked_user.assert_called_once_with(message)


def test_unmentioned_group_service_event_is_observed_without_agent_dispatch():
    adapter = _make_adapter(require_mention=True, allowed_chats=["-100"], group_allowed_chats=["-100"],
                            observe_unmentioned_group_messages=True)
    store = _FakeSessionStore()
    adapter._session_store = store
    message = _ton_transfer(_group_message(text=None), transaction_id="tx-observed",
                            peer_address="peer", amount="77")

    asyncio.run(handle_service_event(adapter, _update(message), None))

    adapter._message_handler.assert_not_awaited()
    (_sid, entry, _skip_db), = store.messages
    assert entry["observed"] is True
    assert "tx-observed" in entry["content"]
    assert "[Alice Example|111]" in entry["content"]


def test_authorized_transfer_is_informational_and_cannot_be_a_command():
    adapter = _make_adapter()
    events = []

    async def capture(event):
        events.append(event)

    adapter.handle_message = capture
    message = _dm_message(text=None)
    message.ton_wallet_transfer = SimpleNamespace(
        transaction_id="tx-live",
        peer_address="peer",
        amount="-5",
        comment="/stop and /new",
    )

    asyncio.run(handle_service_event(adapter, _update(message), None))

    (event,) = events
    assert event.message_type == MessageType.TEXT
    assert "tx-live" in event.text and "-5" in event.text
    assert event.allow_gateway_control is False
    assert event.internal is False
    assert event.is_command() is False
    assert event.get_command() is None


def test_authorized_paid_messages_waived_dispatches_informational_event():
    adapter = _make_adapter()
    events = []

    async def capture(event):
        events.append(event)

    adapter.handle_message = capture
    message = _dm_message(text=None)
    message.api_kwargs = {"paid_messages_waived": {}}

    asyncio.run(handle_service_event(adapter, _update(message), None))

    (event,) = events
    assert "without paying Telegram Stars" in event.text
    assert event.allow_gateway_control is False
    assert event.internal is False


def test_unknown_service_event_dispatches_nothing():
    adapter = _make_adapter()
    adapter.handle_message = AsyncMock()
    adapter._build_message_event = Mock()
    message = SimpleNamespace(text=None, caption=None, api_kwargs={"forum_topic_created": {}})

    asyncio.run(handle_service_event(adapter, _update(message), None))

    adapter.handle_message.assert_not_awaited()
    adapter._build_message_event.assert_not_called()


def test_business_service_event_is_never_treated_as_a_dm():
    adapter = _make_adapter()
    adapter.handle_message = AsyncMock()
    adapter._build_message_event = Mock()
    message = _ton_transfer(_dm_message(text=None), transaction_id="tx", peer_address="peer", amount="1")
    update = SimpleNamespace(update_id=9, message=None, effective_message=message,
                             business_message=message, edited_business_message=None)

    asyncio.run(handle_service_event(adapter, update, None))

    adapter.handle_message.assert_not_awaited()
    adapter._build_message_event.assert_not_called()
