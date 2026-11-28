"""Regression: an @mention of a *linked device* arrives as a bare UUID, not as the phone number.

A linked device has no Signal identity of its own — it shares the owner's number — so a modern
client sends only the ACI (UUID) inside ``dataMessage.mentions`` and leaves ``number`` empty. The
adapter caches ``number → UUID`` from the account's own envelopes (``_recipient_uuid_by_number``),
and the mention gate has to consult that cache; comparing the mention UUID against the phone
number alone can never match.

These tests drive ``_handle_envelope`` end to end and assert what actually reaches
``handle_message`` — no re-implementation of the comparison inside the test.
"""

from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig

BOT_NUMBER = "+15550001111"
BOT_UUID = "3eb65f7a-6a00-4263-a72c-280a262aa28b"
MEMBER_NUMBER = "+15550002222"
MEMBER_UUID = "b2187174-cd8e-4bb5-9f7f-dd6d6890d9e7"
OTHER_UUID = "8f14e45f-ceea-467a-9d3b-2e0e90a2b0f1"
GROUP_ID = "grp-linked-device-test"
PLACEHOLDER = "\ufffc"  # Signal's mention placeholder in the message text


def make_adapter(monkeypatch):
    """Adapter with require_mention on and exactly one allowed group, no network client."""
    from plugins.platforms.signal.adapter import SignalAdapter

    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", GROUP_ID)
    adapter = SignalAdapter(PlatformConfig(
        enabled=True,
        extra={"http_url": "http://127.0.0.1:8090", "account": BOT_NUMBER,
               "require_mention": True}))
    adapter.handle_message = AsyncMock()
    return adapter


def own_envelope(timestamp: int, text: str = "note to self") -> dict:
    """What signal-cli emits for a message this account sent: source == destination == own number."""
    return {"envelope": {
        "sourceNumber": BOT_NUMBER, "sourceUuid": BOT_UUID, "timestamp": timestamp,
        "syncMessage": {"sentMessage": {"destinationNumber": BOT_NUMBER, "timestamp": timestamp,
                                        "message": text}}}}


def group_envelope(mentions: list, text: str = f"{PLACEHOLDER} hallo") -> dict:
    """A group message from another member."""
    return {"envelope": {
        "sourceNumber": MEMBER_NUMBER, "sourceUuid": MEMBER_UUID, "timestamp": 2,
        "dataMessage": {"message": text, "mentions": mentions,
                        "groupInfo": {"groupId": GROUP_ID, "groupName": "R T F"}}}}


async def seed_own_mapping(adapter) -> None:
    """Seed number→UUID the way production does: from the account's own envelope."""
    assert BOT_NUMBER not in adapter._recipient_uuid_by_number
    await adapter._handle_envelope(own_envelope(1))
    assert adapter._recipient_uuid_by_number[BOT_NUMBER] == BOT_UUID
    adapter.handle_message.reset_mock()


@pytest.mark.asyncio
async def test_own_envelope_seeds_the_number_to_uuid_mapping(monkeypatch):
    adapter = make_adapter(monkeypatch)
    await seed_own_mapping(adapter)


@pytest.mark.asyncio
async def test_uuid_only_mention_reaches_handle_message(monkeypatch):
    """The reported bug: mention metadata carries the ACI only — it must still count as a mention."""
    adapter = make_adapter(monkeypatch)
    await seed_own_mapping(adapter)

    await adapter._handle_envelope(group_envelope([{"start": 0, "length": 1, "uuid": BOT_UUID}]))

    assert adapter.handle_message.await_count == 1, "UUID-only @mention of the linked device was dropped"
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "hallo", "the bot's own mention must be stripped before the agent sees it"
    assert event.source.chat_type == "group"


@pytest.mark.asyncio
async def test_uuid_only_mention_without_cached_mapping_is_filtered(monkeypatch):
    """Pins the mechanism: without the cached UUID the same envelope is (correctly) not a mention."""
    adapter = make_adapter(monkeypatch)

    await adapter._handle_envelope(group_envelope([{"start": 0, "length": 1, "uuid": BOT_UUID}]))

    assert adapter.handle_message.await_count == 0


@pytest.mark.asyncio
async def test_mention_of_another_uuid_is_filtered(monkeypatch):
    """Someone else was @-mentioned — the bot stays silent."""
    adapter = make_adapter(monkeypatch)
    await seed_own_mapping(adapter)

    await adapter._handle_envelope(group_envelope([{"start": 0, "length": 1, "uuid": OTHER_UUID}]))

    assert adapter.handle_message.await_count == 0


@pytest.mark.asyncio
async def test_phone_number_mention_still_works(monkeypatch):
    """The original path must keep working for clients that do send the number."""
    adapter = make_adapter(monkeypatch)
    mention = {"start": 0, "length": 1, "number": BOT_NUMBER, "uuid": BOT_UUID}

    await adapter._handle_envelope(group_envelope([mention]))

    assert adapter.handle_message.await_count == 1
    assert adapter.handle_message.await_args.args[0].text == "hallo"


@pytest.mark.asyncio
async def test_group_message_without_mention_is_filtered(monkeypatch):
    """require_mention still means what it says."""
    adapter = make_adapter(monkeypatch)
    await seed_own_mapping(adapter)

    await adapter._handle_envelope(group_envelope([], text="kein mention hier"))

    assert adapter.handle_message.await_count == 0
