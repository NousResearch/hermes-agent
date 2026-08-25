"""Inbound BlueBubbles Tapbacks stay local, idempotent, and non-conversational."""

import asyncio

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.bluebubbles import BlueBubblesAdapter
from gateway.reactions import TapbackAction, TapbackStatus


def _adapter(monkeypatch) -> BlueBubblesAdapter:
    monkeypatch.setenv("BLUEBUBBLES_SERVER_URL", "http://localhost:1234")
    monkeypatch.setenv("BLUEBUBBLES_PASSWORD", "secret")
    adapter = BlueBubblesAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "server_url": "http://localhost:1234",
                "password": "secret",
                "message_revision_wait_seconds": 0,
            },
        )
    )
    adapter.set_authorization_check(lambda _user_id, _chat_type, _chat_id: True)
    return adapter


def _payload(associated_type) -> dict:
    return {
        "type": "new-message",
        "data": {
            "guid": "reaction-event-guid",
            "text": "must not become a conversational message",
            "associatedMessageType": associated_type,
            "associatedMessageGuid": "p:0/target-message-guid",
            "handle": {"address": "sender@example.com"},
            "isFromMe": False,
            "isGroup": True,
            "chatGuid": "iMessage;+;exact-chat-guid",
            "chatIdentifier": "family",
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("associated_type", [2006, "3999", {"malformed": True}])
async def test_unsupported_or_malformed_reaction_never_becomes_a_message(
    monkeypatch, associated_type
):
    adapter = _adapter(monkeypatch)
    normal_messages = []
    platform_events = []
    outbound_reactions = []

    async def exact_membership(_message_guid, candidate_chat_guid):
        return candidate_chat_guid

    async def capture_message(event):
        normal_messages.append(event)

    async def capture_platform_event(event, source):
        platform_events.append((event, source))

    async def capture_outbound(*args, **kwargs):
        outbound_reactions.append((args, kwargs))
        raise AssertionError("inbound Tapbacks must never emit outbound Tapbacks")

    monkeypatch.setattr(adapter, "_verify_inbound_message_membership", exact_membership)
    monkeypatch.setattr(adapter, "handle_message", capture_message)
    monkeypatch.setattr(adapter, "send_reaction", capture_outbound)
    adapter.set_platform_event_handler(capture_platform_event)

    response = await adapter._handle_webhook(object(), _trusted_payload=_payload(associated_type))
    await asyncio.sleep(0)

    assert response.status == 200
    assert normal_messages == []
    assert platform_events == []
    assert outbound_reactions == []
    assert adapter._tapback_event_states == {}


@pytest.mark.asyncio
async def test_removal_from_one_sender_preserves_other_sender_and_never_sends_outbound(
    monkeypatch,
):
    adapter = _adapter(monkeypatch)
    platform_events = []
    outbound_reactions = []

    async def capture_platform_event(event, source):
        platform_events.append((event, source))

    async def capture_outbound(*args, **kwargs):
        outbound_reactions.append((args, kwargs))
        raise AssertionError("inbound Tapbacks must never emit outbound Tapbacks")

    adapter.set_platform_event_handler(capture_platform_event)
    monkeypatch.setattr(adapter, "send_reaction", capture_outbound)

    common = {
        "payload": {"type": "updated-message"},
        "target_guid": "target-message-guid",
        "part_index": 0,
        "session_chat_id": "iMessage;+;exact-chat-guid",
        "chat_identifier": "family",
        "is_group": True,
    }
    await adapter._dispatch_inbound_tapback(
        **common,
        tapback=("added", "like"),
        sender="first@example.com",
        message_guid="first-add-guid",
    )
    await adapter._dispatch_inbound_tapback(
        **common,
        tapback=("added", "love"),
        sender="second@example.com",
        message_guid="second-add-guid",
    )
    await adapter._dispatch_inbound_tapback(
        **common,
        tapback=("removed", "like"),
        sender="first@example.com",
        message_guid="first-remove-guid",
    )

    assert outbound_reactions == []
    assert [event["payload"]["action"] for event, _source in platform_events] == [
        "added",
        "added",
        "removed",
    ]
    assert len(adapter._tapback_event_states) == 2
    states = {
        operation.sender_id: operation
        for operation, _serial in adapter._tapback_event_states.values()
    }
    assert states["first@example.com"].action is TapbackAction.REMOVE
    assert states["first@example.com"].status is TapbackStatus.APPLIED
    assert states["second@example.com"].action is TapbackAction.ADD
    assert states["second@example.com"].status is TapbackStatus.APPLIED


@pytest.mark.asyncio
async def test_out_of_order_replay_and_overlapping_participants_stay_chat_local(monkeypatch):
    adapter = _adapter(monkeypatch)
    platform_events = []
    outbound_reactions = []

    async def capture_platform_event(event, source):
        platform_events.append((event, source))

    async def capture_outbound(*args, **kwargs):
        outbound_reactions.append((args, kwargs))
        raise AssertionError("inbound Tapbacks must never emit outbound Tapbacks")

    adapter.set_platform_event_handler(capture_platform_event)
    monkeypatch.setattr(adapter, "send_reaction", capture_outbound)
    common = {
        "payload": {"type": "updated-message"},
        "target_guid": "reused-target-guid",
        "part_index": 0,
        "chat_identifier": "same-visible-name",
        "sender": "shared-participant@example.com",
        "is_group": True,
    }

    # A stale add replay after a newer removal is accepted as a state transition,
    # but its duplicate transport replay is suppressed in that exact chat only.
    for chat_id, action, source_id in (
        ("iMessage;+;chat-a", "added", "source-add"),
        ("iMessage;+;chat-a", "removed", "source-remove"),
        ("iMessage;+;chat-a", "added", "source-add"),
        ("iMessage;+;chat-a", "added", "source-add"),
        ("iMessage;+;chat-b", "added", "source-add"),
    ):
        await adapter._dispatch_inbound_tapback(
            **common,
            tapback=(action, "like"),
            session_chat_id=chat_id,
            message_guid=source_id,
        )

    assert outbound_reactions == []
    assert [event["payload"]["chat_id"] for event, _source in platform_events] == [
        "iMessage;+;chat-a",
        "iMessage;+;chat-a",
        "iMessage;+;chat-a",
        "iMessage;+;chat-b",
    ]
    assert [event["payload"]["action"] for event, _source in platform_events] == [
        "added",
        "removed",
        "added",
        "added",
    ]
    assert len(adapter._tapback_event_states) == 2
