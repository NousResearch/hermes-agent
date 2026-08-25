"""BlueBubbles integration contracts for the shared Tapback model."""

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.bluebubbles import BlueBubblesAdapter


def _adapter(monkeypatch):
    monkeypatch.setenv("BLUEBUBBLES_SERVER_URL", "http://localhost:1234")
    monkeypatch.setenv("BLUEBUBBLES_PASSWORD", "secret")
    return BlueBubblesAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "server_url": "http://localhost:1234",
                "password": "secret",
                "message_revision_wait_seconds": 0,
            },
        )
    )


@pytest.mark.asyncio
async def test_outbound_reaction_uses_exact_resolved_chat_and_target(monkeypatch):
    adapter = _adapter(monkeypatch)
    adapter.client = object()
    adapter._private_api_enabled = True
    adapter._helper_connected = True
    posts = []

    async def resolve(_chat_id):
        return "iMessage;+;exact-family-guid"

    async def exact(chat_id, message_id, **_kwargs):
        assert (chat_id, message_id) == (
            "iMessage;+;exact-family-guid",
            "target-message-guid",
        )
        return {"guid": message_id, "chats": [{"guid": chat_id}]}

    async def post(path, payload):
        posts.append((path, payload))
        return {"data": {"guid": "reaction-guid"}}

    monkeypatch.setattr(adapter, "_resolve_chat_guid", resolve)
    monkeypatch.setattr(adapter, "_query_exact_message", exact)
    monkeypatch.setattr(adapter, "_api_post", post)

    result = await adapter.send_reaction(
        "family-alias",
        "target-message-guid",
        "-like",
        part_index=2,
        source_event_id="approved-request-id",
    )

    assert result.success is True
    assert posts == [
        (
            "/api/v1/message/react",
            {
                "chatGuid": "iMessage;+;exact-family-guid",
                "selectedMessageGuid": "target-message-guid",
                "reaction": "-like",
                "partIndex": 2,
            },
        )
    ]


@pytest.mark.asyncio
async def test_inbound_shared_model_is_idempotent_and_chat_local(monkeypatch):
    adapter = _adapter(monkeypatch)
    events = []

    async def capture(event, _source):
        events.append(event)

    adapter.set_platform_event_handler(capture)
    common = {
        "payload": {"type": "updated-message"},
        "tapback": ("added", "like"),
        "target_guid": "reused-target-guid",
        "part_index": 0,
        "chat_identifier": "family",
        "sender": "sender@example.com",
        "is_group": True,
        "message_guid": "tapback-source-guid",
    }

    await adapter._dispatch_inbound_tapback(
        **common,
        session_chat_id="iMessage;+;first-chat",
    )
    await adapter._dispatch_inbound_tapback(
        **common,
        session_chat_id="iMessage;+;first-chat",
    )
    await adapter._dispatch_inbound_tapback(
        **common,
        session_chat_id="iMessage;+;second-chat",
    )

    assert [event["payload"]["chat_id"] for event in events] == [
        "iMessage;+;first-chat",
        "iMessage;+;second-chat",
    ]
    assert events[0]["payload"]["deduplication_key"] != events[1]["payload"]["deduplication_key"]
    assert all(event["payload"]["direction"] == "inbound" for event in events)
    assert all(event["payload"]["status"] == "processing" for event in events)


@pytest.mark.asyncio
async def test_inbound_add_remove_add_are_distinct_current_state_transitions(monkeypatch):
    adapter = _adapter(monkeypatch)
    events = []

    async def capture(event, _source):
        events.append(event)

    adapter.set_platform_event_handler(capture)
    common = {
        "payload": {},
        "target_guid": "target-guid",
        "part_index": 0,
        "session_chat_id": "iMessage;+;exact-chat",
        "chat_identifier": "family",
        "sender": "sender@example.com",
        "is_group": True,
        "message_guid": "provider-reused-guid",
    }

    for action in ("added", "removed", "added"):
        await adapter._dispatch_inbound_tapback(
            **common,
            tapback=(action, "love"),
        )

    assert [event["payload"]["action"] for event in events] == [
        "added",
        "removed",
        "added",
    ]


@pytest.mark.asyncio
async def test_inbound_tapback_without_source_event_fails_closed(monkeypatch):
    adapter = _adapter(monkeypatch)
    events = []

    async def capture(event, _source):
        events.append(event)

    adapter.set_platform_event_handler(capture)
    await adapter._dispatch_inbound_tapback(
        payload={},
        tapback=("added", "like"),
        target_guid="target-guid",
        part_index=0,
        session_chat_id="iMessage;+;exact-chat",
        chat_identifier="family",
        sender="sender@example.com",
        is_group=True,
        message_guid=None,
    )

    assert events == []
    assert adapter._tapback_event_states == {}
