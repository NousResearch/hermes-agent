from types import SimpleNamespace

from gateway.platforms.base import MessageEvent, MessageType, merge_pending_message_event
from gateway.session import SessionSource
from run_agent import AIAgent


def test_string_persistence_marker_replaces_multimodal_content():
    agent = object.__new__(AIAgent)
    agent._persist_user_message_idx = 0
    agent._persist_user_message_override = "[private inbound]"
    agent._persist_user_message_timestamp = None
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "private caption"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,secret"}},
    ]}]

    agent._apply_persist_user_message_override(messages)

    assert messages == [{"role": "user", "content": "[private inbound]"}]


def test_message_event_positional_metadata_and_timestamp_compatibility():
    source = SessionSource(platform=None, chat_id="chat")
    metadata = {"legacy": True}
    timestamp = SimpleNamespace()
    values = ["text", MessageType.TEXT, source, None, "id", None, [], [], None,
              None, None, None, False, None, None, None, False, metadata, timestamp]

    event = MessageEvent(*values)

    assert event.metadata is metadata
    assert event.timestamp is timestamp
    assert event.delivery_mode is None


def test_pending_merge_does_not_contaminate_distinct_dispatch_contracts():
    source = SessionSource(platform=None, chat_id="chat")
    first = MessageEvent(
        "private", MessageType.PHOTO, source, media_urls=["a.png"],
        delivery_mode="suppress", persist_user_message="[private]",
        gateway_dispatch_applied=True,
    )
    second = MessageEvent(
        "public", MessageType.PHOTO, source, media_urls=["b.png"],
        gateway_dispatch_applied=True,
    )
    pending = {"key": first}

    merge_pending_message_event(pending, "key", second)

    assert pending["key"] is second
    assert second.text == "public"
    assert second.media_urls == ["b.png"]