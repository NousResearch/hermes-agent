import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType
from plugins.platforms.photon import adapter as photon_adapter
from plugins.platforms.photon.adapter import PhotonAdapter, _clean_imessage_body_text


@pytest.fixture
def adapter():
    return PhotonAdapter(PlatformConfig(enabled=True, extra={}))


def test_clean_imessage_body_text_removes_object_replacement_placeholder():
    placeholder = "\uFFFC"
    assert _clean_imessage_body_text(f"Here is the file {placeholder} ") == "Here is the file"
    assert _clean_imessage_body_text(f"before {placeholder} after") == "before after"
    assert _clean_imessage_body_text("plain text") == "plain text"


@pytest.mark.asyncio
async def test_group_text_placeholder_removed_while_attachment_metadata_remains(
    adapter, monkeypatch
):
    captured = []

    async def capture_message(event):
        captured.append(event)

    adapter.handle_message = capture_message
    monkeypatch.setattr(
        photon_adapter,
        "_cache_inbound_attachment",
        lambda content, name, mime, force_audio=False: "/tmp/hermes-photon/photo.jpg",
    )

    await adapter._dispatch_inbound(
        {
            "messageId": "msg-1",
            "space": {"id": "space-1", "type": "dm", "phone": "+15551234567"},
            "sender": {"id": "+15557654321"},
            "content": {
                "type": "group",
                "items": [
                    {"id": "text-1", "content": {"type": "text", "text": "Here is the file \uFFFC "}},
                    {
                        "id": "attach-1",
                        "content": {
                            "type": "attachment",
                            "id": "att-1",
                            "name": "photo.jpg",
                            "mimeType": "image/jpeg",
                            "size": 123,
                            "data": "ignored-by-test",
                            "encoding": "base64",
                        },
                    },
                ],
            },
            "timestamp": "2026-07-06T00:00:00.000Z",
        }
    )

    assert len(captured) == 1
    event = captured[0]
    assert event.text == "Here is the file"
    assert "\uFFFC" not in event.text
    assert event.message_type == MessageType.PHOTO
    assert event.media_urls == ["/tmp/hermes-photon/photo.jpg"]
    assert event.media_types == ["image/jpeg"]


@pytest.mark.asyncio
async def test_reaction_target_text_placeholder_removed(adapter):
    adapter._sent_message_ids["target-1"] = 1.0
    captured = []

    async def capture_message(event):
        captured.append(event)

    adapter.handle_message = capture_message

    await adapter._dispatch_inbound(
        {
            "messageId": "reaction-1",
            "space": {"id": "space-1", "type": "dm", "phone": "+15551234567"},
            "sender": {"id": "+15557654321"},
            "content": {
                "type": "reaction",
                "emoji": "❤️",
                "targetMessageId": "target-1",
                "targetDirection": "outbound",
                "targetText": "Photo \uFFFC",
            },
            "timestamp": "2026-07-06T00:00:00.000Z",
        }
    )

    assert len(captured) == 1
    assert captured[0].reply_to_text == "Photo"
