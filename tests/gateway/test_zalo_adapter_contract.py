from gateway.config import PlatformConfig
from tests.gateway._plugin_adapter_loader import load_plugin_adapter


_zalo = load_plugin_adapter("zalo")
ZaloAdapter = _zalo.ZaloAdapter


def test_get_updates_accepts_documented_single_result_envelope():
    adapter = ZaloAdapter(PlatformConfig(enabled=True, extra={"bot_token": "token"}))
    event = {
        "event_name": _zalo.EVENT_TEXT,
        "message": {
            "from": {"id": "user-1", "display_name": "Ted", "is_bot": False},
            "chat": {"id": "chat-1", "chat_type": "PRIVATE"},
            "text": "Xin chào",
            "message_id": "message-1",
            "date": 1750316131602,
        },
    }

    assert adapter._extract_updates({"ok": True, "result": event}) == [event]


def test_get_updates_accepts_batched_result_without_guessing_other_envelopes():
    adapter = ZaloAdapter(PlatformConfig(enabled=True, extra={"bot_token": "token"}))
    events = [
        {"event_name": _zalo.EVENT_TEXT, "message": {"message_id": "1"}},
        {"event_name": _zalo.EVENT_IMAGE, "message": {"message_id": "2"}},
    ]

    assert adapter._extract_updates({"ok": True, "result": events}) == events
    assert adapter._extract_updates({"events": events}) == []
