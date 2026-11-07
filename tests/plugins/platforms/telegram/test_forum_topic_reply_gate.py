"""Opt-in topic-anchor classification through real Telegram messages/config (#118678)."""

import json
import os
from types import SimpleNamespace

import pytest

pytest.importorskip("telegram")
from telegram import Message

from gateway.config import Platform, load_gateway_config
from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.fixture
def adapter_factory(tmp_path, monkeypatch):
    # Never inherit the developer's real group gates or credentials.
    for key in tuple(os.environ):
        if key.startswith("TELEGRAM_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def make(enabled=None, *, nested=False, **overrides):
        extra = {} if enabled is None else {"ignore_forum_topic_anchors": enabled}
        settings = {
            "require_mention": True,
            "allowed_chats": ["-1001234567890"],
            "group_allowed_chats": ["-1001234567890"],
            "observe_unmentioned_group_messages": True,
            "extra": extra,
            **overrides,
        }
        raw = {"platforms": {"telegram": settings}} if nested else {"telegram": settings}
        (tmp_path / "config.yaml").write_text(json.dumps(raw))
        config = load_gateway_config().platforms[Platform.TELEGRAM]
        adapter = object.__new__(TelegramAdapter)
        adapter.platform = Platform.TELEGRAM
        adapter.config = config
        adapter._bot = SimpleNamespace(id=999, username="example_bot")
        adapter._mention_patterns = adapter._compile_mention_patterns()
        # Authorization is independent; these tests exercise an already-authorized sender.
        adapter._is_callback_user_authorized = lambda *args, **kwargs: True
        return adapter

    return make


def message(kind="anchor", *, chat_id=-1001234567890, mention=False, media=False):
    payload = {
        "message_id": 100,
        "date": 0,
        "chat": {"id": chat_id, "type": "supergroup", "is_forum": True},
        "from": {"id": 111, "is_bot": False, "first_name": "Example"},
        "message_thread_id": 73,
        "is_topic_message": True,
        "text": "@example_bot hello" if mention else "hello",
    }
    if mention:
        payload["entities"] = [{"type": "mention", "offset": 0, "length": 12}]
    if kind != "plain":
        payload["reply_to_message"] = {
            "message_id": 73 if kind == "anchor" else 80,
            "date": 0,
            "chat": payload["chat"],
            "from": {"id": 999, "is_bot": True, "first_name": "Example bot"},
        }
        if kind == "anchor":
            payload["reply_to_message"]["forum_topic_created"] = {"name": "Example", "icon_color": 0}
        else:
            payload["reply_to_message"]["text"] = "A real bot response"
    if media:
        payload["photo"] = [{"file_id": "fixture", "file_unique_id": "fixture", "width": 1, "height": 1}]
        payload["caption"] = payload.pop("text")
        if mention:
            payload["caption_entities"] = payload.pop("entities")
    return Message.de_json(payload, None)


@pytest.mark.parametrize("enabled", [None, False, True])
@pytest.mark.parametrize("media", [False, True])
def test_anchor_opt_in_preserves_real_replies_mentions_and_observation(adapter_factory, enabled, media):
    adapter = adapter_factory(enabled)
    anchor = message(media=media)
    assert adapter._should_process_message(anchor) is (enabled is not True)
    assert adapter._should_observe_unmentioned_group_message(anchor) is (enabled is True)
    assert adapter._should_process_message(message("reply", media=media)) is True
    assert adapter._should_process_message(message(mention=True, media=media)) is True
    assert adapter._should_process_message(message("plain", media=media)) is False


@pytest.mark.parametrize("overrides, kind, expected", [
    ({"guest_mode": True, "allowed_chats": ["-1009876543210"]}, "reply", False),
    ({"ignored_threads": ["73"]}, "reply", False),
    ({"require_mention": False}, "anchor", True),
    ({"free_response_chats": ["-1001234567890"]}, "anchor", True),
])
def test_opt_in_preserves_other_routing_gates(adapter_factory, overrides, kind, expected):
    adapter = adapter_factory(True, **overrides)
    assert adapter._should_process_message(message(kind)) is expected


@pytest.mark.parametrize("nested", [False, True])
def test_anchor_setting_is_adapter_local_and_does_not_reclassify_other_replies(adapter_factory, nested):
    opted_in = adapter_factory(True, nested=nested)
    legacy = adapter_factory(False, nested=nested)
    assert opted_in._should_process_message(message()) is False
    assert legacy._should_process_message(message()) is True
    # Equal IDs alone are insufficient: a real reply is not a creation service message.
    payload = message("reply").to_dict()
    payload["reply_to_message"]["message_id"] = 73
    assert opted_in._is_reply_to_bot(Message.de_json(payload, None)) is True
    # DMs retain their reply classification even for a service-shaped fixture.
    payload = message().to_dict()
    payload["chat"] = {"id": 111, "type": "private", "first_name": "Example"}
    assert opted_in._is_reply_to_bot(Message.de_json(payload, None)) is True
    assert opted_in._should_process_message(Message.de_json(payload, None)) is True
