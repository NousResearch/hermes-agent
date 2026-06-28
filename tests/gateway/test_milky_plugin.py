"""Tests for the bundled Milky QQ platform plugin."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session import SessionSource

from ._plugin_adapter_loader import load_plugin_adapter


milky_adapter = load_plugin_adapter("milky")
MilkyAdapter = milky_adapter.MilkyAdapter
_chat_route = milky_adapter._chat_route
_sanitize_outgoing_text = milky_adapter._sanitize_outgoing_text
media_kind = milky_adapter.media_kind


def _milky_adapter(extra=None):
    config = PlatformConfig(enabled=True, extra=extra or {})
    return MilkyAdapter(config)


def _milky_group_event(
    segments=None,
    *,
    group_id="67890",
    user_id="222",
    self_id="12345",
    message_seq=100,
    member_name="BobCard",
):
    return {
        "time": 1780000000,
        "self_id": self_id,
        "event_type": "message_receive",
        "data": {
            "message_scene": "group",
            "peer_id": int(group_id),
            "message_seq": int(message_seq),
            "sender_id": int(user_id),
            "time": 1780000000,
            "segments": segments or [{"type": "text", "data": {"text": "hello"}}],
            "group": {"group_id": int(group_id), "group_name": "Test Group"},
            "group_member": {
                "user_id": int(user_id),
                "nickname": "BobNick",
                "card": member_name,
            },
        },
    }


def _milky_private_event(text="你好", *, user_id="222", message_seq=200):
    return {
        "time": 1780000000,
        "self_id": "12345",
        "event_type": "message_receive",
        "data": {
            "message_scene": "friend",
            "peer_id": int(user_id),
            "message_seq": int(message_seq),
            "sender_id": int(user_id),
            "time": 1780000000,
            "segments": [{"type": "text", "data": {"text": text}}],
            "friend": {"user_id": int(user_id), "nickname": "Private Friend", "remark": ""},
        },
    }


def test_milky_chat_route_supports_group_and_private_prefixes():
    assert _chat_route("12345") == ("friend", "12345")
    assert _chat_route("group:67890") == ("group", "67890")
    assert _chat_route("g:67890") == ("group", "67890")
    assert _chat_route("private:12345") == ("friend", "12345")


def test_milky_send_group_uses_native_message_segments():
    adapter = _milky_adapter()
    adapter.session = SimpleNamespace(closed=False)
    adapter._api_post = AsyncMock(return_value={"status": "ok", "data": {"message_seq": 456}})

    result = asyncio.run(adapter.send("group:67890", "hello", reply_to="123"))

    assert result.success is True
    assert result.message_id == "456"
    action, payload = adapter._api_post.await_args.args
    assert action == "send_group_message"
    assert payload["group_id"] == 67890
    assert payload["message"] == [
        {"type": "reply", "data": {"message_seq": 123}},
        {"type": "text", "data": {"text": "hello"}},
    ]


def test_milky_media_kind_routes_images_voice_video_and_files():
    assert media_kind("/tmp/picture.png") == "image"
    assert media_kind("/tmp/clip.mp4") == "video"
    assert media_kind("/tmp/audio.wav") == "record"
    assert media_kind("/tmp/unknown.bin") == "file"
    assert media_kind("/tmp/document.pdf", is_voice=True) == "record"


def test_milky_sanitizes_internal_trigger_text():
    content = (
        "XIAOXING_AUTONOMY_TRIGGER\n"
        "判断：应该发送提醒\n\n"
        "爸爸：我知道啦，先不打扰。"
    )

    assert _sanitize_outgoing_text(content, {"xiaoxing_autonomy_trigger": True}) == "爸爸：我知道啦，先不打扰。"
    assert _sanitize_outgoing_text("判断：工具调用", {"xiaoxing_autonomy_trigger": True}) == "[SILENT]"


def test_milky_send_drops_silent_marker():
    adapter = _milky_adapter()
    adapter._api_post = AsyncMock()

    result = asyncio.run(adapter.send("group:67890", "[SILENT]"))

    assert result.success is True
    assert result.message_id == "skip"
    adapter._api_post.assert_not_called()


def test_milky_send_extracts_inline_image_media(tmp_path):
    image_path = tmp_path / "result.png"
    image_path.write_bytes(b"png")
    adapter = _milky_adapter()
    adapter._api_post = AsyncMock(
        side_effect=[
            {"status": "ok", "data": {"message_seq": 456}},
            {"status": "ok", "data": {"message_seq": 457}},
        ]
    )

    result = asyncio.run(adapter.send("group:67890", f"做好了\nMEDIA:{image_path}"))

    assert result.success is True
    assert result.message_id == "457"
    assert adapter._api_post.await_count == 2
    text_action, text_payload = adapter._api_post.await_args_list[0].args
    image_action, image_payload = adapter._api_post.await_args_list[1].args
    assert text_action == "send_group_message"
    assert text_payload["message"] == [{"type": "text", "data": {"text": "做好了"}}]
    assert image_action == "send_group_message"
    assert image_payload["message"][0]["type"] == "image"
    assert image_payload["message"][0]["data"]["uri"].endswith("/result.png")


def test_milky_send_document_uses_file_segment(tmp_path):
    doc_path = tmp_path / "report.pdf"
    doc_path.write_bytes(b"pdf")
    adapter = _milky_adapter()
    adapter._api_post = AsyncMock(return_value={"status": "ok", "data": {"message_seq": 888}})

    result = asyncio.run(adapter.send_document("group:67890", str(doc_path)))

    assert result.success is True
    action, payload = adapter._api_post.await_args.args
    assert action == "send_group_message"
    assert payload["message"] == [
        {
            "type": "file",
            "data": {
                "file_name": "report.pdf",
                "uri": doc_path.as_uri(),
            },
        }
    ]


def test_milky_send_remembers_outbound_for_later_group_reply_context():
    adapter = _milky_adapter({"group_policy": "ambient", "group_allow_from": ["67890"]})
    adapter._api_post = AsyncMock(return_value={"status": "ok", "data": {"message_seq": 456}})
    adapter.handle_message = AsyncMock()
    adapter._message_handler = AsyncMock()

    result = asyncio.run(adapter.send("group:67890", "这是小星刚说的话"))
    assert result.success is True

    segments = [
        {"type": "reply", "data": {"message_seq": 456}},
        {"type": "text", "data": {"text": "我回复你这句"}},
    ]
    asyncio.run(adapter._handle_event(_milky_group_event(segments, message_seq=457, member_name="Dad")))

    event = adapter.handle_message.await_args.args[0]
    assert event.reply_to_message_id == "456"
    assert event.reply_to_text == "小星: 这是小星刚说的话"


def test_milky_private_event_preserves_identity_and_message_seq():
    adapter = _milky_adapter({"allow_all_users": True})
    adapter.handle_message = AsyncMock()
    adapter._message_handler = AsyncMock()

    asyncio.run(adapter._handle_event(_milky_private_event()))

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.source.chat_id == "222"
    assert event.source.chat_type == "dm"
    assert event.source.user_id == "222"
    assert event.source.user_name == "Private Friend"
    assert event.message_id == "200"
    assert event.text == "你好"


def test_milky_group_event_marks_bot_at_and_reply_context():
    adapter = _milky_adapter({"group_policy": "ambient", "group_allow_from": ["67890"]})
    adapter.handle_message = AsyncMock()
    adapter._message_handler = AsyncMock()
    segments = [
        {
            "type": "reply",
            "data": {
                "message_seq": 99,
                "sender_id": 333,
                "sender_name": "Alice",
                "segments": [{"type": "text", "data": {"text": "原来那句话"}}],
            },
        },
        {"type": "mention", "data": {"user_id": 12345, "name": "小星"}},
        {"type": "text", "data": {"text": "你怎么看"}},
    ]

    asyncio.run(adapter._handle_event(_milky_group_event(segments)))

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.source.chat_id == "group:67890"
    assert event.source.chat_type == "group"
    assert event.source.user_id == "222"
    assert event.source.user_name == "BobCard"
    assert event.message_id == "100"
    assert event.text == "[@你] 你怎么看"
    assert event.reply_to_message_id == "99"
    assert event.reply_to_text == "Alice|QQ:333: 原来那句话"
    assert "明确 @ 了你" in event.channel_prompt
    assert "QQ reply" in event.channel_prompt


def test_milky_group_reply_prefers_recent_cache_with_qq_identity():
    adapter = _milky_adapter({"group_policy": "ambient", "group_allow_from": ["67890"]})
    adapter.handle_message = AsyncMock()
    adapter._message_handler = AsyncMock()

    asyncio.run(
        adapter._handle_event(
            _milky_group_event(
                [{"type": "text", "data": {"text": "原来那句话"}}],
                user_id="333",
                message_seq=99,
                member_name="李泽铭",
            )
        )
    )
    segments = [
        {"type": "reply", "data": {"message_seq": 99}},
        {"type": "text", "data": {"text": "确实是这个意思"}},
    ]
    asyncio.run(
        adapter._handle_event(
            _milky_group_event(
                segments,
                user_id="444",
                message_seq=100,
                member_name="王泽铭",
            )
        )
    )

    assert adapter.handle_message.await_count == 2
    event = adapter.handle_message.await_args.args[0]
    assert event.source.user_id == "444"
    assert event.reply_to_message_id == "99"
    assert event.reply_to_text == "李泽铭|QQ:333: 原来那句话"


def test_milky_group_event_preserves_other_at_as_silent_context():
    adapter = _milky_adapter({"group_policy": "ambient", "group_allow_from": ["67890"]})
    adapter.handle_message = AsyncMock()
    adapter._message_handler = AsyncMock()
    segments = [
        {"type": "mention", "data": {"user_id": 99999, "name": "Other"}},
        {"type": "text", "data": {"text": "你看看这个"}},
    ]

    asyncio.run(adapter._handle_event(_milky_group_event(segments)))

    event = adapter.handle_message.await_args.args[0]
    assert event.text == "[@QQ:99999] 你看看这个"
    assert "本条消息 @ 的不是你" in event.channel_prompt
    assert "本轮最终只输出 [SILENT]" in event.channel_prompt


def test_gateway_authorizes_milky_group_by_config_chat_allowlist():
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform("milky"): PlatformConfig(
                enabled=True,
                extra={"group_policy": "ambient", "group_allow_from": ["group:67890"]},
            )
        }
    )
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_args: False)
    source = SessionSource(
        platform=Platform("milky"),
        chat_id="group:67890",
        chat_type="group",
        user_id="unlisted-user",
    )

    assert runner._is_user_authorized(source) is True


def test_gateway_authorizes_milky_private_by_config_allow_all():
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform("milky"): PlatformConfig(
                enabled=True,
                extra={"allowed_users": "owner", "allow_all_users": True},
            )
        }
    )
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_args: False)
    source = SessionSource(
        platform=Platform("milky"),
        chat_id="new-friend",
        chat_type="dm",
        user_id="new-friend",
    )

    assert runner._is_user_authorized(source) is True
